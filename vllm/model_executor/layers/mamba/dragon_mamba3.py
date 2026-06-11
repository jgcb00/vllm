# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dragon Mamba3 MIMO mixer for vLLM.

Native port of ``DragonMamba3MimoFast`` from ``modeling_dragon.py``. The
mixer holds the 4 temporal states (angle, ssm, k, v) in vLLM's paged mamba
state pool; per-request slots come from
``Mamba3AttentionMetadata.state_indices_tensor`` produced by the
``MAMBA3`` backend.

Prefill packs the whole batch into ``(B=1, S=total_tokens)`` and dispatches
to ``mamba_ssm.ops.tilelang.mamba3.mamba3_mimo`` with ``cu_seqlens`` (the
official varlen path). Decode uses the packed CuteDSL
``mamba3_step_fn`` over the full decode-token batch.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mamba3_attn import Mamba3AttentionMetadata

# Prefill-length bucketing (DEFAULT OFF). Historical: a local fork of the
# TileLang varlen prefill kernel specialized on S (total packed token count),
# so variable-length serving paid a fresh ~20-90s JIT compile per distinct S;
# padding S up to a bucket multiple bounded the compile count. The upstream
# kernel keeps S dynamic (one compile, no measurable perf difference vs
# padded buckets — verified 2026-06-11), so padding is pure wasted compute
# now. Set DRAGON_PREFILL_BUCKET=<n> to re-enable if running a static-S
# kernel build.
import os as _os
_PREFILL_BUCKET = int(_os.environ.get("DRAGON_PREFILL_BUCKET", "0"))

# Decode-step state access. The default path passes the state pools + slot
# indices straight into the CuteDSL step kernel (``state_indices=``), which
# reads/updates pool rows in place. The legacy path gathered all 4 state pools
# with ``pool[slots]`` and scattered them back every step — ~3x the necessary
# HBM traffic on the fp32 SSM state (measured ~58% of decode GPU time at
# batch 256). DRAGON_INDEXED_STEP=0 restores the gather/scatter path.
_INDEXED_STEP = _os.environ.get("DRAGON_INDEXED_STEP", "1") != "0"

# Fused decode preamble: the eager pre-kernel section of ``_decode`` ran
# ~14 tiny kernels per layer per step (x/z compaction copies, A/dt/trap
# softplus/clamp/sigmoid soup, B/C RMS norms + weight muls) — pure launch
# overhead at decode shapes. Two Triton kernels replace them.
# DRAGON_FUSED_PREAMBLE=0 restores the eager ops.
_FUSED_PREAMBLE = _os.environ.get("DRAGON_FUSED_PREAMBLE", "1") != "0"

# CuteDSL decode-step launch config. tile_D covers the full headdim (64), so
# exactly one CTA owns each (batch, head) row — the precondition for asking
# the kernel to store the new B/x key/value states itself
# (``update_kv_state``). Keep call sites and that decision on this single
# pair of constants.
_STEP_TILE_D = 64
_STEP_NUM_WARPS = 4

# External Mamba3 kernels (from the official ``mamba_ssm`` package). Imported
# lazily so that this module is importable on hosts without the kernels.
_mamba3_mimo = None
_mamba3_step_fn = None
_apply_rotary_qk_inference_fwd = None


@triton.jit
def _dragon_decode_preamble_kernel(
    zxdt_ptr,     # (N, H*(2D+3)) bf16 — in_proj output, head-interleaved
    x_ptr,        # out (N, H, D) contiguous, input dtype
    z_ptr,        # out (N, H, D) contiguous, input dtype
    a_ptr,        # out (N, H) fp32: clamp(-softplus(A), max=-a_floor)
    dt_ptr,       # out (N, H) fp32: softplus(dt + dt_bias[h])
    trap_ptr,     # out (N, H) fp32: sigmoid(trap)
    dt_bias_ptr,  # (H,)
    a_floor,
    H,
    stride_n,
    D: tl.constexpr,
):
    """One program per (token, head): replaces the x/z .contiguous() copies
    and the A/dt/trap softplus/clamp/sigmoid kernel soup of ``_decode``."""
    pid = tl.program_id(0)
    h = pid % H
    base = zxdt_ptr + (pid // H) * stride_n + h * (2 * D + 3)
    offs = tl.arange(0, D)  # D is a power of 2 (guarded at call site)
    tl.store(z_ptr + pid * D + offs, tl.load(base + offs))
    tl.store(x_ptr + pid * D + offs, tl.load(base + D + offs))
    dt = tl.load(base + 2 * D).to(tl.float32)
    a = tl.load(base + 2 * D + 1).to(tl.float32)
    trap = tl.load(base + 2 * D + 2).to(tl.float32)
    # F.softplus with its overflow threshold (x for x > 20)
    sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    v = dt + tl.load(dt_bias_ptr + h).to(tl.float32)
    sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
    tl.store(a_ptr + pid, tl.minimum(-sp_a, -a_floor))
    tl.store(dt_ptr + pid, sp_dt)
    tl.store(trap_ptr + pid, 1.0 / (1.0 + tl.exp(-trap)))


@triton.jit
def _dragon_bc_norm_kernel(
    bc_ptr,      # (N, 2*R*S + n_angles) — in_proj_dyn output (ngroups == 1)
    out_ptr,     # (N, 2, R, S) contiguous: [:,0]=B_norm(B), [:,1]=C_norm(C)
    wb_ptr,      # (S,) B_norm weight
    wc_ptr,      # (S,) C_norm weight
    eps,
    off_c,       # offset of the C block inside a bc row (= R*S)
    R,
    stride_n,
    ZERO_CENTERED: tl.constexpr,
    S: tl.constexpr,
):
    """One program per (token, B/C, mimo-row): fused RMS norm + weight."""
    pid = tl.program_id(0)  # n*(2R) + which*R + r
    j = pid % (2 * R)
    base = bc_ptr + (pid // (2 * R)) * stride_n + (j % R) * S + (j // R) * off_c
    offs = tl.arange(0, S)  # S is a power of 2 (guarded at call site)
    v = tl.load(base + offs).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(v * v) / S + eps)
    wb = tl.load(wb_ptr + offs).to(tl.float32)
    wc = tl.load(wc_ptr + offs).to(tl.float32)
    w = tl.where(j >= R, wc, wb)
    if ZERO_CENTERED:
        w = w + 1.0
    tl.store(out_ptr + pid * S + offs,
             (v * inv * w).to(out_ptr.dtype.element_ty))


def _lazy_import_kernels():
    global _mamba3_mimo, _mamba3_step_fn, _apply_rotary_qk_inference_fwd
    if _mamba3_mimo is not None:
        return
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd,
    )
    _mamba3_mimo = mamba3_mimo
    _mamba3_step_fn = mamba3_step_fn
    _apply_rotary_qk_inference_fwd = apply_rotary_qk_inference_fwd


class _DragonRMSNormInner(nn.Module):
    """Port of Dragon's ``DragonRMSNorm`` (optionally zero-centered).

    Matches the checkpoint layout: the weight is stored as ``self.weight``
    on this module, and the affine-free ``nn.RMSNorm`` is the unparameter-
    ised normaliser.
    """

    def __init__(self, hidden_size: int, eps: float, zero_centered: bool):
        super().__init__()
        self.rms = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        init = torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)
        self.zero_centered = zero_centered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rms(x)
        if self.zero_centered:
            return y * (1.0 + self.weight)
        return y * self.weight


class _DragonNorm(nn.Module):
    """Port of Dragon's ``DragonNorm`` wrapper (``self.norm = DragonRMSNorm``).

    Matches the checkpoint path ``<name>.norm.weight``.
    """

    def __init__(self, hidden_size: int, *, eps: float, zero_centered: bool):
        super().__init__()
        self.norm = _DragonRMSNormInner(
            hidden_size, eps=eps, zero_centered=zero_centered
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# Back-compat alias: older code in this module referred to ``_DragonRMSNorm``.
_DragonRMSNorm = _DragonNorm


class _DragonCosNetBranch(nn.Module):
    """Low-rank cos-modulated residual branch (mirrors DragonCosNetBranch)."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.omega = nn.Parameter(torch.ones(rank))
        self.phase = nn.Parameter(torch.zeros(rank))
        self.scale = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.A
        h = torch.cos(h * self.omega + self.phase)
        h = h @ self.B
        return h * self.scale


class _DragonLinear(nn.Linear):
    """Port of Dragon's ``DragonLinear``: ``nn.Linear`` + optional CosNet branch.

    Subclasses ``nn.Linear`` directly so that the checkpoint's
    ``<name>.weight`` / ``<name>.bias`` paths match without remapping. No TP
    sharding: the Dragon in-proj layouts are head-interleaved, which would
    need a custom weight loader. The initial native path runs tp_size == 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        cosnet: bool,
        cosnet_rank: int,
        bias: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.cosnet_branch = (
            _DragonCosNetBranch(in_features, out_features, cosnet_rank)
            if cosnet
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.cosnet_branch is not None:
            out = out + self.cosnet_branch(x)
        return out


class DragonMamba3Mixer(nn.Module, MambaBase):
    """vLLM-native wrapper around Dragon's Mamba3 MIMO mixer.

    State layout (per request slot):
        - angle_state : (nheads, num_rope_angles)    float32
        - ssm_state   : (nheads, headdim, d_state)   float32
        - k_state     : (mimo_dim, nheads, d_state)  state_dtype
        - v_state     : (nheads, headdim)            state_dtype
    """

    # populated by the model runner via MambaSpec after init
    kv_cache: tuple[torch.Tensor, ...]

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        tp = get_tensor_model_parallel_world_size()
        self.tp_size = tp
        if tp > 1 and getattr(config, "cosnet", False):
            raise NotImplementedError(
                "DragonMamba3Mixer: cosnet=True is not supported with "
                "tensor_parallel_size > 1 (would require a custom weight "
                "loader for the CosNet sidecar)."
            )

        # --- Dragon config-derived shape knobs (mirror DragonMamba3MimoFast)
        self.d_model = config.hidden_size
        self.d_inner = 2 * self.d_model
        self.mimo_dim = config.mamba_mimo_dim
        self.d_state = config.mamba_d_state
        self.headdim = config.mamba_headdim
        self.ngroups = config.mamba_ngroups
        self.nheads = self.d_inner // self.headdim
        self.nheads_local = divide(self.nheads, tp)
        self.rope_fraction = 0.5
        self.rotary_dim_divisor = 4
        self.A_floor = 1e-4
        split = int(self.d_state * self.rope_fraction)
        if split % 2:
            split -= 1
        self.num_rope_angles = split // 2
        self.chunk_size = 64 // self.mimo_dim

        # --- Projections ----------------------------------------------------
        # in_proj output is head-interleaved
        # ``[h0: z(headdim), x(headdim), dt, A, trap | h1: ... | ...]`` and has
        # total width ``nheads * (2*headdim + 3) = d_inner*2 + 3*nheads``.
        # Row-splitting the output axis by TP therefore gives a clean
        # per-head shard (``ColumnParallelLinear`` shards output axis).
        if tp > 1:
            assert self.nheads % tp == 0, (
                f"DragonMamba3Mixer: nheads={self.nheads} must be divisible "
                f"by tp_size={tp}"
            )
        self.in_proj = ColumnParallelLinear(
            input_size=self.d_model,
            output_size=self.d_inner * 2 + 3 * self.nheads,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.in_proj",
        )
        # in_proj_dyn produces B/C (per-group, shared across heads) and the
        # rope angles (shared across heads). Replicate across ranks.
        self.in_proj_dyn = ReplicatedLinear(
            input_size=self.d_model,
            output_size=(
                2 * self.ngroups * self.d_state * self.mimo_dim
                + self.num_rope_angles
            ),
            bias=False,
            prefix=f"{prefix}.in_proj_dyn",
        )

        self.B_bias = nn.Parameter(
            torch.ones(self.nheads_local, self.mimo_dim, self.d_state,
                       dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads_local, self.mimo_dim, self.d_state,
                       dtype=torch.float32)
        )
        set_weight_attrs(self.B_bias, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.C_bias, {"weight_loader": sharded_weight_loader(0)})

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        self.B_norm = _DragonRMSNorm(self.d_state, eps=eps, zero_centered=zc)
        self.C_norm = _DragonRMSNorm(self.d_state, eps=eps, zero_centered=zc)

        self.in_proj_mimo_x = nn.Parameter(
            torch.full((self.nheads_local, self.mimo_dim, self.headdim),
                       1.0 / self.mimo_dim, dtype=torch.float32)
        )
        self.in_proj_mimo_z = nn.Parameter(
            torch.ones(self.nheads_local, self.mimo_dim, self.headdim,
                       dtype=torch.float32)
        )
        self.out_proj_mimo = nn.Parameter(
            torch.full((self.nheads_local, self.mimo_dim, self.headdim),
                       1.0 / self.mimo_dim, dtype=torch.float32)
        )
        set_weight_attrs(
            self.in_proj_mimo_x, {"weight_loader": sharded_weight_loader(0)}
        )
        set_weight_attrs(
            self.in_proj_mimo_z, {"weight_loader": sharded_weight_loader(0)}
        )
        set_weight_attrs(
            self.out_proj_mimo, {"weight_loader": sharded_weight_loader(0)}
        )

        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.nheads_local)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.D = nn.Parameter(torch.ones(self.nheads_local))
        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})

        self._postgate_norm = getattr(config, "mamba3_postgate_norm", False)
        if self._postgate_norm:
            # Per-rank local-shard norm: each rank applies RMSNorm only over
            # its local d_inner slice. This differs numerically from a
            # global RMSNorm but matches how we split the mixer output.
            self.output_norm = _DragonRMSNorm(
                self.d_inner // tp, eps=eps, zero_centered=zc
            )
            set_weight_attrs(
                self.output_norm.norm.weight,
                {"weight_loader": sharded_weight_loader(0)},
            )

        # Register in the static forward-context table so the attention
        # runner can locate our metadata via self.prefix.
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    # -- MambaBase surface ---------------------------------------------------
    @property
    def mamba_type(self) -> str:
        return "mamba3"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.mamba3_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.mamba3_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.nheads,
            head_dim=self.headdim,
            d_state=self.d_state,
            mimo_dim=self.mimo_dim,
            num_rope_angles=self.num_rope_angles,
        )

    # -- Forward -------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: ``(num_tokens, hidden_size)``."""
        _lazy_import_kernels()

        fwd_ctx = get_forward_context()
        attn_md = fwd_ctx.attn_metadata
        d_inner_local = self.nheads_local * self.headdim
        if attn_md is None:
            # profile / dry run — return zeros with local (per-rank) d_inner.
            return torch.zeros(hidden_states.shape[0], d_inner_local,
                               device=hidden_states.device,
                               dtype=hidden_states.dtype)
        md: Mamba3AttentionMetadata = attn_md[self.prefix]

        angle_pool, ssm_pool, k_pool, v_pool = self.kv_cache
        slots = md.state_indices_tensor

        out = torch.empty(hidden_states.shape[0], d_inner_local,
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
        nd, np_ = md.num_decodes, md.num_prefills
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens

        if nd > 0:
            self._decode(
                hidden_states[:ndt],
                out[:ndt],
                slots[:nd],
                angle_pool, ssm_pool, k_pool, v_pool,
            )

        if np_ > 0:
            self._prefill(
                hidden_states[ndt:ndt + npt],
                out[ndt:ndt + npt],
                slots[nd:nd + np_],
                md.has_initial_state,
                md.query_start_loc_p,
                angle_pool, ssm_pool, k_pool, v_pool,
                has_initial_any=getattr(md, "has_initial_any", None),
                has_init_cpu=getattr(md, "has_initial_state_cpu", None),
                qsl_cpu=getattr(md, "query_start_loc_p_cpu", None),
            )

        return out

    # -- Prefill / decode ----------------------------------------------------
    def _project_in(self, h: torch.Tensor):
        """Run in_proj/in_proj_dyn on ``(N, D)`` and return the unpacked pieces."""
        zxdtAtrap, _ = self.in_proj(h)
        per_head = zxdtAtrap.view(h.shape[0], self.nheads_local, 2 * self.headdim + 3)
        z = per_head[..., 0:self.headdim]
        x = per_head[..., self.headdim:2 * self.headdim]
        dt = per_head[..., 2 * self.headdim]
        A = per_head[..., 2 * self.headdim + 1]
        trap = per_head[..., 2 * self.headdim + 2]

        bc, _ = self.in_proj_dyn(h)
        off = self.ngroups * self.mimo_dim * self.d_state
        B = bc[..., :off]
        C = bc[..., off:2 * off]
        angle = bc[..., 2 * off:]
        B = rearrange(B, "n (G r s) -> n r G s", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "n (G r s) -> n r G s", G=self.ngroups, r=self.mimo_dim)
        return z, x, dt, A, trap, B, C, angle

    def _prefill(
        self,
        h: torch.Tensor,        # (T, D)
        out: torch.Tensor,      # (T, D_inner_local)
        slots: torch.Tensor,    # (P,)
        has_initial: torch.Tensor | None,
        qsl: torch.Tensor,      # (P+1,) rebased to 0
        angle_pool: torch.Tensor,
        ssm_pool: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        *,
        has_initial_any: bool | None = None,
        has_init_cpu: torch.Tensor | None = None,
        qsl_cpu: torch.Tensor | None = None,
    ) -> None:
        # The metadata builder provides CPU mirrors (one sync per step);
        # fall back to syncing here if absent (29 M-layers -> 29 syncs).
        if has_initial_any is None:
            has_initial_any = (
                has_initial is not None and bool(has_initial.any().item())
            )
        # Fast path: no continuation chunks — run the packed varlen kernel once.
        if has_initial is None or not has_initial_any:
            self._prefill_zero_start(
                h, out, slots, qsl, angle_pool, ssm_pool, k_pool, v_pool,
            )
            return

        # Mixed case: split into zero-start (fresh prompts) and continuation
        # (chunk N>0 of a previously-prefilled prompt). The Mamba3 MIMO varlen
        # kernel does not accept an init state; continuation chunks are
        # processed by looping the decode step kernel across tokens (reads
        # and writes per-request paged state in place).
        if has_init_cpu is None:
            has_init_cpu = has_initial.detach().to("cpu", torch.bool)
        if qsl_cpu is None:
            qsl_cpu = qsl.detach().to("cpu")
        num_prefills = has_init_cpu.numel()

        zero_req_idxs: list[int] = [
            i for i in range(num_prefills) if not bool(has_init_cpu[i].item())
        ]
        cont_req_idxs: list[int] = [
            i for i in range(num_prefills) if bool(has_init_cpu[i].item())
        ]

        if zero_req_idxs:
            z_lens = [
                int(qsl_cpu[i + 1].item() - qsl_cpu[i].item())
                for i in zero_req_idxs
            ]
            z_h = torch.cat(
                [
                    h[int(qsl_cpu[i].item()):int(qsl_cpu[i + 1].item())]
                    for i in zero_req_idxs
                ],
                dim=0,
            )
            z_out = torch.empty(
                sum(z_lens), out.shape[1],
                device=out.device, dtype=out.dtype,
            )
            z_qsl = torch.tensor(
                [0] + list(itertools.accumulate(z_lens)),
                device=qsl.device, dtype=qsl.dtype,
            )
            z_slots = slots[torch.as_tensor(
                zero_req_idxs, device=slots.device, dtype=torch.long,
            )]
            self._prefill_zero_start(
                z_h, z_out, z_slots, z_qsl,
                angle_pool, ssm_pool, k_pool, v_pool,
            )
            cursor = 0
            for i, length in zip(zero_req_idxs, z_lens):
                start = int(qsl_cpu[i].item())
                out[start:start + length].copy_(z_out[cursor:cursor + length])
                cursor += length

        if cont_req_idxs:
            c_lens = [
                int(qsl_cpu[i + 1].item() - qsl_cpu[i].item())
                for i in cont_req_idxs
            ]
            c_starts = [int(qsl_cpu[i].item()) for i in cont_req_idxs]
            c_slots = slots[torch.as_tensor(
                cont_req_idxs, device=slots.device, dtype=torch.long,
            )]
            max_len = max(c_lens)
            for t in range(max_len):
                active = [k for k, L in enumerate(c_lens) if t < L]
                if not active:
                    break
                idxs = torch.as_tensor(
                    [c_starts[k] + t for k in active],
                    device=h.device, dtype=torch.long,
                )
                slots_t = c_slots[torch.as_tensor(
                    active, device=c_slots.device, dtype=torch.long,
                )]
                h_t = h.index_select(0, idxs)
                out_t = torch.empty(
                    len(active), out.shape[1],
                    device=out.device, dtype=out.dtype,
                )
                self._decode(
                    h_t, out_t, slots_t,
                    angle_pool, ssm_pool, k_pool, v_pool,
                )
                out.index_copy_(0, idxs, out_t)

    def _prefill_zero_start(
        self,
        h: torch.Tensor,        # (T, D)
        out: torch.Tensor,      # (T, D_inner_local)
        slots: torch.Tensor,    # (P,)
        qsl: torch.Tensor,      # (P+1,) rebased to 0
        angle_pool: torch.Tensor,
        ssm_pool: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
    ) -> None:
        # The varlen mamba3_mimo kernel ingests B=1, S=total_tokens packed
        # input + cu_seqlens for per-request boundaries. All requests in the
        # packed batch start from zero state.
        z, x, dt, A, trap, B, C, angle = self._project_in(h)
        z = z.unsqueeze(0)                                # (1, T, H, p)
        x = x.unsqueeze(0)                                # (1, T, H, p)
        dt = dt.unsqueeze(0).to(torch.float32)            # (1, T, H)
        A = A.unsqueeze(0)                                # (1, T, H)
        trap = trap.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1, H, T)
        B = B.unsqueeze(0)                                # (1, T, R, G, N)
        C = C.unsqueeze(0)                                # (1, T, R, G, N)
        # Angles per the official module: expand over heads, cast to fp32.
        # The kernel computes the angle*dt cumsum internally (Angles_Cumsum).
        angle = (
            angle.unsqueeze(0)
            .unsqueeze(-2)
            .expand(-1, -1, self.nheads_local, -1)
            .to(torch.float32)
            .contiguous()
        )

        B = self.B_norm(B)
        C = self.C_norm(C)

        _A = -F.softplus(A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dt + self.dt_bias)
        ADT = (_A * DT).permute(0, 2, 1).contiguous()    # (1, H, T)
        DT = DT.permute(0, 2, 1).contiguous()            # (1, H, T)

        # --- bucket the total packed length S to a fixed multiple so the
        # @tilelang.jit varlen kernel (specialized on S) only ever compiles a
        # handful of shapes (see _PREFILL_BUCKET note at module top). A dummy
        # trailing segment of zeros absorbs the padding; the P real sequences
        # are independent in the varlen kernel so their outputs/states are
        # unchanged.
        P = qsl.numel() - 1
        T = x.shape[1]
        real_last_idx = qsl[1:].long() - 1               # (P,) real seq ends
        bucket = _PREFILL_BUCKET
        T_pad = ((T + bucket - 1) // bucket) * bucket if bucket > 0 else T
        pad = T_pad - T
        if pad > 0:
            z = F.pad(z, (0, 0, 0, 0, 0, pad))           # (1,T,H,p)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))           # (1,T,H,p)
            B = F.pad(B, (0, 0, 0, 0, 0, 0, 0, pad))     # (1,T,R,G,N)
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad))     # (1,T,R,G,N)
            angle = F.pad(angle, (0, 0, 0, 0, 0, pad))   # (1,T,H,N)
            ADT = F.pad(ADT, (0, pad))                   # (1,H,T)
            DT = F.pad(DT, (0, pad))                     # (1,H,T)
            trap = F.pad(trap, (0, pad))                 # (1,H,T)
            qsl_k = torch.cat([qsl, qsl.new_tensor([T_pad])])  # +dummy segment
        else:
            qsl_k = qsl

        cu_seqlens = qsl_k.to(torch.int32)               # (P+1[+1],)

        # Constant fp32 weight casts, computed once on first prefill and
        # cached (was 6 cast kernels + allocations per layer per prefill).
        cw = getattr(self, "_prefill_const_w", None)
        if cw is None:
            cw = (
                self.C_bias.to(torch.float32),
                self.B_bias.to(torch.float32),
                self.in_proj_mimo_x.to(torch.float32),
                self.in_proj_mimo_z.to(torch.float32),
                self.out_proj_mimo.to(torch.float32),
                self.D.to(torch.float32),
            )
            self._prefill_const_w = cw
        q_bias_f32, k_bias_f32, mimo_v_f32, mimo_z_f32, mimo_out_f32, d_f32 = cw

        Out, Final_Angle, Final_SSM, Final_K, Final_V = _mamba3_mimo(
            Q=C.contiguous().bfloat16(),
            K=B.contiguous().bfloat16(),
            V=x.contiguous().bfloat16(),
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=q_bias_f32,
            K_bias=k_bias_f32,
            MIMO_V=mimo_v_f32,
            MIMO_Z=mimo_z_f32,
            MIMO_Out=mimo_out_f32,
            Angles=angle,
            D=d_f32,
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=True,
            cu_seqlens=cu_seqlens,
        )

        # Drop the dummy padding segment's final state (kept only to fix S);
        # keep the P real sequences. The kernel returns Final_V over the global
        # last token, so recompute per-sequence last tokens from the *real*
        # boundaries (real_last_idx, < T) regardless of padding.
        Final_Angle = Final_Angle[:P]
        Final_SSM = Final_SSM[:P]
        Final_K = Final_K[:P]
        Final_V = x[0, real_last_idx]                     # (P, H, p)

        angle_pool[slots] = Final_Angle.to(angle_pool.dtype)
        ssm_pool[slots] = Final_SSM.to(ssm_pool.dtype)
        k_pool[slots] = Final_K.to(k_pool.dtype)
        v_pool[slots] = Final_V.to(v_pool.dtype)

        y = rearrange(Out.squeeze(0)[:T], "t h p -> t (h p)")
        if self._postgate_norm:
            y = self.output_norm(y)
        out.copy_(y.to(out.dtype))

    def _decode_const_weights(self):
        """Rearranged constant mixer weights used by the decode-step kernels.

        ``C_bias``/``B_bias`` (rotary biases) and the MIMO projection weights
        are constant after weight loading, so their per-step ``rearrange`` (and
        the ``.contiguous()`` copies for the MIMO projections) are pure
        redundant work. Compute them once on first decode and cache, saving 3
        copy kernels per layer per step (x nheads M-layers).
        """
        cw = getattr(self, "_decode_const_w", None)
        if cw is None:
            cw = (
                rearrange(self.C_bias, "h r n -> r h n"),
                rearrange(self.B_bias, "h r n -> r h n"),
                rearrange(self.in_proj_mimo_x, "h r p -> r h p").contiguous(),
                rearrange(self.in_proj_mimo_z, "h r p -> r h p").contiguous(),
                rearrange(self.out_proj_mimo, "h r p -> r h p").contiguous(),
            )
            self._decode_const_w = cw
        return cw

    def _decode(
        self,
        u: torch.Tensor,        # (N, D)
        out: torch.Tensor,      # (N, D)
        slots: torch.Tensor,    # (N,)
        angle_pool: torch.Tensor,
        ssm_pool: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
    ) -> None:
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state
        fused_pre = (
            _FUSED_PREAMBLE
            and u.is_cuda
            and self.ngroups == 1
            and (D & (D - 1)) == 0
            and (S & (S - 1)) == 0
        )
        if fused_pre:
            # Two Triton kernels replace ~14 tiny eager kernels per call:
            # (1) x/z head-deinterleave + A/dt/trap activation scalars,
            # (2) B/C RMS norm + weight, written bf16 into one buffer.
            # Numerics note: DT/trap_s come out fp32 here (the eager path
            # below yields bf16, rounding per op) — strictly more precise;
            # greedy outputs can differ from the eager path within bf16 noise.
            zxdt, _ = self.in_proj(u)
            bc, _ = self.in_proj_dyn(u)
            n_tok = u.shape[0]
            x = torch.empty(n_tok, H, D, dtype=u.dtype, device=u.device)
            z = torch.empty_like(x)
            _A = torch.empty(n_tok, H, dtype=torch.float32, device=u.device)
            DT = torch.empty_like(_A)
            trap_s = torch.empty_like(_A)
            _dragon_decode_preamble_kernel[(n_tok * H,)](
                zxdt, x, z, _A, DT, trap_s, self.dt_bias,
                self.A_floor, H, zxdt.stride(0), D=D,
            )
            off_c = R * S
            bn_cn = torch.empty(n_tok, 2, R, S, dtype=u.dtype, device=u.device)
            _dragon_bc_norm_kernel[(n_tok * 2 * R,)](
                bc, bn_cn,
                self.B_norm.norm.weight, self.C_norm.norm.weight,
                self.B_norm.norm.rms.eps, off_c, R, bc.stride(0),
                ZERO_CENTERED=self.B_norm.norm.zero_centered, S=S,
            )
            B = bn_cn[:, 0].unsqueeze(2).expand(-1, -1, H, -1)
            C = bn_cn[:, 1].unsqueeze(2).expand(-1, -1, H, -1)
            angle = bc[..., 2 * off_c:]
        else:
            z, x, dt, A, trap, B, C, angle = self._project_in(u)

            # Slices from _project_in's .view() are non-contiguous. The
            # CuteDSL mamba3_step_fn kernel requires strides divisible by 8.
            x = x.contiguous()
            z = z.contiguous()

            _A = -F.softplus(A.to(torch.float32))
            _A = torch.clamp(_A, max=-self.A_floor)
            DT = F.softplus(dt.contiguous() + self.dt_bias)
            trap_s = torch.sigmoid(trap.contiguous())

            B = self.B_norm(B).expand(-1, -1, self.nheads_local, -1)
            C = self.C_norm(C).expand(-1, -1, self.nheads_local, -1)
        angle = angle.unsqueeze(-2).expand(-1, self.nheads_local, -1)

        # Constant rearranged weights (rotary biases + MIMO projections),
        # computed once and cached (see _decode_const_weights).
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()

        use_indexed = (
            _INDEXED_STEP
            and ssm_pool.dtype in (torch.float32, torch.bfloat16)
            and k_pool.dtype == torch.bfloat16
            and v_pool.dtype == x.dtype == torch.bfloat16
        )
        # With tile_D >= headdim the indexed kernel also stores the new B/x
        # key/value states itself (one CTA per (b, h)) — skip those scatters.
        kernel_writes_kv = use_indexed and _STEP_TILE_D >= self.headdim
        slots32 = slots.to(torch.int32)

        if use_indexed:
            # Rotary reads/updates angle_pool rows in place via slot indices
            # (PAD_SLOT_ID lanes: q/k zeroed, pool untouched, kernel-side).
            C, B, _ = _apply_rotary_qk_inference_fwd(
                q=C, k=B,
                angle_state=angle_pool,
                angle_proj=angle,
                dt=DT,
                bias_q=bias_q, bias_k=bias_k,
                conjugate=False, inplace=False, rotate_pairwise=False,
                state_batch_indices=slots32,
            )
        else:
            # Under CUDA-graph batch padding, padded lanes carry PAD_SLOT_ID
            # (-1). PyTorch advanced indexing wraps -1 to the LAST pool row,
            # so a padded lane would clobber a live request's state. Redirect
            # pad lanes to row 0 (vLLM's reserved null block) for all pool
            # writes; reads of row -1 only feed padded outputs (discarded).
            slots_w = torch.where(slots >= 0, slots, torch.zeros_like(slots))
            angle_state = angle_pool[slots]
            C, B, nxt_angle = _apply_rotary_qk_inference_fwd(
                q=C, k=B,
                angle_state=angle_state,
                angle_proj=angle,
                dt=DT,
                bias_q=bias_q, bias_k=bias_k,
                conjugate=False, inplace=False, rotate_pairwise=False,
            )

        # Write the step kernel's output straight into ``out`` (it is a
        # contiguous (N, H*D) slice) instead of a temp + final copy_.
        y_is_out = (
            not self._postgate_norm
            and out.dtype == x.dtype
            and out.is_contiguous()
        )
        y = out.view(out.shape[0], H, D) if y_is_out else torch.empty_like(x)
        if use_indexed:
            # Kernel reads/updates the pool rows directly via slot indices —
            # no gather/scatter round-trip on the 1.5 MB/seq fp32 SSM state.
            _mamba3_step_fn(
                ssm_pool,
                k_pool,
                v_pool,
                _A,
                B.to(torch.bfloat16),
                C.to(torch.bfloat16),
                self.D,
                x,
                DT,
                trap_s,
                xpj,
                outpj,
                None,  # in-place pool update
                y,
                z=z,
                zproj=zpj,
                state_batch_indices=slots32,
                update_kv_state=kernel_writes_kv,
                tile_D=_STEP_TILE_D,
                num_warps=_STEP_NUM_WARPS,
            )
            if not kernel_writes_kv:  # only if _STEP_TILE_D < headdim
                slots_w = torch.where(slots >= 0, slots, torch.zeros_like(slots))
                k_pool[slots_w] = B
                v_pool[slots_w] = x
        else:
            ssm_state = ssm_pool[slots].to(torch.float32)
            k_state = k_pool[slots]
            v_state = v_pool[slots]
            state_out = torch.empty_like(ssm_state)
            _mamba3_step_fn(
                ssm_state,
                k_state.to(torch.bfloat16),
                v_state.to(torch.bfloat16),
                _A,
                B.to(torch.bfloat16),
                C.to(torch.bfloat16),
                self.D,
                x,
                DT,
                trap_s,
                xpj,
                outpj,
                state_out,
                y,
                z=z,
                zproj=zpj,
                tile_D=_STEP_TILE_D,
                num_warps=_STEP_NUM_WARPS,
            )
            # New angle/B/x states for the next step. The kernels have
            # already consumed the old pool rows. (The indexed path needs
            # none of these scatters: rotary and step kernels write the
            # pools in place.)
            ssm_pool[slots_w] = state_out
            angle_pool[slots_w] = nxt_angle
            k_pool[slots_w] = B
            v_pool[slots_w] = x

        if not y_is_out:
            y = rearrange(y, "n h p -> n (h p)")
            if self._postgate_norm:
                y = self.output_norm(y)
            out.copy_(y.to(out.dtype))
