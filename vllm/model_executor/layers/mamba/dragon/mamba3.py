# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dragon Mamba3 MIMO mixer.

Native port of ``DragonMamba3MimoFast`` from ``modeling_dragon.py``. The four
temporal states (angle, ssm, k, v) live in vLLM's paged mamba state pool;
per-request slots arrive as ``Mamba3AttentionMetadata.state_indices_tensor``
from the MAMBA3 backend.

Prefill packs the batch into ``(B=1, S=total_tokens)`` and calls the TileLang
varlen kernel ``mamba_ssm.ops.tilelang.mamba3.mamba3_mimo`` with
``cu_seqlens``. Decode runs the packed CuteDSL ``mamba3_step_fn`` over the
whole decode-token batch, indexing the state pools in place.
"""

from __future__ import annotations

import itertools
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.dragon.norm import DragonNorm
from vllm.model_executor.layers.mamba.dragon.state import (
    mamba3_state_dtype,
    mamba3_state_shape,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mamba3_attn import Mamba3AttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# CuteDSL decode-step launch config. ``tile_D`` covers the full headdim, so
# exactly one CTA owns each (batch, head) row — the precondition for letting
# the kernel store the new B/x key/value states itself (``update_kv_state``).
_STEP_TILE_D = 64
_STEP_NUM_WARPS = 4

# Fusing bias+rotary into the step kernel avoids materializing the per-head
# rotated B/C, but its smem rotation stages serialize against the cp.async
# pipeline once occupancy is high. Measured on GH200 under graph capture:
# ~1 us/layer faster at batch 1, ~26 us/layer slower at batch 256. Dispatch on
# batch size; each cudagraph capture size takes one consistent branch.
_FUSE_ROTARY_MAX_BATCH = 64

# External Mamba3 kernels (from the official ``mamba_ssm`` package), imported
# lazily so this module stays importable on hosts without them.
_mamba3_mimo = None
_mamba3_mimo_grouped = None
_mamba3_step_fn = None
_apply_rotary_qk_inference_fwd = None
# True when mamba3_mimo accepts Input_States (mamba_ssm mimo_input_state
# merge): continuation prefills then run as one packed varlen call instead of
# a serial per-token recurrence.
_MIMO_SUPPORTS_INPUT_STATES = False


def _is_pow2(n: int) -> bool:
    return n & (n - 1) == 0


def _lazy_import_kernels() -> None:
    global _mamba3_mimo, _mamba3_mimo_grouped, _mamba3_step_fn
    global _apply_rotary_qk_inference_fwd, _MIMO_SUPPORTS_INPUT_STATES
    if _mamba3_mimo is not None:
        return
    import inspect

    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd,
    )

    _mamba3_mimo = mamba3_mimo
    _mamba3_step_fn = mamba3_step_fn
    _apply_rotary_qk_inference_fwd = apply_rotary_qk_inference_fwd
    _MIMO_SUPPORTS_INPUT_STATES = (
        "Input_States" in inspect.signature(mamba3_mimo).parameters
    )
    # Group-parallel exact prefill (mamba_ssm mamba3-prefill-opt): splits long
    # sequences into virtual groups for GPU occupancy — same math, ~1.4x on
    # long prompts. Disable with DRAGON_GROUPED_PREFILL=0.
    if os.environ.get("DRAGON_GROUPED_PREFILL", "1") != "0":
        try:
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import (
                mamba3_mimo_varlen_grouped,
            )

            _mamba3_mimo_grouped = mamba3_mimo_varlen_grouped
        except ImportError:
            _mamba3_mimo_grouped = None


@triton.jit
def _decode_preamble_kernel(
    zxdt_ptr,  # (N, H*(2D+3)) — in_proj output, head-interleaved
    x_ptr,  # out (N, H, D) contiguous, input dtype
    z_ptr,  # out (N, H, D) contiguous, input dtype
    a_ptr,  # out (N, H) fp32: clamp(-softplus(A), max=-a_floor)
    dt_ptr,  # out (N, H) fp32: softplus(dt + dt_bias[h])
    trap_ptr,  # out (N, H) fp32: sigmoid(trap)
    dt_bias_ptr,  # (H,)
    a_floor,
    H,
    stride_n,
    D: tl.constexpr,
):
    """One program per (token, head).

    Replaces the x/z ``.contiguous()`` copies and the A/dt/trap
    softplus/clamp/sigmoid chain — ~14 tiny kernels per layer per step, which
    is pure launch overhead at decode shapes.
    """
    pid = tl.program_id(0)
    h = pid % H
    base = zxdt_ptr + (pid // H) * stride_n + h * (2 * D + 3)
    offs = tl.arange(0, D)  # D is a power of 2 (guarded at the call site)
    tl.store(z_ptr + pid * D + offs, tl.load(base + offs))
    tl.store(x_ptr + pid * D + offs, tl.load(base + D + offs))
    dt = tl.load(base + 2 * D).to(tl.float32)
    a = tl.load(base + 2 * D + 1).to(tl.float32)
    trap = tl.load(base + 2 * D + 2).to(tl.float32)
    # F.softplus, including its overflow threshold (identity for x > 20).
    sp_a = tl.where(a > 20.0, a, tl.log(1.0 + tl.exp(a)))
    v = dt + tl.load(dt_bias_ptr + h).to(tl.float32)
    sp_dt = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(v)))
    tl.store(a_ptr + pid, tl.minimum(-sp_a, -a_floor))
    tl.store(dt_ptr + pid, sp_dt)
    tl.store(trap_ptr + pid, 1.0 / (1.0 + tl.exp(-trap)))


@triton.jit
def _bc_norm_kernel(
    bc_ptr,  # (N, 2*R*S + n_angles) — in_proj_dyn output (ngroups == 1)
    out_ptr,  # (N, 2, R, S) contiguous: [:,0]=B_norm(B), [:,1]=C_norm(C)
    wb_ptr,  # (S,) B_norm gain
    wc_ptr,  # (S,) C_norm gain
    eps,
    off_c,  # offset of the C block inside a bc row (= R*S)
    R,
    stride_n,
    ZERO_CENTERED: tl.constexpr,
    S: tl.constexpr,
):
    """One program per (token, B/C, mimo-row): fused RMS norm + gain."""
    pid = tl.program_id(0)  # n*(2R) + which*R + r
    j = pid % (2 * R)
    base = bc_ptr + (pid // (2 * R)) * stride_n + (j % R) * S + (j // R) * off_c
    offs = tl.arange(0, S)  # S is a power of 2 (guarded at the call site)
    v = tl.load(base + offs).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(v * v) / S + eps)
    wb = tl.load(wb_ptr + offs).to(tl.float32)
    wc = tl.load(wc_ptr + offs).to(tl.float32)
    w = tl.where(j >= R, wc, wb)
    if ZERO_CENTERED:
        w = w + 1.0
    tl.store(out_ptr + pid * S + offs, (v * inv * w).to(out_ptr.dtype.element_ty))


class DragonMamba3Mixer(PluggableLayer, MambaBase):
    """Mamba3 MIMO mixer over vLLM's paged recurrent state.

    State layout per request slot:
        angle : (nheads, num_rope_angles)    float32
        ssm   : (nheads, headdim, d_state)   ssm cache dtype
        k     : (mimo_dim, nheads, d_state)  state dtype
        v     : (nheads, headdim)            state dtype
    """

    # Populated by the model runner via MambaSpec after init.
    kv_cache: tuple[torch.Tensor, ...]

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        tp = get_tensor_model_parallel_world_size()
        self.tp_size = tp

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

        # in_proj is head-interleaved
        # ``[h0: z(headdim), x(headdim), dt, A, trap | h1: ... ]``, total width
        # ``nheads * (2*headdim + 3)``. Splitting the output axis by TP
        # therefore yields a clean per-head shard.
        self.in_proj = ColumnParallelLinear(
            input_size=self.d_model,
            output_size=self.d_inner * 2 + 3 * self.nheads,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.in_proj",
        )
        # in_proj_dyn produces B/C (per group, shared across heads) and the
        # rope angles (shared across heads), so it is replicated.
        self.in_proj_dyn = ReplicatedLinear(
            input_size=self.d_model,
            output_size=(
                2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles
            ),
            bias=False,
            prefix=f"{prefix}.in_proj_dyn",
        )

        head_shard = {"weight_loader": sharded_weight_loader(0)}
        self.B_bias = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.d_state, dtype=torch.float32
            )
        )
        self.C_bias = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.d_state, dtype=torch.float32
            )
        )
        set_weight_attrs(self.B_bias, head_shard)
        set_weight_attrs(self.C_bias, head_shard)

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        self.B_norm = DragonNorm(self.d_state, eps=eps, zero_centered=zc)
        self.C_norm = DragonNorm(self.d_state, eps=eps, zero_centered=zc)

        self.in_proj_mimo_x = nn.Parameter(
            torch.full(
                (self.nheads_local, self.mimo_dim, self.headdim),
                1.0 / self.mimo_dim,
                dtype=torch.float32,
            )
        )
        self.in_proj_mimo_z = nn.Parameter(
            torch.ones(
                self.nheads_local, self.mimo_dim, self.headdim, dtype=torch.float32
            )
        )
        self.out_proj_mimo = nn.Parameter(
            torch.full(
                (self.nheads_local, self.mimo_dim, self.headdim),
                1.0 / self.mimo_dim,
                dtype=torch.float32,
            )
        )
        set_weight_attrs(self.in_proj_mimo_x, head_shard)
        set_weight_attrs(self.in_proj_mimo_z, head_shard)
        set_weight_attrs(self.out_proj_mimo, head_shard)

        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt = torch.exp(
            torch.rand(self.nheads_local) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        set_weight_attrs(self.dt_bias, head_shard)

        self.D = nn.Parameter(torch.ones(self.nheads_local))
        set_weight_attrs(self.D, head_shard)

        self._prefill_const_w: tuple[torch.Tensor, ...] | None = None
        self._decode_const_w: tuple[torch.Tensor, ...] | None = None

        self._postgate_norm = getattr(config, "mamba3_postgate_norm", False)
        if self._postgate_norm:
            # Per-rank local-shard norm: each rank normalizes only its own
            # d_inner slice, which differs numerically from a global RMSNorm
            # but matches how the mixer output is split.
            self.output_norm = DragonNorm(self.d_inner // tp, eps=eps, zero_centered=zc)
            set_weight_attrs(self.output_norm.norm.weight, head_shard)

        # Register in the static forward-context table so the runner can find
        # this layer's metadata by prefix.
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    # -- MambaBase ----------------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA3

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = mamba3_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
        # The CuteDSL step kernel indexes the pools in place, so their dtypes
        # are a hard precondition rather than something we can cast around.
        _, ssm_dtype, k_dtype, v_dtype = dtypes
        if ssm_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                f"Dragon mamba3 supports an fp32 or bf16 SSM state, got "
                f"{ssm_dtype}. Set --mamba-ssm-cache-dtype to float32 or auto."
            )
        if k_dtype != torch.bfloat16 or v_dtype != torch.bfloat16:
            raise ValueError(
                f"Dragon mamba3 requires a bf16 key/value state, got "
                f"{k_dtype}/{v_dtype}. Leave --mamba-cache-dtype at auto with "
                f"a bf16 model."
            )
        return dtypes

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return mamba3_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.nheads,
            head_dim=self.headdim,
            d_state=self.d_state,
            mimo_dim=self.mimo_dim,
            num_rope_angles=self.num_rope_angles,
        )

    # -- Forward ------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``(num_tokens, hidden_size)`` in, ``(num_tokens, d_inner/tp)`` out.

        The state-dependent work runs outside the compiled graph as the
        ``vllm::dragon_mamba3`` custom op, mirroring ``mamba_mixer2``.
        """
        out = torch.empty(
            hidden_states.shape[0],
            self.nheads_local * self.headdim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        torch.ops.vllm.dragon_mamba3(
            hidden_states, out, _encode_layer_name(self.prefix)
        )
        return out

    def _forward_impl(self, hidden_states: torch.Tensor, out: torch.Tensor) -> None:
        _lazy_import_kernels()

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # Profiling / dry run.
            out.zero_()
            return
        md: Mamba3AttentionMetadata = attn_metadata[self.prefix]

        pools = self.kv_cache
        slots = md.state_indices_tensor
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens

        if md.num_decodes > 0:
            if md.spec is not None:
                self._decode_spec(hidden_states[:ndt], out[:ndt], md.spec, pools)
            else:
                self._decode(
                    hidden_states[:ndt], out[:ndt], slots[: md.num_decodes], pools
                )

        if md.num_prefills > 0:
            self._prefill(
                hidden_states[ndt : ndt + npt],
                out[ndt : ndt + npt],
                slots[md.num_decodes : md.num_decodes + md.num_prefills],
                md,
                pools,
            )

    # -- Projections --------------------------------------------------------
    def _project_in(self, h: torch.Tensor):
        """Run in_proj/in_proj_dyn on ``(N, D)`` and unpack the pieces."""
        zxdtAtrap, _ = self.in_proj(h)
        per_head = zxdtAtrap.view(h.shape[0], self.nheads_local, 2 * self.headdim + 3)
        z = per_head[..., 0 : self.headdim]
        x = per_head[..., self.headdim : 2 * self.headdim]
        dt = per_head[..., 2 * self.headdim]
        A = per_head[..., 2 * self.headdim + 1]
        trap = per_head[..., 2 * self.headdim + 2]

        bc, _ = self.in_proj_dyn(h)
        off = self.ngroups * self.mimo_dim * self.d_state
        B = rearrange(
            bc[..., :off], "n (G r s) -> n r G s", G=self.ngroups, r=self.mimo_dim
        )
        C = rearrange(
            bc[..., off : 2 * off],
            "n (G r s) -> n r G s",
            G=self.ngroups,
            r=self.mimo_dim,
        )
        angle = bc[..., 2 * off :]
        return z, x, dt, A, trap, B, C, angle

    # -- Prefill ------------------------------------------------------------
    def _prefill(
        self,
        h: torch.Tensor,  # (T, D)
        out: torch.Tensor,  # (T, d_inner_local)
        slots: torch.Tensor,  # (P,)
        md: Mamba3AttentionMetadata,
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        # Fast path: every prompt starts from zero state, so one packed varlen
        # launch covers the batch. DragonForCausalLMConfig disables chunked
        # prefill precisely to keep us here.
        if not md.has_initial_any:
            self._prefill_zero_start(
                h, out, slots, md.query_start_loc_p, pools,
                qsl_cpu=md.query_start_loc_p_cpu,
            )
            return

        # Continuation chunks resume a cached state. With an Input_States-
        # capable mamba3_mimo (mamba_ssm mimo_input_state), one packed varlen
        # call still covers the whole batch: zero-start sequences get zero
        # init states, continuations get their pool rows.
        if _MIMO_SUPPORTS_INPUT_STATES:
            init = self._gather_init_states(pools, slots, md.has_initial_state, h.dtype)
            self._prefill_zero_start(
                h, out, slots, md.query_start_loc_p, pools, init_states=init,
                qsl_cpu=md.query_start_loc_p_cpu,
            )
            return

        # Legacy kernel: split zero-start sequences (one packed varlen call)
        # from continuations (serial per-token recurrence).
        has_init = md.has_initial_state_cpu
        qsl_cpu = md.query_start_loc_p_cpu
        zero_idxs = [i for i in range(has_init.numel()) if not has_init[i]]
        cont_idxs = [i for i in range(has_init.numel()) if has_init[i]]

        if zero_idxs:
            lens = [int(qsl_cpu[i + 1] - qsl_cpu[i]) for i in zero_idxs]
            packed = torch.cat(
                [h[int(qsl_cpu[i]) : int(qsl_cpu[i + 1])] for i in zero_idxs], dim=0
            )
            packed_out = torch.empty(
                sum(lens), out.shape[1], device=out.device, dtype=out.dtype
            )
            packed_qsl = torch.tensor(
                [0] + list(itertools.accumulate(lens)),
                device=md.query_start_loc_p.device,
                dtype=md.query_start_loc_p.dtype,
            )
            self._prefill_zero_start(
                packed,
                packed_out,
                slots[torch.as_tensor(zero_idxs, device=slots.device)],
                packed_qsl,
                pools,
            )
            cursor = 0
            for i, length in zip(zero_idxs, lens):
                start = int(qsl_cpu[i])
                out[start : start + length].copy_(packed_out[cursor : cursor + length])
                cursor += length

        if cont_idxs:
            self._prefill_continuation(h, out, slots, qsl_cpu, cont_idxs, pools)

    def _prefill_continuation(
        self,
        h: torch.Tensor,
        out: torch.Tensor,
        slots: torch.Tensor,
        qsl_cpu: torch.Tensor,
        req_idxs: list[int],
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        """Serial per-token recurrence for chunks that resume a cached state."""
        lens = [int(qsl_cpu[i + 1] - qsl_cpu[i]) for i in req_idxs]
        starts = [int(qsl_cpu[i]) for i in req_idxs]
        req_slots = slots[torch.as_tensor(req_idxs, device=slots.device)]
        for t in range(max(lens)):
            active = [k for k, length in enumerate(lens) if t < length]
            if not active:
                break
            rows = torch.as_tensor(
                [starts[k] + t for k in active], device=h.device, dtype=torch.long
            )
            step_out = torch.empty(
                len(active), out.shape[1], device=out.device, dtype=out.dtype
            )
            self._decode(
                h.index_select(0, rows),
                step_out,
                req_slots[torch.as_tensor(active, device=req_slots.device)],
                pools,
            )
            out.index_copy_(0, rows, step_out)

    @staticmethod
    def _gather_init_states(
        pools: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
        has_initial: torch.Tensor,  # (P,) bool, on device
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sequence init states for the varlen kernel.

        Continuations read their pool rows; zero-start sequences get zeros
        (masked on device — no CPU sync).
        """
        angle_pool, ssm_pool, k_pool, v_pool = pools
        idx = slots.long()
        mask1 = has_initial.view(-1, *([1] * (angle_pool.dim() - 1)))
        angle = angle_pool[idx].to(torch.float32) * mask1
        mask = has_initial.view(-1, *([1] * (ssm_pool.dim() - 1)))
        ssm = ssm_pool[idx].to(torch.float32) * mask
        k = k_pool[idx].to(compute_dtype) * has_initial.view(
            -1, *([1] * (k_pool.dim() - 1))
        )
        v = v_pool[idx].to(compute_dtype) * has_initial.view(
            -1, *([1] * (v_pool.dim() - 1))
        )
        return angle, ssm, k, v

    def _prefill_zero_start(
        self,
        h: torch.Tensor,  # (T, D)
        out: torch.Tensor,  # (T, d_inner_local)
        slots: torch.Tensor,  # (P,)
        qsl: torch.Tensor,  # (P+1,) rebased to 0
        pools: tuple[torch.Tensor, ...],
        init_states: tuple[torch.Tensor, ...] | None = None,
        qsl_cpu: torch.Tensor | None = None,
    ) -> None:
        angle_pool, ssm_pool, k_pool, v_pool = pools
        z, x, dt, A, trap, B, C, angle = self._project_in(h)
        z = z.unsqueeze(0)  # (1, T, H, p)
        x = x.unsqueeze(0)  # (1, T, H, p)
        dt = dt.unsqueeze(0).to(torch.float32)  # (1, T, H)
        A = A.unsqueeze(0)  # (1, T, H)
        trap = trap.unsqueeze(0).permute(0, 2, 1).contiguous()  # (1, H, T)
        B = B.unsqueeze(0)  # (1, T, R, G, N)
        C = C.unsqueeze(0)  # (1, T, R, G, N)
        # Angles expand over heads in fp32; the kernel does the angle*dt cumsum.
        angle = (
            angle.unsqueeze(0)
            .unsqueeze(-2)
            .expand(-1, -1, self.nheads_local, -1)
            .to(torch.float32)
            .contiguous()
        )

        B = self.B_norm(B)
        C = self.C_norm(C)

        _A = torch.clamp(-F.softplus(A.to(torch.float32)), max=-self.A_floor)
        DT = F.softplus(dt + self.dt_bias)
        ADT = (_A * DT).permute(0, 2, 1).contiguous()  # (1, H, T)
        DT = DT.permute(0, 2, 1).contiguous()  # (1, H, T)

        # Constant fp32 weight casts, computed once and cached (otherwise 6
        # cast kernels + allocations per layer per prefill). Invalidated by
        # invalidate_weight_caches() on weight reload.
        cw = self._prefill_const_w
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
        q_bias, k_bias, mimo_v, mimo_z, mimo_out, d_f32 = cw

        kernel_kwargs = dict(
            Q=C.contiguous().bfloat16(),
            K=B.contiguous().bfloat16(),
            V=x.contiguous().bfloat16(),
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=q_bias,
            K_bias=k_bias,
            MIMO_V=mimo_v,
            MIMO_Z=mimo_z,
            MIMO_Out=mimo_out,
            Angles=angle,
            D=d_f32,
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            cu_seqlens=qsl.to(torch.int32),
        )
        if _mamba3_mimo_grouped is not None:
            # Group-parallel exact prefill: long sequences split into virtual
            # groups (kernel occupancy), short ones take the single-pass path
            # inside the wrapper. Same numerics either way.
            Out, Final_Angle, Final_SSM, Final_K, _ = _mamba3_mimo_grouped(
                **kernel_kwargs,
                Input_States=init_states,
                cu_seqlens_cpu=qsl_cpu,
            )
        else:
            Out, Final_Angle, Final_SSM, Final_K, _ = _mamba3_mimo(
                **kernel_kwargs,
                return_state=True,
                **({"Input_States": init_states} if init_states is not None else {}),
            )

        # The kernel's Final_V is taken at the global last token, which is
        # wrong for a packed varlen batch — recompute it per sequence.
        angle_pool[slots] = Final_Angle.to(angle_pool.dtype)
        ssm_pool[slots] = Final_SSM.to(ssm_pool.dtype)
        k_pool[slots] = Final_K.to(k_pool.dtype)
        v_pool[slots] = x[0, qsl[1:].long() - 1].to(v_pool.dtype)

        y = rearrange(Out.squeeze(0), "t h p -> t (h p)")
        if self._postgate_norm:
            y = self.output_norm(y)
        out.copy_(y.to(out.dtype))

    # -- Decode -------------------------------------------------------------
    def invalidate_weight_caches(self) -> None:
        """Drop the cached weight-derived tensors.

        Must be called after any in-place weight reload: the caches mix live
        views (the rearranged rotary biases) with snapshots (the fp32 casts
        and contiguous copies), so running on stale caches after a reload
        would silently blend old and new weights.
        """
        self._prefill_const_w = None
        self._decode_const_w = None

    def _decode_const_weights(self):
        """Rearranged constant weights for the decode-step kernels.

        The rotary biases and MIMO projections never change between weight
        loads, so their per-step ``rearrange`` + ``.contiguous()`` copies are
        pure redundant work (3 copy kernels per layer per step). Invalidated
        by invalidate_weight_caches() on weight reload.
        """
        cw = self._decode_const_w
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

    def _decode_spec(
        self,
        u: torch.Tensor,  # (ndt, D) — all decode tokens, ragged per request
        out: torch.Tensor,  # (ndt, d_inner_local)
        spec,  # DragonSpecMetadata
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        """Speculative verify: chain the dual-slot step kernel over positions.

        Position t of each request reads the state written by position t-1
        (column t-1; for t = 0 the column selected by last step's acceptance)
        and writes column t. Inactive lanes (requests with fewer tokens) get
        slot -1: the kernel zeroes their output and suppresses every write.
        Always uses the fused rotary (correct at any batch size) so the angle
        pool follows the same column protocol.
        """
        angle_pool, ssm_pool, k_pool, v_pool = pools
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state
        if not (self.ngroups == 1 and _is_pow2(D) and _is_pow2(S)):
            raise NotImplementedError(
                "Dragon spec decode requires ngroups == 1 and power-of-two "
                "headdim/d_state (the fused preamble shapes)."
            )
        assert self.headdim <= _STEP_TILE_D
        ndt = u.shape[0]

        # Preamble over every decode token at once (same kernels as _decode).
        zxdt, _ = self.in_proj(u)
        bc, _ = self.in_proj_dyn(u)
        x = torch.empty(ndt, H, D, dtype=u.dtype, device=u.device)
        z = torch.empty_like(x)
        _A = torch.empty(ndt, H, dtype=torch.float32, device=u.device)
        DT = torch.empty_like(_A)
        trap = torch.empty_like(_A)
        _decode_preamble_kernel[(ndt * H,)](
            zxdt,
            x,
            z,
            _A,
            DT,
            trap,
            self.dt_bias,
            self.A_floor,
            H,
            zxdt.stride(0),
            D=D,
        )
        bn_cn = torch.empty(ndt, 2, R, S, dtype=u.dtype, device=u.device)
        _bc_norm_kernel[(ndt * 2 * R,)](
            bc,
            bn_cn,
            self.B_norm.norm.weight,
            self.C_norm.norm.weight,
            self.B_norm.norm.eps,
            R * S,
            R,
            bc.stride(0),
            ZERO_CENTERED=self.B_norm.norm.zero_centered,
            S=S,
        )
        angle = bc[..., 2 * R * S :]
        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()

        qsl = spec.query_start_loc_d
        starts = qsl[:-1].long()
        qlens = torch.diff(qsl)
        cols = spec.state_cols.to(torch.int32)
        in0 = spec.initial_slots().to(torch.int32)
        neg1 = torch.full_like(in0, -1)
        dummy = torch.full_like(starts, ndt)
        # Extra dummy row absorbs inactive lanes' outputs.
        y_pad = torch.zeros(ndt + 1, H, D, dtype=u.dtype, device=u.device)

        for t in range(spec.max_qlen):
            active = qlens > t
            rows = torch.where(active, starts + t, starts)
            src = in0 if t == 0 else cols[:, t - 1].contiguous()
            in_t = torch.where(active, src, neg1)
            out_t = torch.where(active, cols[:, t].contiguous(), neg1)
            bpre = bn_cn.index_select(0, rows)
            y_t = torch.empty(rows.shape[0], H, D, dtype=u.dtype, device=u.device)
            _mamba3_step_fn(
                ssm_pool,
                k_pool,
                v_pool,
                _A.index_select(0, rows),
                bpre[:, 0].unsqueeze(2).expand(-1, -1, H, -1).to(torch.bfloat16),
                bpre[:, 1].unsqueeze(2).expand(-1, -1, H, -1).to(torch.bfloat16),
                self.D,
                x.index_select(0, rows),
                DT.index_select(0, rows),
                trap.index_select(0, rows),
                xpj,
                outpj,
                None,
                y_t,
                z=z.index_select(0, rows),
                zproj=zpj,
                state_batch_indices=in_t,
                state_batch_indices_out=out_t,
                update_kv_state=True,
                tile_D=_STEP_TILE_D,
                num_warps=_STEP_NUM_WARPS,
                rotary_dim=2 * self.num_rope_angles,
                rotary_bias_q=bias_q,
                rotary_bias_k=bias_k,
                rotary_angle_proj=angle.index_select(0, rows)
                .unsqueeze(-2)
                .expand(-1, H, -1),
                rotary_angle_state=angle_pool,
            )
            y_pad.index_copy_(0, torch.where(active, starts + t, dummy), y_t)

        y = y_pad[:ndt].reshape(ndt, H * D)
        if self._postgate_norm:
            y = self.output_norm(y)
        out.copy_(y.to(out.dtype))

    def _decode(
        self,
        u: torch.Tensor,  # (N, D)
        out: torch.Tensor,  # (N, d_inner_local)
        slots: torch.Tensor,  # (N,)
        pools: tuple[torch.Tensor, ...],
    ) -> None:
        angle_pool, ssm_pool, k_pool, v_pool = pools
        H, D = self.nheads_local, self.headdim
        R, S = self.mimo_dim, self.d_state

        # Two Triton kernels replace ~14 tiny eager kernels per call: x/z
        # head-deinterleave plus the A/dt/trap activations, and the B/C RMS
        # norm. They also keep DT/trap in fp32 where the eager chain rounded
        # to bf16 per op, so greedy output can differ within bf16 noise.
        fuse_preamble = self.ngroups == 1 and _is_pow2(D) and _is_pow2(S)
        if fuse_preamble:
            zxdt, _ = self.in_proj(u)
            bc, _ = self.in_proj_dyn(u)
            n_tok = u.shape[0]
            x = torch.empty(n_tok, H, D, dtype=u.dtype, device=u.device)
            z = torch.empty_like(x)
            _A = torch.empty(n_tok, H, dtype=torch.float32, device=u.device)
            DT = torch.empty_like(_A)
            trap = torch.empty_like(_A)
            _decode_preamble_kernel[(n_tok * H,)](
                zxdt,
                x,
                z,
                _A,
                DT,
                trap,
                self.dt_bias,
                self.A_floor,
                H,
                zxdt.stride(0),
                D=D,
            )
            bn_cn = torch.empty(n_tok, 2, R, S, dtype=u.dtype, device=u.device)
            _bc_norm_kernel[(n_tok * 2 * R,)](
                bc,
                bn_cn,
                self.B_norm.norm.weight,
                self.C_norm.norm.weight,
                self.B_norm.norm.eps,
                R * S,
                R,
                bc.stride(0),
                ZERO_CENTERED=self.B_norm.norm.zero_centered,
                S=S,
            )
            B = bn_cn[:, 0].unsqueeze(2).expand(-1, -1, H, -1)
            C = bn_cn[:, 1].unsqueeze(2).expand(-1, -1, H, -1)
            angle = bc[..., 2 * R * S :]
        else:
            z, x, dt, A, trap_raw, B, C, angle = self._project_in(u)
            # Views from _project_in are non-contiguous; the CuteDSL step
            # kernel needs strides divisible by 8.
            x = x.contiguous()
            z = z.contiguous()
            _A = torch.clamp(-F.softplus(A.to(torch.float32)), max=-self.A_floor)
            DT = F.softplus(dt.contiguous() + self.dt_bias)
            trap = torch.sigmoid(trap_raw.contiguous())
            B = self.B_norm(B).expand(-1, -1, H, -1)
            C = self.C_norm(C).expand(-1, -1, H, -1)
        angle = angle.unsqueeze(-2).expand(-1, H, -1)

        bias_q, bias_k, xpj, zpj, outpj = self._decode_const_weights()
        slots32 = slots.to(torch.int32).contiguous()

        # With tile_D >= headdim one CTA owns each (b, h) row, so the step
        # kernel can also store the new B/x key/value states itself.
        kernel_writes_kv = self.headdim <= _STEP_TILE_D
        # The fused rotary reads B/C head-broadcast, which only holds for a
        # single group. The kernel must also own the k_pool write, or the
        # caller-side scatter below would store the pre-rotation B.
        fuse_rotary = (
            kernel_writes_kv
            and self.ngroups == 1
            and u.shape[0] <= _FUSE_ROTARY_MAX_BATCH
        )

        if fuse_rotary:
            # No separate rotary launch: the step kernel applies bias+rotary to
            # the head-broadcast pre-rotation B/C and updates angle_pool itself.
            rotary_kwargs = dict(
                rotary_dim=2 * self.num_rope_angles,
                rotary_bias_q=bias_q,
                rotary_bias_k=bias_k,
                rotary_angle_proj=angle,
                rotary_angle_state=angle_pool,
            )
        else:
            rotary_kwargs = {}
            C, B, _ = _apply_rotary_qk_inference_fwd(
                q=C,
                k=B,
                angle_state=angle_pool,
                angle_proj=angle,
                dt=DT,
                bias_q=bias_q,
                bias_k=bias_k,
                conjugate=False,
                inplace=False,
                rotate_pairwise=False,
                state_batch_indices=slots32,
            )

        # Write the step kernel's output straight into ``out`` when it is a
        # contiguous (N, H*D) slice, instead of a temp plus a final copy.
        y_is_out = (
            not self._postgate_norm and out.dtype == x.dtype and out.is_contiguous()
        )
        y = out.view(out.shape[0], H, D) if y_is_out else torch.empty_like(x)

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
            trap,
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
            **rotary_kwargs,
        )
        if not kernel_writes_kv:
            k_pool[slots] = B
            v_pool[slots] = x

        if not y_is_out:
            y = rearrange(y, "n h p -> n (h p)")
            if self._postgate_norm:
                y = self.output_norm(y)
            out.copy_(y.to(out.dtype))


def dragon_mamba3(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward_impl(hidden_states, output)


def dragon_mamba3_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="dragon_mamba3",
    op_func=dragon_mamba3,
    mutates_args=["output"],
    fake_impl=dragon_mamba3_fake,
)
