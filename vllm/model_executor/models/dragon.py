# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Dragon model for vLLM.

Dragon is a hybrid model stack:
    - ``M`` layers: ``DragonMamba3Mixer``  (SSM with 4 temporal states,
      paged via vLLM's Mamba state manager through the MAMBA3 backend).
    - ``V`` layers: ``DragonDiffTPAAttention``  (Differential TPA softmax
      attention with paged KV + optional 1-token shift buffer via the
      DRAGON_DIFF_TPA side-channel backend).
    - MoE MLP: ``DragonLatentMoE`` (latent projection + 256-expert
      ``SharedFusedMoE`` + shared expert), dense ``DragonMLP`` fallback.

This file wires those mixers into a native vLLM ``DragonForCausalLM`` that
owns the paged KV / paged SSM state, the block residual / geodesic-norm
residual path, and weight loading.

Known fidelity gaps (flagged, not silently papered over):
    - vLLM's default ``get_rope`` replaces Dragon's custom ``p-rope``.
    - Per-layer sliding window, ``intra_doc_masking``, ``scalable_softmax``
      path is wired but uses vLLM's softmax scale (scalable-softmax pre-
      scales ``q``); sliding-window attention bounds the log_pos only,
      the main SW mask still depends on vLLM's default.
    - ``token_conv1d_attn``, ``xsa``, ``use_value_embedding`` not ported.
    - TP is supported by row-splitting the head axis on the head-interleaved
      projections. ``cosnet=True`` is not supported with tp > 1. The Mamba3
      post-gate norm (when enabled) runs as a local per-rank shard norm,
      which differs numerically from a global RMSNorm.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Callable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fused_moe import (
    GateLinear,
    SharedFusedMoE,
    activation_without_mul,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.dragon_diff_tpa import DragonDiffTPAAttention
from vllm.model_executor.layers.mamba.dragon_mamba3 import (
    DragonMamba3Mixer,
    _DragonNorm,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Fused Triton geodesic-norm (DRAGON_FUSED_GEODESIC=0 to fall back to eager).
import os as _os
_FUSED_GEODESIC = _os.environ.get("DRAGON_FUSED_GEODESIC", "1") != "0"

# --- optional per-component CUDA-event profiler (DRAGON_PROFILE=1) -----------
_DPROF = bool(_os.environ.get("DRAGON_PROFILE"))
if _DPROF:
    import json as _json
    import collections as _c
    _ACC = _c.defaultdict(float)
    _CNT = _c.defaultdict(int)
    _PROF_OUT = _os.environ.get("DRAGON_PROFILE_OUT",
                                "dragon_comp_prof.json")

    def _timed(key, fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); r = fn(); e.record(); e.synchronize()
        _ACC[key] += s.elapsed_time(e); _CNT[key] += 1
        return r

    def _dump_prof():
        with open(_PROF_OUT, "w") as f:
            _json.dump({"ms": dict(_ACC), "count": dict(_CNT)}, f)
else:
    def _timed(key, fn):
        return fn()

    def _dump_prof():
        pass


# =============================================================================
# MLP building blocks
# =============================================================================


class DragonMLP(nn.Module):
    """Dense ReLU² MLP (``fc_1 → relu² → fc_2``), Dragon naming, TP-aware.

    Used for the dense (non-MoE) layers. Checkpoint names match Dragon's
    ``fc_1.weight`` / ``fc_2.weight``. TP splits ``fc_1`` column-wise and
    ``fc_2`` row-wise, allreducing at the output.
    """

    def __init__(self, config, prefix: str = "", quant_config=None):
        super().__init__()
        if get_tensor_model_parallel_world_size() > 1 and getattr(
            config, "cosnet", False
        ):
            raise NotImplementedError(
                "DragonMLP: cosnet=True is not supported with "
                "tensor_parallel_size > 1."
            )
        hidden_size = config.hidden_size
        intermediate = config.intermediate_size
        self.fc_1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_1",
        )
        self.fc_2 = RowParallelLinear(
            input_size=intermediate,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_2",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_1(x)
        x = self.act_fn(x)
        x, _ = self.fc_2(x)
        return x


class DragonSharedMLP(nn.Module):
    """Shared-expert ReLU² MLP, TP-aware.

    Used as the ``shared_experts`` submodule of ``DragonLatentMoE``. TP
    splits ``up_proj`` column-wise and ``down_proj`` row-wise. Checkpoint
    names (``fc_1`` / ``fc_2``) are remapped to ``up_proj`` / ``down_proj``
    in ``load_weights``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class DragonLatentMoE(nn.Module):
    """vLLM-native replacement for Dragon's ``DragonMoE``.

    Latent (down-projected) MoE with sigmoid routing and a non-gated ReLU²
    expert MLP, modelled on Nemotron-H's ``NemotronHMoE``. Differences from
    Nemotron-H: flat top-k (no expert grouping) and a single shared expert.
    """

    def __init__(self, config, prefix: str = "", quant_config=None):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.moe_routed_scaling_factor
        self.hidden_size = config.hidden_size
        self.latent_size = config.moe_routed_input_dim
        self.num_experts = config.moe_num_routed_experts
        self.top_k = config.moe_num_active_experts
        assert self.latent_size, "DragonLatentMoE requires moe_routed_input_dim"

        self.gate = GateLinear(
            self.hidden_size,
            self.num_experts,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )

        self.fc1_latent_proj = ReplicatedLinear(
            input_size=self.hidden_size,
            output_size=self.latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.fc2_latent_proj = ReplicatedLinear(
            input_size=self.latent_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )

        shared_intermediate = config.moe_shared_intermediate_size
        if shared_intermediate and shared_intermediate > 0:
            self.shared_experts = DragonSharedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                quant_config=quant_config,
            )
        else:
            self.shared_experts = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.latent_size,
            intermediate_size=config.moe_routed_intermediate_size,
            reduce_results=False,
            renormalize=(self.top_k > 1),
            use_grouped_topk=False,
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=activation_without_mul("relu2"),
            is_act_and_mul=False,
            routed_input_transform=self.fc1_latent_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_states = x.reshape(-1, orig_shape[-1])

        router_logits, _ = self.gate(hidden_states)
        shared_output, routed_output = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if hidden_states.dtype != torch.float16:
            routed_output = routed_output * self.routed_scaling_factor
        elif self.shared_experts is not None:
            shared_output = shared_output * (1.0 / self.routed_scaling_factor)

        routed_output, _ = self.fc2_latent_proj(routed_output)

        if self.shared_experts is not None:
            out = routed_output + shared_output
        else:
            out = routed_output

        if self.tp_size > 1:
            out = self.experts.maybe_all_reduce_tensor_model_parallel(out)

        return out.reshape(orig_shape)


# =============================================================================
# Geodesic residual update
# =============================================================================


@triton.jit
def _geodesic_norm_kernel(
    x_ptr,
    g_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    inv_depth,
    clamp_val,
    D,
    stride_x,
    stride_g,
    stride_o,
    BLOCK: tl.constexpr,
):
    """Fused geodesic residual update — one program per token row.

    Single load of ``x``/``g``, fp32 reductions, single store; replaces the
    ~15 small elementwise/reduction kernels of the eager reference (which
    dominate decode launch/memory overhead at 2 calls/layer).
    Uses ``‖g − c·x‖² = ‖g‖² − 2c(x·g) + c²‖x‖²`` to avoid a second pass.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)

    xx = tl.sum(x * x, axis=0)
    xg = tl.sum(x * g, axis=0)
    gg = tl.sum(g * g, axis=0)

    x_norm_sq = tl.maximum(xx, 1e-12)
    coeff = xg / x_norm_sq
    tangent_sq = gg - 2.0 * coeff * xg + coeff * coeff * xx
    tangent_norm = tl.sqrt(tl.maximum(tangent_sq, 0.0))
    safe_tangent = tl.maximum(tangent_norm, 1e-8)
    R = tl.sqrt(xx)
    safe_R = tl.maximum(R, 1e-6)

    scale = tl.load(scale_ptr).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    theta = tl.minimum(safe_tangent / safe_R, clamp_val)
    theta = tl.minimum((theta * scale + bias) * inv_depth, clamp_val)

    cos_t = tl.cos(theta)
    sin_t = tl.sin(theta)
    out = x * cos_t + (g - coeff * x) * (safe_R * sin_t / safe_tangent)
    tl.store(
        out_ptr + row * stride_o + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


class DragonGeodesicNorm(nn.Module):
    """Geodesic residual update: port of Dragon's ``DragonGeodesicNorm``.

    Blends the residual ``x`` with the mixer / MLP output ``g`` along the
    tangent direction on the hypersphere, bounded by a per-layer rotation
    angle ``θ = clamp(safe_tangent_norm / R, self.clamp) * scale + bias``
    then divided by ``layer_idx + 1``.

    Checkpoint tensors: ``scale``, ``bias``, ``prosres_scalar`` (buffer).

    The forward runs a fused Triton kernel (one launch instead of ~15 small
    ops; fp32 intermediates instead of per-op bf16 rounding). Set
    ``DRAGON_FUSED_GEODESIC=0`` to fall back to the eager reference.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("prosres_scalar", torch.tensor(1.0))
        self.clamp = torch.pi / 4.0

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if _FUSED_GEODESIC and x.is_cuda and x.stride(-1) == 1 and g.stride(-1) == 1:
            out = torch.empty_like(x)
            D = x.shape[-1]
            n_rows = x.numel() // D
            _geodesic_norm_kernel[(n_rows,)](
                x,
                g,
                out,
                self.scale,
                self.bias,
                1.0 / (self.layer_idx + 1),
                self.clamp,
                D,
                x.stride(-2) if x.dim() > 1 else 0,
                g.stride(-2) if g.dim() > 1 else 0,
                out.stride(-2) if out.dim() > 1 else 0,
                BLOCK=triton.next_power_of_2(D),
                num_warps=8,
            )
            return out
        return self._forward_ref(x, g)

    def _forward_ref(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proj_coeff = (x * g).sum(dim=-1, keepdim=True) / x_norm_sq
        gradient = g - proj_coeff * x
        tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True)
        safe_tangent_norm = tangent_norm.clamp_min(1e-8)
        unit_tangent = gradient / safe_tangent_norm
        R = torch.norm(x, p=2, dim=-1, keepdim=True)
        safe_R = R.clamp_min(1e-6)
        theta = (safe_tangent_norm / safe_R).clamp_max(self.clamp)
        theta = ((theta * self.scale + self.bias) / (self.layer_idx + 1)).clamp_max(
            self.clamp
        )
        return x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)


# =============================================================================
# Decoder block
# =============================================================================


_SUPPORTED_LAYER_TYPES = {"M", "V"}


class DragonMonoBlock(nn.Module):
    """Dragon single-mixer block (pre-norm, residual, gate, mixer_proj, MLP).

    Supports layer types ``M`` (Mamba3 MIMO) and ``V`` (Differential-TPA);
    other types from ``modeling_dragon.py`` (``g``, ``v``, ``w``, ``t``,
    ``2``) would need their own ported mixers.
    """

    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise NotImplementedError(
                f"DragonMonoBlock: layer type {layer_type!r} not ported yet "
                f"(supported: {sorted(_SUPPORTED_LAYER_TYPES)})."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        # fp8 / quant only applies to the dense GEMMs (gate/output projections
        # and the MLP/MoE). The mixers keep bf16: their projections feed the
        # mamba3/rotary/DiffTPA kernels, which are numerically sensitive and
        # have no fp8 path.
        quant_config = vllm_config.quant_config

        # --- Mixer -----------------------------------------------------------
        if layer_type == "M":
            self.mixer: nn.Module = DragonMamba3Mixer(
                config, vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
            head_dim = self.mixer.headdim
            num_heads = self.mixer.nheads
            self.use_gate = False  # mamba3 has internal z-gate
        else:  # "V"
            self.mixer = DragonDiffTPAAttention(
                config, vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
            head_dim = self.mixer.head_dim
            num_heads = self.mixer.num_signal_heads
            self.use_gate = bool(config.gate_attn)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mixer_out_dim = head_dim * num_heads

        tp = get_tensor_model_parallel_world_size()
        if tp > 1:
            assert self.num_heads % tp == 0, (
                f"DragonMonoBlock: num_heads={self.num_heads} not divisible "
                f"by tp={tp}"
            )
        self.num_heads_local = self.num_heads // tp
        self.mixer_out_dim_local = self.head_dim * self.num_heads_local

        # --- Block-level gate (only for attention-like mixers) --------------
        cosnet = getattr(config, "cosnet", False)
        cosnet_rank = getattr(config, "cosnet_rank", 128)
        if tp > 1 and cosnet:
            raise NotImplementedError(
                "DragonMonoBlock: cosnet=True is not supported with "
                "tensor_parallel_size > 1."
            )
        if self.use_gate:
            gate_type = config.gate_type
            if gate_type != "elementwise":
                raise NotImplementedError(
                    f"DragonMonoBlock gate_type={gate_type!r} not ported "
                    f"yet (only 'elementwise' is supported)."
                )
            # Column-parallel on the output head axis so each rank produces
            # its local ``(num_heads_local * head_dim)`` gate.
            self.gate_proj = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=self.mixer_out_dim,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            self.gate_bias = 1.15 if config.zero_centered_gate else 0.0
            if config.gate_act == "silu":
                self.gate_act: Callable[[torch.Tensor], torch.Tensor] = (
                    torch.nn.functional.silu
                )
            elif config.gate_act == "sigmoid":
                self.gate_act = torch.sigmoid
            else:
                raise NotImplementedError(
                    f"DragonMonoBlock gate_act={config.gate_act!r} not ported."
                )

        # --- Block-level output projection ----------------------------------
        # Row-parallel: input is the sharded mixer output; each rank runs a
        # partial matmul over its own head slice, then allreduces.
        self.mixer_proj = RowParallelLinear(
            input_size=self.mixer_out_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer_proj",
        )

        # --- Pre-norms / residual update ------------------------------------
        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        if config.geodesic_update:
            self.input_norm: nn.Module = nn.Identity()
            self.postmixer_norm: nn.Module = nn.Identity()
            self.geodesic_mixer = DragonGeodesicNorm(layer_idx)
            self.geodesic_mlp = DragonGeodesicNorm(layer_idx)
        else:
            self.input_norm = _DragonNorm(
                config.hidden_size, eps=eps, zero_centered=zc
            )
            self.postmixer_norm = _DragonNorm(
                config.hidden_size, eps=eps, zero_centered=zc
            )
            self.geodesic_mixer = None
            self.geodesic_mlp = None

        # --- MLP ------------------------------------------------------------
        if config.moe:
            self.mlp: nn.Module = DragonLatentMoE(
                config, prefix=f"{prefix}.mlp", quant_config=quant_config
            )
        else:
            if config.mlp_type != "simple":
                raise NotImplementedError(
                    f"DragonMonoBlock: non-MoE mlp_type={config.mlp_type!r} "
                    "not ported (only 'simple' is supported)."
                )
            self.mlp = DragonMLP(
                config, prefix=f"{prefix}.mlp", quant_config=quant_config
            )

        # --- Residual scale ``a/b/lns`` -------------------------------------
        import math as _math

        lns = 1.0
        if config.layer_norm_scaling:
            lns = 1.0 / _math.sqrt(layer_idx + 1)
        self.lns = float(lns)
        if config.use_completed_p:
            depth_ratio = len(config.layers_config) / config.base_depth
            self.a = float(depth_ratio ** (-config.completed_p_alpha))
        else:
            self.a = 1.0
        self.b = 1.0

        self.is_geodesic = bool(config.geodesic_update)

    # ------------------------------------------------------------------
    def _mix(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch to the configured mixer. Input shape: ``(N, D)``."""
        if self.layer_type == "M":
            return self.mixer(hidden_states)
        # "V" — differential-TPA.
        return self.mixer(positions, hidden_states)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Flat-token forward: ``(N, D)`` in, ``(N, D)`` out."""
        n = hidden_states.shape[0]

        # -- Mixer path -------------------------------------------------------
        residual = hidden_states
        x = self.lns * self.input_norm(hidden_states)
        y_mix_flat = _timed(f"mixer_{self.layer_type}", lambda: self._mix(x, positions))  # (N, H_local*D)
        if self.use_gate:
            g_all, _ = self.gate_proj(x)
            g = g_all.view(n, self.num_heads_local, self.head_dim)
            g = self.gate_act(g + self.gate_bias).to(y_mix_flat.dtype)
            y_mix_flat = y_mix_flat.view(n, self.num_heads_local, self.head_dim)
            y_mix_flat = y_mix_flat * g
            y_mix_flat = y_mix_flat.reshape(n, self.mixer_out_dim_local)
        y_mix, _ = self.mixer_proj(y_mix_flat)

        if self.is_geodesic:
            hidden_states = _timed("geodesic", lambda: self.geodesic_mixer(residual, y_mix))
        else:
            hidden_states = self.b * residual + self.a * y_mix

        # -- MLP path --------------------------------------------------------
        residual = hidden_states
        x = self.lns * self.postmixer_norm(hidden_states)
        y_mlp = _timed("mlp/moe", lambda: self.mlp(x))
        if self.is_geodesic:
            hidden_states = _timed("geodesic", lambda: self.geodesic_mlp(residual, y_mlp))
        else:
            hidden_states = self.b * residual + self.a * y_mlp

        return hidden_states


# =============================================================================
# Top-level model
# =============================================================================


def _iter_dragon_layers(
    config,
) -> Iterator[tuple[int, str]]:
    """Yield ``(layer_idx, layer_type)`` for each block in ``layers_config``."""
    for i, ch in enumerate(config.layers_config):
        yield i, ch


class DragonModel(nn.Module):
    """Token embedding → stack of ``DragonMonoBlock`` → optional final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        def _make(prefix: str) -> DragonMonoBlock:
            idx = int(prefix.rsplit(".", 1)[1])
            layer_type = config.layers_config[idx]
            return DragonMonoBlock(
                config=config,
                vllm_config=vllm_config,
                layer_idx=idx,
                layer_type=layer_type,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.layers_config), _make, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

        if config.final_norm and get_pp_group().is_last_rank:
            eps = config.norm_epsilon
            zc = getattr(config, "zero_centered_gamma", False)
            self.final_norm: nn.Module = _DragonNorm(
                config.hidden_size, eps=eps, zero_centered=zc
            )
        else:
            self.final_norm = PPMissingLayer() if not get_pp_group().is_last_rank \
                else nn.Identity()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if self.config.normalize_embeddings_ngpt:
                hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)
            if self.config.normalize_embeddings:
                import math as _m
                hidden_states = torch.nn.functional.normalize(
                    hidden_states, dim=-1
                ) * _m.sqrt(self.config.hidden_size)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        from itertools import islice
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions=positions, hidden_states=hidden_states)

        if _DPROF:
            _dump_prof()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if isinstance(self.final_norm, nn.Identity):
            return hidden_states
        return self.final_norm(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # DragonLatentMoE uses non-gated experts (only w1/w2), so only one
        # ckpt→param mapping is needed per shard role. We remap the Dragon
        # checkpoint ``experts.experts`` → FusedMoE's ``w1`` (up_proj stand-in)
        # and ``experts.output_experts`` → ``w2`` explicitly in load_weights,
        # so no SharedFusedMoE.make_expert_params_mapping is needed.
        return []


# =============================================================================
# CausalLM
# =============================================================================


def _moe_layer_indices(config) -> list[int]:
    """Return the layer indices that have a MoE MLP.

    Dragon's ``layers_mlp_config`` is ``''`` → all layers MoE (if
    ``config.moe``) or all dense; otherwise it is a per-layer ``'m'``/``'d'``
    string.
    """
    if config.layers_mlp_config:
        return [i for i, c in enumerate(config.layers_mlp_config) if c == "m"]
    return (
        list(range(len(config.layers_config))) if config.moe else []
    )


class DragonForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
    SupportsLoRA,
):
    """vLLM-native Dragon causal LM with paged KV + paged SSM state."""

    # Dragon uses custom projection names (c_q / W_A_k / in_proj / in_proj_dyn
    # / …), none of which map onto vLLM's default qkv_proj / gate_up_proj
    # packing. Leave empty to disable the default packed-modules handling.
    packed_modules_mapping: dict[str, list[str]] = {}

    # LoRA-visible modules.
    embedding_modules = {
        "embedding": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # ------------------------------------------------------------------
    # MambaBase / hybrid state config surface
    # ------------------------------------------------------------------
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        # Representative: the larger Mamba3 state (4 tensors).
        return MambaStateDtypeCalculator.mamba3_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        d_inner = 2 * hf_config.hidden_size
        nheads = d_inner // hf_config.mamba_headdim
        rope_fraction = 0.5
        split = int(hf_config.mamba_d_state * rope_fraction)
        if split % 2:
            split -= 1
        num_rope_angles = split // 2
        return MambaStateShapeCalculator.mamba3_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            num_heads=nheads,
            head_dim=hf_config.mamba_headdim,
            d_state=hf_config.mamba_d_state,
            mimo_dim=hf_config.mamba_mimo_dim,
            num_rope_angles=num_rope_angles,
        )

    # ------------------------------------------------------------------
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config

        self.model = DragonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_lm_head or getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embedding.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # MixtureOfExperts bookkeeping.
        self.expert_weights: list[object] = []
        self.moe_layers: list[nn.Module] = []
        example_moe: DragonLatentMoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, DragonMonoBlock) and isinstance(
                layer.mlp, DragonLatentMoE
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)
        if example_moe is not None:
            experts = example_moe.experts  # SharedFusedMoE
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_shared_experts = (
                1 if example_moe.shared_experts is not None else 0
            )
            self.num_logical_experts = experts.logical_num_experts
            self.num_physical_experts = experts.global_num_experts
            self.num_local_physical_experts = experts.local_num_experts
            self.num_routed_experts = experts.logical_num_experts
            self.num_redundant_experts = (
                experts.global_num_experts - experts.logical_num_experts
            )
        else:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0

    # ------------------------------------------------------------------
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, DragonMonoBlock) and isinstance(
                layer.mlp, DragonLatentMoE
            ):
                moe = layer.mlp.experts
                moe.local_num_experts = num_local_physical_experts
                moe.global_num_experts = num_physical_experts
                moe.update_expert_map()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load a Dragon safetensors checkpoint.

        Remaps:
            - ``mlp.moe_gate.weight`` → ``mlp.gate.weight``
            - ``mlp.down_proj.weight`` (latent down) → ``mlp.fc1_latent_proj.weight``
            - ``mlp.up_proj.weight``   (latent up)   → ``mlp.fc2_latent_proj.weight``
            - ``mlp.expert_bias``      → ``mlp.gate.e_score_correction_bias``
            - ``mlp.shared_experts.fc_1/fc_2`` → ``shared_experts.up_proj/down_proj``
            - ``mlp.experts.experts``  → FusedMoE ``w13`` (w1 shard)
            - ``mlp.experts.output_experts`` → FusedMoE ``w2`` (w2 shard)
            - ``mlp.tokens_per_expert`` — skipped (non-persistent buffer)
        """
        moe_layers = set(_moe_layer_indices(self.config))

        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()

        scalar_moe_remap = {
            "moe_gate.weight": "gate.weight",
            "down_proj.weight": "fc1_latent_proj.weight",
            "up_proj.weight": "fc2_latent_proj.weight",
            "shared_experts.fc_1.weight": "shared_experts.up_proj.weight",
            "shared_experts.fc_2.weight": "shared_experts.down_proj.weight",
        }

        for name, loaded_weight in weights:
            # --- MoE-layer-scoped remaps ------------------------------------
            handled = False
            if name.startswith("model.layers.") and ".mlp." in name:
                head, tail = name.split(".mlp.", 1)
                try:
                    layer_idx = int(head.rsplit(".", 1)[1])
                except ValueError:
                    layer_idx = -1
                if layer_idx in moe_layers:
                    tgt_prefix = f"model.layers.{layer_idx}.mlp"
                    if tail in (
                        "experts.experts.weight",
                        "experts.output_experts.weight",
                    ):
                        # Shard per-expert weights into FusedMoE's packed
                        # ``w13_weight`` / ``w2_weight`` parameters.
                        shard_id = (
                            "w1"
                            if tail.endswith("experts.experts.weight")
                            else "w2"
                        )
                        target = (
                            f"{tgt_prefix}.experts."
                            + ("w13_weight" if shard_id == "w1" else "w2_weight")
                        )
                        param = params_dict[target]
                        weight_loader = param.weight_loader
                        for expert_id in range(loaded_weight.shape[0]):
                            weight_loader(
                                param,
                                loaded_weight[expert_id],
                                target,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                        loaded.add(target)
                        handled = True
                    elif tail == "expert_bias":
                        target = f"{tgt_prefix}.gate.e_score_correction_bias"
                        param = params_dict[target]
                        param.data.copy_(
                            loaded_weight.to(param.dtype).to(param.device)
                        )
                        loaded.add(target)
                        handled = True
                    elif tail in scalar_moe_remap:
                        target = f"{tgt_prefix}.{scalar_moe_remap[tail]}"
                        param = params_dict[target]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded.add(target)
                        handled = True
                    elif tail == "tokens_per_expert":
                        handled = True  # drop
                    elif tail.startswith("shared_experts."):
                        # Fall through to default-loader path under remapped name.
                        pass

            if handled:
                continue

            # --- Default path ----------------------------------------------
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded.add(name)
            elif name in buffers_dict:
                buf = buffers_dict[name]
                buf.data.copy_(loaded_weight.to(buf.dtype).to(buf.device))
                loaded.add(name)
            else:
                logger.warning_once(
                    "Dragon load_weights: parameter %s not found, skipping.",
                    name,
                )

        return loaded

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
