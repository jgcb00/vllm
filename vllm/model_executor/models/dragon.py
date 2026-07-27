# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Dragon model for vLLM.

Dragon is a hybrid stack:

* ``M`` layers — :class:`DragonMamba3Mixer`, an SSM with four temporal states
  paged through the MAMBA3 backend.
* ``V`` layers — :class:`DragonDiffTPAAttention`, differential tensor-product
  softmax attention over paged KV, plus a one-token shift buffer paged through
  the DRAGON_DIFF_TPA side-channel backend.
* MLP — :class:`DragonLatentMoE` (latent projection, 256 routed experts, one
  shared expert) or the dense :class:`DragonMLP`.

Known fidelity gaps, flagged rather than papered over:

* vLLM's ``get_rope`` stands in for Dragon's custom p-rope.
* ``scalable_softmax`` pre-scales q but the softmax scale itself is vLLM's;
  sliding-window bounds only the log position, not the attention mask.
* ``token_conv1d_attn``, ``xsa``, ``use_value_embedding`` and ``cosnet`` are
  not ported.
* TP splits the head-interleaved projections on the head axis. The Mamba3
  post-gate norm, when enabled, is a per-rank shard norm, which differs
  numerically from a global RMSNorm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    activation_without_mul,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.dragon import (
    DragonDiffTPAAttention,
    DragonMamba3Mixer,
    DragonNorm,
)
from vllm.model_executor.layers.mamba.dragon.state import (
    mamba3_state_dtype,
    mamba3_state_shape,
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

SUPPORTED_LAYER_TYPES = frozenset({"M", "V"})


class DragonMLP(nn.Module):
    """Dense ReLU² MLP (``fc_1 -> relu² -> fc_2``) in Dragon's naming."""

    def __init__(self, config, prefix: str = "", quant_config=None):
        super().__init__()
        self.fc_1 = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_1",
        )
        self.fc_2 = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
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
    """Shared-expert ReLU² MLP.

    The checkpoint's ``fc_1``/``fc_2`` are remapped to ``up_proj``/``down_proj``
    in :meth:`DragonForCausalLM.load_weights`. Results are not reduced here —
    the MoE runner all-reduces the combined shared plus routed output once.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
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
            reduce_results=False,
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
    """Latent MoE with sigmoid routing and non-gated ReLU² experts.

    Tokens are projected down to ``moe_routed_input_dim`` before the experts
    and back up afterwards; the MoE runner owns both transforms, the routed
    scaling factor, the shared-expert add and the all-reduce.
    """

    def __init__(self, config, parallel_config, prefix: str = "", quant_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_size = config.moe_routed_input_dim
        self.num_experts = config.moe_num_routed_experts
        self.top_k = config.moe_num_active_experts
        if not self.latent_size:
            raise ValueError("DragonLatentMoE requires moe_routed_input_dim")

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

        shared_intermediate = config.moe_shared_intermediate_size
        if shared_intermediate and shared_intermediate > 0:
            self.shared_experts = DragonSharedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                prefix=f"{prefix}.shared_experts",
                quant_config=quant_config,
            )
        else:
            self.shared_experts = None

        # Registered before `experts` so the latent projections keep their own
        # parameter names: the runner holds the same module objects, and
        # named_parameters reports whichever path it reaches first.
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

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.latent_size,
            intermediate_size=config.moe_routed_intermediate_size,
            renormalize=self.top_k > 1,
            use_grouped_topk=False,
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=activation_without_mul("relu2"),
            routed_input_transform=self.fc1_latent_proj,
            routed_output_transform=self.fc2_latent_proj,
            routed_scaling_factor=config.moe_routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_states = x.reshape(-1, orig_shape[-1])
        router_logits, _ = self.gate(hidden_states)
        out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return out.reshape(orig_shape)


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
    """Fused geodesic residual update, one program per token row.

    A single load of ``x``/``g``, fp32 reductions and a single store replace
    the ~15 small elementwise/reduction kernels of the eager reference, which
    otherwise dominate decode launch overhead at two calls per layer. Uses
    ``||g - c*x||² = ||g||² - 2c(x·g) + c²||x||²`` to avoid a second pass.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)

    xx = tl.sum(x * x, axis=0)
    xg = tl.sum(x * g, axis=0)
    gg = tl.sum(g * g, axis=0)

    coeff = xg / tl.maximum(xx, 1e-12)
    tangent_sq = gg - 2.0 * coeff * xg + coeff * coeff * xx
    safe_tangent = tl.maximum(tl.sqrt(tl.maximum(tangent_sq, 0.0)), 1e-8)
    R = tl.sqrt(xx)
    safe_R = tl.maximum(R, 1e-6)

    scale = tl.load(scale_ptr).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    theta = tl.minimum(safe_tangent / safe_R, clamp_val)
    theta = tl.minimum((theta * scale + bias) * inv_depth, clamp_val)

    out = x * tl.cos(theta) + (g - coeff * x) * (safe_R * tl.sin(theta) / safe_tangent)
    tl.store(
        out_ptr + row * stride_o + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


class DragonGeodesicNorm(nn.Module):
    """Geodesic residual update, port of Dragon's ``DragonGeodesicNorm``.

    Blends the residual ``x`` with the mixer/MLP output ``g`` along the tangent
    direction on the hypersphere, bounded by a per-layer rotation angle
    ``theta = clamp(tangent_norm / R) * scale + bias``, divided by
    ``layer_idx + 1``. Checkpoint tensors: ``scale``, ``bias``, and the
    ``prosres_scalar`` buffer.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("prosres_scalar", torch.tensor(1.0))
        self.clamp = torch.pi / 4.0

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and x.stride(-1) == 1 and g.stride(-1) == 1):
            return self.forward_native(x, g)
        out = torch.empty_like(x)
        D = x.shape[-1]
        _geodesic_norm_kernel[(x.numel() // D,)](
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
            num_warps=2,  # measured fastest at batch 1 (1.56 us) and batch 256
        )
        return out

    def forward_native(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Eager reference. Kept as the non-CUDA path and the parity baseline."""
        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proj_coeff = (x * g).sum(dim=-1, keepdim=True) / x_norm_sq
        gradient = g - proj_coeff * x
        safe_tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        unit_tangent = gradient / safe_tangent_norm
        safe_R = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        theta = (safe_tangent_norm / safe_R).clamp_max(self.clamp)
        theta = ((theta * self.scale + self.bias) / (self.layer_idx + 1)).clamp_max(
            self.clamp
        )
        return x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)


class DragonMonoBlock(nn.Module):
    """Single-mixer block: pre-norm, mixer, optional gate, projection, MLP."""

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
        if layer_type not in SUPPORTED_LAYER_TYPES:
            raise NotImplementedError(
                f"DragonMonoBlock: layer type {layer_type!r} is not ported "
                f"(supported: {sorted(SUPPORTED_LAYER_TYPES)})."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        # Quantization applies only to the dense GEMMs — the block gate and
        # output projections plus the MLP/MoE. The mixers stay bf16: their
        # projections feed the mamba3/rotary/attention kernels, which are
        # numerically sensitive and have no fp8 path.
        quant_config = vllm_config.quant_config

        if layer_type == "M":
            self.mixer: nn.Module = DragonMamba3Mixer(
                config, vllm_config=vllm_config, prefix=f"{prefix}.mixer"
            )
            head_dim = self.mixer.headdim
            num_heads = self.mixer.nheads
            self.use_gate = False  # mamba3 gates internally via z
        else:
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
        if self.num_heads % tp:
            raise ValueError(
                f"DragonMonoBlock: num_heads={self.num_heads} is not divisible "
                f"by tensor_parallel_size={tp}"
            )
        self.num_heads_local = self.num_heads // tp
        self.mixer_out_dim_local = self.head_dim * self.num_heads_local

        if self.use_gate:
            if config.gate_type != "elementwise":
                raise NotImplementedError(
                    f"DragonMonoBlock: gate_type={config.gate_type!r} is not "
                    f"ported (only 'elementwise' is supported)."
                )
            # Column-parallel on the output head axis, so each rank produces
            # its own (num_heads_local * head_dim) gate.
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
                    f"DragonMonoBlock: gate_act={config.gate_act!r} is not ported."
                )

        # Row-parallel: each rank matmuls its own head slice, then all-reduces.
        self.mixer_proj = RowParallelLinear(
            input_size=self.mixer_out_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer_proj",
        )

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        self.is_geodesic = bool(config.geodesic_update)
        if self.is_geodesic:
            self.input_norm: nn.Module = nn.Identity()
            self.postmixer_norm: nn.Module = nn.Identity()
            self.geodesic_mixer = DragonGeodesicNorm(layer_idx)
            self.geodesic_mlp = DragonGeodesicNorm(layer_idx)
        else:
            self.input_norm = DragonNorm(config.hidden_size, eps=eps, zero_centered=zc)
            self.postmixer_norm = DragonNorm(
                config.hidden_size, eps=eps, zero_centered=zc
            )

        if config.moe:
            self.mlp: nn.Module = DragonLatentMoE(
                config,
                vllm_config.parallel_config,
                prefix=f"{prefix}.mlp",
                quant_config=quant_config,
            )
        else:
            if config.mlp_type != "simple":
                raise NotImplementedError(
                    f"DragonMonoBlock: mlp_type={config.mlp_type!r} is not "
                    f"ported (only 'simple' is supported)."
                )
            self.mlp = DragonMLP(
                config, prefix=f"{prefix}.mlp", quant_config=quant_config
            )

        # Residual scales. lns == 1.0 and the identity norms of geodesic mode
        # would still launch a scalar-mul kernel per call — 72 wasted kernels
        # per step across the stack — so the multiply is skipped when it is a
        # no-op (see forward).
        self.lns = 1.0 / math.sqrt(layer_idx + 1) if config.layer_norm_scaling else 1.0
        if config.use_completed_p:
            depth_ratio = len(config.layers_config) / config.base_depth
            self.a = float(depth_ratio ** (-config.completed_p_alpha))
        else:
            self.a = 1.0
        self.b = 1.0

    def _mix(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        if self.layer_type == "M":
            return self.mixer(hidden_states)
        return self.mixer(positions, hidden_states)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Flat-token forward: ``(N, D)`` in, ``(N, D)`` out."""
        n = hidden_states.shape[0]

        residual = hidden_states
        x = self.input_norm(hidden_states)
        if self.lns != 1.0:
            x = self.lns * x
        y_mix = self._mix(x, positions)  # (N, H_local * D)
        if self.use_gate:
            g_all, _ = self.gate_proj(x)
            g = self.gate_act(
                g_all.view(n, self.num_heads_local, self.head_dim) + self.gate_bias
            ).to(y_mix.dtype)
            y_mix = (y_mix.view(n, self.num_heads_local, self.head_dim) * g).reshape(
                n, self.mixer_out_dim_local
            )
        y_mix, _ = self.mixer_proj(y_mix)

        if self.is_geodesic:
            hidden_states = self.geodesic_mixer(residual, y_mix)
        else:
            hidden_states = self.b * residual + self.a * y_mix

        residual = hidden_states
        x = self.postmixer_norm(hidden_states)
        if self.lns != 1.0:
            x = self.lns * x
        y_mlp = self.mlp(x)
        if self.is_geodesic:
            return self.geodesic_mlp(residual, y_mlp)
        return self.b * residual + self.a * y_mlp


class DragonModel(nn.Module):
    """Token embedding, a stack of :class:`DragonMonoBlock`, optional norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embeddings_prenormalized = False

        self.embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        def build_block(prefix: str) -> DragonMonoBlock:
            idx = int(prefix.rsplit(".", 1)[1])
            return DragonMonoBlock(
                config=config,
                vllm_config=vllm_config,
                layer_idx=idx,
                layer_type=config.layers_config[idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.layers_config), build_block, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        if not get_pp_group().is_last_rank:
            self.final_norm: nn.Module = PPMissingLayer()
        elif config.final_norm:
            self.final_norm = DragonNorm(
                config.hidden_size,
                eps=config.norm_epsilon,
                zero_centered=getattr(config, "zero_centered_gamma", False),
            )
        else:
            self.final_norm = nn.Identity()

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
            from_table = inputs_embeds is None
            hidden_states = (
                self.embed_input_ids(input_ids) if from_table else inputs_embeds
            )
            if self.config.normalize_embeddings_ngpt and not (
                from_table and self.embeddings_prenormalized
            ):
                # Table rows are pre-normalized at load time where possible;
                # externally supplied embeddings still normalize here.
                hidden_states = torch.nn.functional.normalize(hidden_states, dim=-1)
            if self.config.normalize_embeddings:
                hidden_states = torch.nn.functional.normalize(
                    hidden_states, dim=-1
                ) * math.sqrt(self.config.hidden_size)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions=positions, hidden_states=hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.final_norm(hidden_states)


def moe_layer_indices(config) -> list[int]:
    """Indices of the layers that carry a MoE MLP.

    ``layers_mlp_config`` is either empty — every layer follows ``config.moe``
    — or a per-layer ``'m'``/``'d'`` string.
    """
    if config.layers_mlp_config:
        return [i for i, c in enumerate(config.layers_mlp_config) if c == "m"]
    return list(range(len(config.layers_config))) if config.moe else []


class DragonForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
    SupportsLoRA,
):
    """Dragon causal LM with paged KV and paged recurrent state."""

    # Dragon's projection names (c_q / W_A_k / in_proj / in_proj_dyn / ...) do
    # not map onto vLLM's qkv_proj / gate_up_proj packing, so the default
    # packed-module handling is disabled.
    packed_modules_mapping: dict[str, list[str]] = {}

    embedding_modules = {
        "embedding": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        # Representative: the larger Mamba3 state.
        return mamba3_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        hf_config = vllm_config.model_config.hf_config
        d_inner = 2 * hf_config.hidden_size
        split = int(hf_config.mamba_d_state * 0.5)  # rope_fraction
        if split % 2:
            split -= 1
        return mamba3_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            num_heads=d_inner // hf_config.mamba_headdim,
            head_dim=hf_config.mamba_headdim,
            d_state=hf_config.mamba_d_state,
            mimo_dim=hf_config.mamba_mimo_dim,
            num_rope_angles=split // 2,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        # Align mode assumes every mamba layer exposes the same number of
        # state tensors; Dragon's M layers have four and its V layers two, so
        # the single flat tuple this protocol expects cannot describe it.
        # DragonForCausalLMConfig rejects --mamba-cache-mode align up front.
        raise NotImplementedError(
            "Dragon does not support mamba prefix caching (align mode): its M "
            "and V layers hold different numbers of state tensors."
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if getattr(config, "cosnet", False):
            # The CosNet sidecar branch is not ported; accepting the flag
            # would silently drop a trained residual path.
            raise NotImplementedError(
                "DragonForCausalLM: config.cosnet is not supported."
            )
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

        self._init_moe_metadata()

    def _init_moe_metadata(self) -> None:
        """Populate the MixtureOfExperts bookkeeping surface."""
        self.expert_weights: list[object] = []
        self.moe_layers: list[nn.Module] = []
        example: DragonLatentMoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, DragonMonoBlock) and isinstance(
                layer.mlp, DragonLatentMoE
            ):
                example = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        routed = example.experts.routed_experts
        self.num_expert_groups = 1
        self.num_shared_experts = 1 if example.shared_experts is not None else 0
        self.num_logical_experts = example.num_experts
        self.num_physical_experts = routed.global_num_experts
        self.num_local_physical_experts = routed.local_num_experts
        self.num_routed_experts = example.num_experts
        self.num_redundant_experts = routed.global_num_experts - example.num_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Dragon's experts are non-gated (w1/w2 only) and arrive packed in a
        # single tensor per layer, which load_weights scatters directly. No
        # checkpoint-name mapping applies, and an empty list keeps the generic
        # machinery from picking up RoutedExperts' gated default.
        return []

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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
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
        for runner in self.moe_layers:
            routed = runner.routed_experts
            routed.local_num_experts = num_local_physical_experts
            routed.global_num_experts = num_physical_experts
            runner.update_expert_map()

    def _expert_params(self) -> dict[int, tuple[nn.Parameter, nn.Parameter]]:
        """Map layer index to that layer's packed ``(w13, w2)`` parameters.

        Resolved by walking the modules rather than by name: the expert
        weights live on the runner's ``RoutedExperts`` submodule, whose path
        is an implementation detail of the MoE composition.
        """
        params: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, DragonMonoBlock) and isinstance(
                layer.mlp, DragonLatentMoE
            ):
                routed = layer.mlp.experts.routed_experts
                params[idx] = (routed.w13_weight, routed.w2_weight)
        return params

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load a Dragon safetensors checkpoint.

        Remaps, for MoE layers:
            ``mlp.moe_gate.weight``            -> ``mlp.gate.weight``
            ``mlp.down_proj.weight``           -> ``mlp.fc1_latent_proj.weight``
            ``mlp.up_proj.weight``             -> ``mlp.fc2_latent_proj.weight``
            ``mlp.expert_bias``                -> ``mlp.gate.e_score_correction_bias``
            ``mlp.shared_experts.fc_1/fc_2``   -> ``shared_experts.up_proj/down_proj``
            ``mlp.experts.experts``            -> packed ``w13`` (w1 shard)
            ``mlp.experts.output_experts``     -> packed ``w2``
            ``mlp.tokens_per_expert``          -> dropped (non-persistent buffer)
        """
        moe_layers = set(moe_layer_indices(self.config))
        expert_params = self._expert_params()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()

        renames = {
            "moe_gate.weight": "gate.weight",
            "down_proj.weight": "fc1_latent_proj.weight",
            "up_proj.weight": "fc2_latent_proj.weight",
            "shared_experts.fc_1.weight": "shared_experts.up_proj.weight",
            "shared_experts.fc_2.weight": "shared_experts.down_proj.weight",
        }

        for name, loaded_weight in weights:
            handled = False
            if name.startswith("model.layers.") and ".mlp." in name:
                head, tail = name.split(".mlp.", 1)
                try:
                    layer_idx = int(head.rsplit(".", 1)[1])
                except ValueError:
                    layer_idx = -1
                if layer_idx in moe_layers:
                    target_prefix = f"model.layers.{layer_idx}.mlp"
                    if tail in (
                        "experts.experts.weight",
                        "experts.output_experts.weight",
                    ):
                        # Dragon packs every expert into one tensor; scatter it
                        # into the fused w13/w2 parameters expert by expert.
                        # NOTE: RoutedExperts.weight_loader dispatches on the
                        # weight NAME ("weight" substring), so the target must
                        # carry the real parameter name.
                        is_w1 = tail.endswith("experts.experts.weight")
                        param = expert_params[layer_idx][0 if is_w1 else 1]
                        shard_id = "w1" if is_w1 else "w2"
                        target = (
                            f"{target_prefix}.experts."
                            f"{'w13_weight' if is_w1 else 'w2_weight'}"
                        )
                        for expert_id in range(loaded_weight.shape[0]):
                            success = param.weight_loader(
                                param,
                                loaded_weight[expert_id],
                                target,
                                shard_id=shard_id,
                                expert_id=expert_id,
                                return_success=True,
                            )
                            if not success:
                                raise RuntimeError(
                                    f"Dragon load_weights: expert weight "
                                    f"{target} (expert {expert_id}) was not "
                                    f"accepted by the loader."
                                )
                        loaded.add(target)
                        handled = True
                    elif tail == "expert_bias":
                        target = f"{target_prefix}.gate.e_score_correction_bias"
                        param = params_dict[target]
                        param.data.copy_(loaded_weight.to(param.dtype).to(param.device))
                        loaded.add(target)
                        handled = True
                    elif tail in renames:
                        target = f"{target_prefix}.{renames[tail]}"
                        param = params_dict[target]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded.add(target)
                        handled = True
                    elif tail == "tokens_per_expert":
                        handled = True  # non-persistent buffer, dropped

            if handled:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
            elif name in buffers_dict:
                buf = buffers_dict[name]
                buf.data.copy_(loaded_weight.to(buf.dtype).to(buf.device))
                loaded.add(name)
            else:
                logger.warning_once(
                    "Dragon load_weights: parameter %s not found, skipping.", name
                )

        self._maybe_prenormalize_embeddings()
        return loaded

    def _maybe_prenormalize_embeddings(self) -> None:
        """Fold the nGPT embedding normalization into the table.

        The lookup only selects rows, so L2-normalizing them offline in fp32 is
        exactly equivalent to ``F.normalize`` after lookup — and slightly more
        precise than normalizing bf16 activations every step. Only valid when
        the lm_head is untied, since normalizing would otherwise change the
        logits.
        """
        if not getattr(self.config, "normalize_embeddings_ngpt", False):
            return
        if getattr(self.config, "tie_lm_head", False) or getattr(
            self.config, "tie_word_embeddings", False
        ):
            return
        weight = self.model.embedding.weight
        with torch.no_grad():
            weight.copy_(
                torch.nn.functional.normalize(weight.float(), dim=-1).to(weight.dtype)
            )
        self.model.embeddings_prenormalized = True
