# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-channel attention backend for Dragon's DifferentialTPA token-shift.

Dragon's V (DifferentialTPA) layers mix the *previous* token's raw K/V into
the current step via an EMA-style shift:

    k_shifted = alpha_k * k_prev + (1 - alpha_k) * k_curr
    v_shifted = alpha_v * v_prev + (1 - alpha_v) * v_curr

The main softmax attention still runs on paged KV via
``vllm.attention.Attention``; this backend only manages the tiny one-token
``(k_prev, v_prev)`` buffer held per request. Each V layer registers a
second mamba-style KV cache with this backend (on top of its paged
attention cache). The metadata exposes state slot indices and the prefill
query layout so the layer can read/write the 1-token buffer at the right
token position (last token of each prefill chunk, or the current token
during decode).
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class DragonDiffTPABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DRAGON_DIFF_TPA"

    @staticmethod
    def get_builder_cls() -> type["DragonDiffTPAMetadataBuilder"]:
        return DragonDiffTPAMetadataBuilder


@dataclass
class DragonDiffTPAMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    # Per-request state slot into the (k_last, v_last) pool. Shape: [batch].
    # Same slot applies to both decode and prefill requests.
    state_indices_tensor: torch.Tensor | None = None

    # For prefills: have we carried over a non-empty last-token k/v from a
    # previous chunk of the same request? Shape: [num_prefills].
    has_initial_state: torch.Tensor | None = None

    # Prefill token start offsets in the flattened stream, rebased to 0.
    # Shape: [num_prefills + 1].
    query_start_loc_p: torch.Tensor | None = None

    # CPU mirrors (built once per step; avoids per-layer GPU syncs in the
    # token-shift prefill path).
    has_initial_state_cpu: torch.Tensor | None = None
    query_start_loc_p_cpu: torch.Tensor | None = None


class DragonDiffTPAMetadataBuilder(
    AttentionMetadataBuilder[DragonDiffTPAMetadata]
):
    # The shift op itself is tiny and cudagraph-friendly for pure-decode
    # batches; prefill needs a gather/scatter over variable-length chunks
    # so we stay in the safe decode-only capture regime.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.device = device

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> DragonDiffTPAMetadata:
        m = common_attn_metadata

        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        state_indices_tensor = block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
        )

        has_initial_state = None
        query_start_loc_p = None
        has_initial_state_cpu = None
        query_start_loc_p_cpu = None
        if num_prefills > 0:
            context_lens = m.compute_num_computed_tokens()
            has_initial_state = context_lens[num_decodes:] > 0
            qsl = m.query_start_loc[num_decodes:]
            query_start_loc_p = qsl - qsl[0]
            # One sync at build time instead of per V-layer downstream.
            has_initial_state_cpu = has_initial_state.to("cpu")
            qsl_cpu = m.query_start_loc_cpu[num_decodes:]
            query_start_loc_p_cpu = qsl_cpu - qsl_cpu[0]

        return DragonDiffTPAMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            state_indices_tensor=state_indices_tensor,
            has_initial_state=has_initial_state,
            query_start_loc_p=query_start_loc_p,
            has_initial_state_cpu=has_initial_state_cpu,
            query_start_loc_p_cpu=query_start_loc_p_cpu,
        )
