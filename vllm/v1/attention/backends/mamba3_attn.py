# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend for Dragon's Mamba3 MIMO mixer.

The Mamba3 layer keeps 4 per-request temporal state tensors
(angle_state, ssm_state, k_state, v_state). There is no causal conv1d,
so this backend is much simpler than GDN's: the metadata only needs
prefill/decode split, state slot lookup, and prefill query offsets.

Speculative decoding and full-cudagraph capture are intentionally
unsupported in the first iteration — they can be layered on once the
non-spec path is validated end-to-end.
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


class Mamba3AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA3"

    @staticmethod
    def get_builder_cls() -> type["Mamba3AttentionMetadataBuilder"]:
        return Mamba3AttentionMetadataBuilder


@dataclass
class Mamba3AttentionMetadata:
    # Batch-level counts.
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    # Per-sequence state slot index into the Mamba state pool.
    # Shape: [batch]. Valid for all requests (prefill + decode).
    state_indices_tensor: torch.Tensor | None = None

    # Prefill-only: does the request have a non-empty cached state
    # carried over from a previous chunk? Shape: [num_prefills].
    has_initial_state: torch.Tensor | None = None

    # Prefill query start offsets in the flattened token stream.
    # Shape: [num_prefills + 1].
    query_start_loc_p: torch.Tensor | None = None

    # CPU mirrors, computed once at build time so per-layer code never has to
    # sync on the GPU tensors (the model runs 29 M-layers per step; a per-layer
    # .item()/.tolist() costs a full pipeline stall each).
    has_initial_state_cpu: torch.Tensor | None = None
    has_initial_any: bool = False
    query_start_loc_p_cpu: torch.Tensor | None = None


class Mamba3AttentionMetadataBuilder(
    AttentionMetadataBuilder[Mamba3AttentionMetadata]
):
    # Mamba3 prefill uses a custom TileLang kernel that is not cudagraph-
    # capturable; decode-only piecewise capture still works.
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
    ) -> Mamba3AttentionMetadata:
        m = common_attn_metadata

        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        # One state slot per request — no spec decode in this backend.
        state_indices_tensor = block_table_tensor[:, 0]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
        )

        has_initial_state = None
        query_start_loc_p = None
        has_initial_state_cpu = None
        has_initial_any = False
        query_start_loc_p_cpu = None
        if num_prefills > 0:
            context_lens = m.compute_num_computed_tokens()
            # Prefill requests are placed after decode requests by the
            # common batch reorder (see AttentionMetadataBuilder contract).
            has_initial_state = context_lens[num_decodes:] > 0
            # query_start_loc covers [decodes..prefills]; take the prefill
            # tail and rebase to 0.
            qsl = m.query_start_loc[num_decodes:]
            query_start_loc_p = qsl - qsl[0]
            # CPU mirrors: one sync here (prefill steps only) instead of one
            # per M-layer downstream.
            has_initial_state_cpu = has_initial_state.to("cpu")
            has_initial_any = bool(has_initial_state_cpu.any())
            qsl_cpu = m.query_start_loc_cpu[num_decodes:]
            query_start_loc_p_cpu = qsl_cpu - qsl_cpu[0]

        return Mamba3AttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            state_indices_tensor=state_indices_tensor,
            has_initial_state=has_initial_state,
            query_start_loc_p=query_start_loc_p,
            has_initial_state_cpu=has_initial_state_cpu,
            has_initial_any=has_initial_any,
            query_start_loc_p_cpu=query_start_loc_p_cpu,
        )
