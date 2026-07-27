# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-channel backend for Dragon's Differential-TPA token shift.

Dragon's ``V`` layers mix the previous token's raw K/V into the current step:

    k_shifted = alpha_k * k_prev + (1 - alpha_k) * k_curr
    v_shifted = alpha_v * v_prev + (1 - alpha_v) * v_curr

The main softmax attention still runs on paged KV through ``Attention``; this
backend owns only the one-token ``(k_prev, v_prev)`` buffer per request. Each
V layer therefore registers a second, mamba-style cache alongside its paged KV
cache. The metadata exposes the state slots and the prefill query layout so
the layer can read and write that buffer at the right token — the last token
of each prefill chunk, or the current token during decode.
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

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class DragonDiffTPAMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    # Per-request slot into the (k_last, v_last) pool, shape [batch].
    state_indices_tensor: torch.Tensor

    # Prefill-only, shape [num_prefills]: is there a last-token K/V carried
    # over from an earlier chunk of the same request?
    has_initial_state: torch.Tensor | None = None

    # Prefill token start offsets, rebased to 0, shape [num_prefills + 1].
    query_start_loc_p: torch.Tensor | None = None

    # CPU mirrors, built once per step so the token-shift prefill path never
    # syncs per layer.
    has_initial_state_cpu: torch.Tensor | None = None
    query_start_loc_p_cpu: torch.Tensor | None = None
    state_indices_p_cpu: list[int] | None = None


class DragonDiffTPAMetadataBuilder(AttentionMetadataBuilder[DragonDiffTPAMetadata]):
    # The shift itself is a single cudagraph-friendly kernel for pure decode
    # batches; prefill gathers over variable-length chunks, so stay in the
    # decode-only capture regime.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
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
        state_indices_p_cpu = None
        if num_prefills > 0:
            has_initial_state = m.compute_num_computed_tokens()[num_decodes:] > 0
            qsl = m.query_start_loc[num_decodes:]
            query_start_loc_p = qsl - qsl[0]
            # Sync once here instead of once per V layer downstream.
            has_initial_state_cpu = has_initial_state.to("cpu")
            qsl_cpu = m.query_start_loc_cpu[num_decodes:]
            query_start_loc_p_cpu = qsl_cpu - qsl_cpu[0]
            state_indices_p_cpu = state_indices_tensor[num_decodes:].tolist()

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
            state_indices_p_cpu=state_indices_p_cpu,
        )
