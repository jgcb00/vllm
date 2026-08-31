# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for Olala's Mamba3 MIMO mixer.

The mixer keeps four per-request temporal states (angle, ssm, k, v) and has no
causal conv1d, so this metadata is much smaller than GDN's: a prefill/decode
split, the state slot lookup, and the prefill query layout.

Speculative decoding uses the column-slot protocol (see olala_spec_utils):
verify batches chain the dual-slot step kernel, one launch per draft
position, leaving per-position states that next step's acceptance selects.
"""

import os
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.olala_spec_utils import (
    OlalaSpecMetadata,
    build_olala_spec_metadata,
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

    @classmethod
    def is_ssm(cls) -> bool:
        return True


def window_decode_reset_enabled() -> bool:
    """Reset the SSM state mid-GENERATION at artificial_seq_len boundaries?

    Off by default: it costs full cudagraph capture, measured at 4.1x decode
    throughput, to correct generations that run past a window boundary. Prompts
    that cross one are handled in prefill regardless -- that path is exact and
    costs nothing. Turn it on with OLALA_WINDOW_DECODE_RESET=1 when generating
    tens of thousands of tokens per request.
    """
    return os.environ.get("OLALA_WINDOW_DECODE_RESET", "0") not in ("0", "", "false")


def _build_window_segments(qsl_cpu, abs_starts_cpu, window, device):
    """Split each prefill request at ABSOLUTE multiples of ``window``.

    Megatron trains with the SSM state cleared every ``artificial_seq_len``
    tokens -- megatron/core/dragon/dragon_mamba3.py (still Dragon-named on the
    training side) merges those boundaries into the cu_seqlens it hands
    mamba3_mimo -- so inference has to reproduce them or it runs a regime the
    model never saw. The boundary is on the ABSOLUTE position: under
    chunked prefill a chunk-local offset points at the wrong tokens.

    Returns ``(cu_seqlens, owner, is_first, last)`` where ``owner[j]`` is the
    request sub-segment j belongs to, ``is_first[j]`` marks the sub-segment
    that inherits the request's carried state, and ``last[p]`` is the
    sub-segment whose final state request p continues from. Returns None when
    no request crosses a boundary, which is the common case.
    """
    qsl = qsl_cpu.tolist()
    starts = abs_starts_cpu.tolist()
    bounds = [qsl[0]]
    owner: list[int] = []
    seed: list[bool] = []
    last: list[int] = []
    any_reset = False
    for p in range(len(qsl) - 1):
        s, e, a = qsl[p], qsl[p + 1], starts[p]
        # A chunk that BEGINS exactly on a boundary must not inherit the state
        # carried from the previous chunk: training cleared it at that very
        # position. Such a chunk has no INTERIOR cut, so it would otherwise
        # look like an ordinary continuation.
        resets_at_start = a > 0 and a % window == 0
        any_reset |= resets_at_start
        step = (-a) % window or window          # first boundary strictly after a
        for k, end in enumerate(list(range(s + step, e, window)) + [e]):
            bounds.append(end)
            owner.append(p)
            seed.append(k == 0 and not resets_at_start)
        last.append(len(owner) - 1)
    if len(owner) == len(qsl) - 1 and not any_reset:
        return None                              # no cut, nothing to re-seed
    return (
        torch.tensor(bounds, dtype=torch.int32, device=device),
        torch.tensor(owner, dtype=torch.long, device=device),
        torch.tensor(seed, dtype=torch.bool, device=device),
        torch.tensor(last, dtype=torch.long, device=device),
    )


@dataclass
class Mamba3AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_actual_tokens: int

    # Per-request state slot into the mamba state pool, shape [batch]. Valid
    # for prefills and decodes alike; cudagraph padding lanes carry
    # NULL_BLOCK_ID, courtesy of the model runner.
    state_indices_tensor: torch.Tensor

    # Prefill-only, shape [num_prefills]: does the request carry a state over
    # from an earlier chunk?
    has_initial_state: torch.Tensor | None = None

    # Prefill token start offsets, rebased to 0, shape [num_prefills + 1].
    query_start_loc_p: torch.Tensor | None = None

    # CPU mirrors, computed once here so per-layer code never syncs on the GPU
    # tensors. The model runs 29 Mamba3 layers per step, and a per-layer
    # .item()/.tolist() costs a full pipeline stall each.
    has_initial_state_cpu: torch.Tensor | None = None
    has_initial_any: bool = False
    query_start_loc_p_cpu: torch.Tensor | None = None

    # artificial_seq_len window resets, both None unless the config sets a
    # window AND this batch actually crosses one.
    # Prefill: (cu_seqlens, owner, is_first, last) from _build_window_segments.
    win_segments: tuple[torch.Tensor, ...] | None = None
    # Decode: state slots sitting exactly on a boundary, so they must start
    # from a zero state. A slot list rather than a full-batch mask because a
    # reset fires once per `window` tokens per sequence -- masking every row
    # every step would roughly double the decode state traffic.
    decode_reset_slots: torch.Tensor | None = None

    # Speculative decoding (verify batches, column-slot protocol). None when
    # spec decode is off or the batch has no decode rows.
    spec: OlalaSpecMetadata | None = None


class Mamba3AttentionMetadataBuilder(AttentionMetadataBuilder[Mamba3AttentionMetadata]):
    # The TileLang varlen prefill kernel is not cudagraph-capturable, so only
    # pure single-token decode batches can be captured.
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
        spec_config = vllm_config.speculative_config
        self.num_spec = (
            spec_config.num_speculative_tokens if spec_config is not None else 0
        )
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        # The captured decode graph bakes the non-spec slot addressing
        # (state in column 0); under spec decode the state column is dynamic,
        # so replaying that graph would read the wrong slot.
        if vllm_config.speculative_config is not None:
            return AttentionCGSupport.NEVER
        # The DECODE half of the artificial_seq_len reset runs in Python, which
        # does not execute on cudagraph replay -- so enabling it means refusing
        # full-graph capture. MEASURED, 16 seqs x 64 tokens on H100: 1165.7
        # tok/s with FULL graphs vs 281.9 tok/s once downgraded to PIECEWISE,
        # a 4.1x decode regression on EVERY request. The reset it buys only
        # matters when a GENERATION crosses a multiple of artificial_seq_len;
        # a prompt that crosses one is handled in prefill, which is exact and
        # free either way. So it is off by default and opt-in.
        if window_decode_reset_enabled():
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def build(
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
        state_indices_tensor = block_table_tensor[:, 0]

        spec_active = self.num_spec > 0 and num_accepted_tokens is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                m,
                decode_threshold=(self.reorder_batch_threshold if spec_active else 1),
                treat_short_extends_as_decodes=False,
            )
            if spec_active
            else split_decodes_and_prefills(m, decode_threshold=1)
        )

        spec = (
            build_olala_spec_metadata(
                m,
                block_table_tensor,
                num_decodes,
                self.num_spec,
                num_accepted_tokens,
            )
            if spec_active
            else None
        )

        has_initial_state = None
        query_start_loc_p = None
        has_initial_state_cpu = None
        has_initial_any = False
        query_start_loc_p_cpu = None
        if num_prefills > 0:
            # Prefills sit after decodes by the common batch reorder.
            has_initial_state = m.compute_num_computed_tokens()[num_decodes:] > 0
            qsl = m.query_start_loc[num_decodes:]
            query_start_loc_p = qsl - qsl[0]
            # One sync here, on prefill steps only, instead of one per layer.
            has_initial_state_cpu = has_initial_state.to("cpu")
            has_initial_any = bool(has_initial_state_cpu.any())
            qsl_cpu = m.query_start_loc_cpu[num_decodes:]
            query_start_loc_p_cpu = qsl_cpu - qsl_cpu[0]

        # artificial_seq_len window resets. num_computed_tokens_cpu is derived
        # from CPU tensors, so none of this costs a device sync.
        window = int(
            getattr(self.vllm_config.model_config.hf_config, "artificial_seq_len", 0)
            or 0
        )
        win_segments = None
        decode_reset_slots = None
        if window > 0:
            if num_decodes > 0 and window_decode_reset_enabled():
                pos = m.num_computed_tokens_cpu[:num_decodes]
                hit = ((pos > 0) & (pos % window == 0)).nonzero().flatten()
                if hit.numel():
                    decode_reset_slots = state_indices_tensor[:num_decodes][
                        hit.to(state_indices_tensor.device)
                    ].long()
            if num_prefills > 0:
                win_segments = _build_window_segments(
                    query_start_loc_p_cpu,
                    m.num_computed_tokens_cpu[num_decodes:],
                    window,
                    query_start_loc_p.device,
                )

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
            win_segments=win_segments,
            decode_reset_slots=decode_reset_slots,
            spec=spec,
        )
