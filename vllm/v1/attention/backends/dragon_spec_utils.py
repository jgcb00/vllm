# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared speculative-decode metadata for the Dragon mamba backends.

Slot protocol (same as upstream mamba2): every decode request carries
``1 + num_spec`` block-table columns. The state after the last committed
token lives in column ``max(num_accepted_prev - 1, 0)``; verify position t
reads the state written by position t-1 (column t-1, or the initial column
for t = 0) and writes column t. Acceptance next step selects the column —
no state copies, no rollback pass.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class DragonSpecMetadata:
    # (num_decodes, 1 + num_spec) int32 — per-request state slot columns.
    state_cols: torch.Tensor
    # (num_decodes,) — previous-step acceptance; initial-state column is
    # max(num_accepted - 1, 0).
    num_accepted_tokens: torch.Tensor
    # (num_decodes + 1,) — decode-token offsets in the flattened stream
    # (decodes come first, so this starts at 0).
    query_start_loc_d: torch.Tensor
    # Per-request decode query lengths (CPU, no sync — from the CPU qsl).
    decode_qlens_cpu: tuple[int, ...]
    max_qlen: int

    def initial_slots(self) -> torch.Tensor:
        """(num_decodes,) int32 — column holding each request's current state."""
        col = (self.num_accepted_tokens.long() - 1).clamp_(min=0)
        return self.state_cols.gather(1, col.unsqueeze(1)).squeeze(1)


def build_dragon_spec_metadata(
    m: CommonAttentionMetadata,
    block_table_tensor: torch.Tensor,
    num_decodes: int,
    num_spec: int,
    num_accepted_tokens: torch.Tensor | None,
) -> DragonSpecMetadata | None:
    if num_spec <= 0 or num_accepted_tokens is None or num_decodes == 0:
        return None
    qsl_cpu = m.query_start_loc_cpu[: num_decodes + 1]
    qlens = tuple(int(v) for v in torch.diff(qsl_cpu).tolist())
    return DragonSpecMetadata(
        state_cols=block_table_tensor[:num_decodes, : 1 + num_spec],
        num_accepted_tokens=num_accepted_tokens[:num_decodes],
        query_start_loc_d=m.query_start_loc[: num_decodes + 1],
        decode_qlens_cpu=qlens,
        max_qlen=max(qlens) if qlens else 1,
    )
