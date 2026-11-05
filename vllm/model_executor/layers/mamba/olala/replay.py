# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM bookkeeping outside the decode step (see csrc/mamba3_replay_step.cu).

A prefill leaves the full state in the anchor (``ssm`` pool). Before the slot
decodes, its ring must be marked empty and, with an fp8 read copy, the copy
refreshed from the anchor. The first window's flush target is staggered by
slot so that a batch of requests admitted together does not flush on the same
step (the window length only affects when the anchor is rewritten, not the
output).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _replay_reset_kernel(
    anchor_ptr, rd_ptr, rds_ptr, cnt_ptr, slots_ptr,
    anc_sst, rd_sst, rds_sst, cnt_sst,
    H, L,
    D: tl.constexpr,
    S: tl.constexpr,
    READ_FP8: tl.constexpr,
):
    i = tl.program_id(0)  # prefill row
    h = tl.program_id(1)
    slot = tl.load(slots_ptr + i).to(tl.int64)
    if slot < 0:
        return
    if READ_FP8:
        rows = tl.arange(0, D)
        cols = tl.arange(0, S)
        off = h * D * S + rows[:, None] * S + cols[None, :]
        v = tl.load(anchor_ptr + slot * anc_sst + off).to(tl.float32)
        amax = tl.max(tl.abs(v), axis=1)
        sc = tl.where(amax > 0.0, amax / 448.0, 1.0)
        q = tl.clamp(v / sc[:, None], -448.0, 448.0)
        tl.store(rd_ptr + slot * rd_sst + off, q.to(rd_ptr.dtype.element_ty))
        tl.store(rds_ptr + slot * rds_sst + h * D + rows, sc)
    tl.store(cnt_ptr + slot * cnt_sst + h * 2, 0)
    tl.store(cnt_ptr + slot * cnt_sst + h * 2 + 1, 1 + (slot % L).to(tl.int32))


def replay_reset(pools: tuple[torch.Tensor, ...], slots: torch.Tensor) -> None:
    """Mark ``slots`` (just prefilled) as checkpoint-only and refresh their read copy."""
    n = slots.numel()
    if n == 0:
        return
    anchor, cnt = pools[1], pools[8]
    read_fp8 = len(pools) > 9
    rd, rds = (pools[9], pools[10]) if read_fp8 else (anchor, anchor)
    H, D, S = anchor.shape[1:]
    L = pools[4].shape[2]
    _replay_reset_kernel[(n, H)](
        anchor, rd, rds, cnt, slots,
        anchor.stride(0), rd.stride(0), rds.stride(0), cnt.stride(0),
        H, L, D=D, S=S, READ_FP8=read_fp8, num_warps=4,
    )
