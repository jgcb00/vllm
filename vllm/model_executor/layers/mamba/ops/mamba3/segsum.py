# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/jgcb00/mamba/blob/9a3daf5c488d9bf01d71441988de0293fd60ef0b/mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py
# (Mamba-3, Dao AI Lab / Goombalab, Apache-2.0); forward/inference parts only.

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [2, 3]
        for w in [4, 8]
    ],
    key=["CHUNK_SIZE"],
)
@triton.jit
def dacs_segsum_kernel_varlen(
    da_ptr,
    da_cs_ptr,
    da_cs_rev_ptr,
    segsum_ptr,
    cu_seqlens_ptr,
    state_seq_mapping_ptr,
    state_chunk_in_seq_ptr,
    stride_da_batch, stride_da_head, stride_da_seq,
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seq,
    stride_da_cs_rev_batch, stride_da_cs_rev_head, stride_da_cs_rev_seq,
    stride_segsum_batch, stride_segsum_head, stride_segsum_chunk,
    stride_segsum_row, stride_segsum_col,
    stride_cu_seqlen, stride_state_seq_mapping, stride_state_chunk_in_seq,
    SEQLEN,
    NUM_SEQUENCES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Varlen version of dacs_segsum_kernel.

    Computes da_cs (forward prefix-sum of da, clipped to ≤ 0), da_cs_rev
    (exclusive reverse prefix-sum, clipped to ≤ 0), and the lower-triangular
    segsum matrix for each chunk, respecting sequence boundaries so that sums
    do not cross from one sequence into another.

    pid_chunk indexes into the global chunk array (length nchunks_global).
    state_seq_mapping and state_chunk_in_seq decode which sequence and local
    chunk position the program is responsible for.  Inactive (padding) slots
    produce an all-False mask and write nothing.

    Grid: (B, H, nchunks_global).
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    curr_seq_ind = tl.load(state_seq_mapping_ptr + pid_chunk * stride_state_seq_mapping)
    local_chunk_ind = tl.load(state_chunk_in_seq_ptr + pid_chunk * stride_state_chunk_in_seq)
    seq_start = tl.load(cu_seqlens_ptr + curr_seq_ind * stride_cu_seqlen)
    seq_end = tl.load(cu_seqlens_ptr + (curr_seq_ind + 1) * stride_cu_seqlen)
    offs = tl.arange(0, CHUNK_SIZE)
    offs_seq = seq_start + local_chunk_ind * CHUNK_SIZE + offs
    mask = (offs_seq < seq_end) & (offs_seq < SEQLEN)

    base_da = pid_batch * stride_da_batch + pid_head * stride_da_head
    da_chunk = tl.load(da_ptr + base_da + offs_seq * stride_da_seq, mask=mask, other=0.0)

    da_cs = tl.cumsum(da_chunk, axis=0)
    da_cs = tl.minimum(da_cs, 0.0)

    da_cs_rev_inclusive = tl.cumsum(da_chunk, axis=0, reverse=True)
    da_cs_rev = da_cs_rev_inclusive - da_chunk
    da_cs_rev = tl.minimum(da_cs_rev, 0.0)

    base_da_cs = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head
    base_da_cs_rev = pid_batch * stride_da_cs_rev_batch + pid_head * stride_da_cs_rev_head
    tl.store(da_cs_ptr + base_da_cs + offs_seq * stride_da_cs_seq, da_cs, mask=mask)
    tl.store(da_cs_rev_ptr + base_da_cs_rev + offs_seq * stride_da_cs_rev_seq, da_cs_rev, mask=mask)

    offs_i = offs[:, None]
    offs_j = offs[None, :]
    segsum = tl.where(offs_i > offs_j, da_chunk[:, None], 0.0)
    segsum = tl.cumsum(segsum, axis=0)
    segsum = tl.minimum(segsum, 0.0)

    base_segsum = (pid_batch * stride_segsum_batch +
                   pid_head * stride_segsum_head +
                   pid_chunk * stride_segsum_chunk)
    tl.store(segsum_ptr + base_segsum + offs_i * stride_segsum_row + offs_j * stride_segsum_col, segsum)


def compute_dacs_segsum_triton_varlen(
    da: torch.Tensor,              # [B, H, S]
    chunk_size: int,
    cu_seqlens: torch.Tensor = None,   # [NS+1], int32
):
    """Compute da_cs, da_cs_rev, and segsum for variable-length packed sequences.

    Parameters
    ----------
    da           : decay increments, shape [B, H, S].
    chunk_size   : tokens per chunk (must be a power of two).
    cu_seqlens   : cumulative sequence lengths, shape [NS+1], starting at 0.

    Returns
    -------
    da_cs        : forward inclusive prefix-sum of da, clipped to ≤ 0. [B, H, S]
    da_cs_rev    : exclusive reverse prefix-sum of da, clipped to ≤ 0. [B, H, S]
    segsum       : lower-triangular intra-chunk segsum. [B, H, nchunks_global, C, C]
                   nchunks_global = (S // chunk_size) + num_sequences.
    """
    assert cu_seqlens is not None, "vLLM uses the varlen path only"

    B, H, S = da.shape
    assert cu_seqlens.ndim == 1, f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}"
    num_sequences = max(int(cu_seqlens.numel()) - 1, 0)
    assert num_sequences > 0

    # Global chunk count: each sequence contributes ceil(len/chunk_size) chunks,
    # which equals (len // chunk_size) + 1 under the "always +1" convention used
    # throughout this module.  Summing over all sequences gives:
    #     nchunks = (S // chunk_size) + num_sequences
    nchunks = (S // chunk_size) + num_sequences

    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.zeros(B, H, nchunks, chunk_size, chunk_size, device=da.device, dtype=da.dtype)

    # Build mapping tensors: both have length nchunks_global.
    # Inactive (padding) slots are given a sentinel local-chunk index that
    # places chunk_start >= seq_end, making the kernel mask all-False.
    #
    # Fully vectorized on device — NO GPU<->CPU sync. (The previous
    # per-sequence cu_seqlens[i].item() loop issued 2*num_sequences+1 syncs
    # and 2*num_sequences slice-fill kernels per call; per layer per packed
    # prefill this dominated TTFT under serving.)
    #
    # Sequence i owns the global chunk slots
    #     [cu[i]//C + i,  cu[i]//C + i + len_i//C + 1)
    # These starts are strictly increasing (floor(a+b) >= floor(a)+floor(b)),
    # so the owner of slot g is searchsorted(starts, g, right) - 1, and
    # slots at/after that owner's end are inactive padding. The end of the
    # last range is always <= nchunks by the same floor inequality, so the
    # old per-sequence overflow assert could never fire and is dropped.
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_idx = torch.arange(
        num_sequences, dtype=cu_seqlens.dtype, device=da.device
    )
    range_starts = cu_seqlens[:-1] // chunk_size + seq_idx
    range_ends = range_starts + seq_lens // chunk_size + 1
    g = torch.arange(nchunks, dtype=cu_seqlens.dtype, device=da.device)
    owner = torch.searchsorted(range_starts, g, right=True) - 1
    active = g < range_ends[owner]
    # Sentinel for inactive slots: ceil(len_0 / C), matching the original.
    default_inactive_local_chunk = (
        (seq_lens[0] + chunk_size - 1) // chunk_size
    )
    state_seq_mapping = torch.where(
        active, owner, torch.zeros_like(owner)
    ).to(torch.int32)
    state_chunk_in_seq = torch.where(
        active, g - range_starts[owner], default_inactive_local_chunk
    ).to(torch.int32)

    grid = (B, H, nchunks)
    dacs_segsum_kernel_varlen[grid](
        da, da_cs, da_cs_rev, segsum, cu_seqlens, state_seq_mapping, state_chunk_in_seq,
        da.stride(0), da.stride(1), da.stride(2),
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        da_cs_rev.stride(0), da_cs_rev.stride(1), da_cs_rev.stride(2),
        segsum.stride(0), segsum.stride(1), segsum.stride(2),
        segsum.stride(3), segsum.stride(4),
        cu_seqlens.stride(0), state_seq_mapping.stride(0), state_chunk_in_seq.stride(0),
        S,
        num_sequences,
        chunk_size,
    )

    return da_cs, da_cs_rev, segsum
