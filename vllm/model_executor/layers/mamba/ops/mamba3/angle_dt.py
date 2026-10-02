# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/jgcb00/mamba/blob/9a3daf5c488d9bf01d71441988de0293fd60ef0b/mamba_ssm/ops/triton/mamba3/angle_dt.py
# (Mamba-3, Dao AI Lab / Goombalab, Apache-2.0); forward/inference parts only.

from typing import Optional, Tuple

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton



# Accurate tanh, the decode step's formula: the phase is a running sum, so tanh.approx's ~5e-4 relative
# error accumulates (~6e-4 rad after 32k tokens) and prefill would disagree with decode.
@triton.jit
def _tanh(x):
    return tl.sigmoid(2.0 * x) * 2.0 - 1.0


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_D", "HAS_INIT_STATE", "RETURN_OUTPUT_STATE", "IS_VARLEN"],
)
@triton.jit
def angle_dt_fwd_kernel(
    # Outputs
    OUT, OUTPUT_STATE,
    # Inputs
    ANGLE, DT, INIT_STATE, CU_SEQLENS,
    # Strides for OUT (batch, seqlen, nheads, dim)
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    # Strides for OUTPUT_STATE (num_sequences, nheads, dim)
    stride_output_state_seq, stride_output_state_head, stride_output_state_dim,
    # Strides for ANGLE (batch, seqlen, nheads, dim)
    stride_angle_batch, stride_angle_seq, stride_angle_head, stride_angle_dim,
    # Strides for DT (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seq,
    # Strides for INIT_STATE (num_sequences, nheads, dim)
    stride_init_seq, stride_init_head, stride_init_dim,
    # Stride for CU_SEQLENS
    stride_cu_seqlen,
    # Dimensions
    seqlen, dim,
    # Meta-parameters
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_INIT_STATE: tl.constexpr,
    RETURN_OUTPUT_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Handle varlen mode
    if IS_VARLEN:
        pid_seq = tl.program_id(2)
        seq_idx = pid_seq
        cu_seqlen_start = tl.load(CU_SEQLENS + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_end = tl.load(CU_SEQLENS + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seq_len = cu_seqlen_end - cu_seqlen_start
        seq_offset = cu_seqlen_start
    else:
        seq_idx = pid_b
        seq_len = seqlen
        seq_offset = 0

    nchunks = tl.cdiv(seq_len, CHUNK_SIZE)

    # Offset base pointers by batch and head
    ANGLE += pid_b * stride_angle_batch + pid_h * stride_angle_head + seq_offset * stride_angle_seq
    DT += pid_b * stride_dt_batch + pid_h * stride_dt_head + seq_offset * stride_dt_seq
    OUT += pid_b * stride_out_batch + pid_h * stride_out_head + seq_offset * stride_out_seq

    dim_range = tl.arange(0, BLOCK_D)
    dim_mask = dim_range < dim

    # Initialize state from init_state or zeros
    if HAS_INIT_STATE:
        init_ptrs = INIT_STATE + seq_idx * stride_init_seq + pid_h * stride_init_head + dim_range * stride_init_dim
        state = tl.load(init_ptrs, mask=dim_mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    PI = 3.141592653589793
    TWO_PI = 2 * PI

    for chunk_idx in range(nchunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        seq_range = tl.arange(0, CHUNK_SIZE)
        seq_mask = (chunk_start + seq_range) < seq_len

        # Load angle (CHUNK_SIZE, BLOCK_D)
        angle_ptrs = ANGLE + (chunk_start + seq_range[:, None]) * stride_angle_seq + dim_range[None, :] * stride_angle_dim
        angle_vals = tl.load(angle_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
        angle_vals = _tanh(angle_vals) * PI

        # Load dt (CHUNK_SIZE,)
        dt_ptrs = DT + (chunk_start + seq_range) * stride_dt_seq
        dt_vals = tl.load(dt_ptrs, mask=seq_mask, other=0.0).to(tl.float32)

        # Compute vals = angle * dt
        vals = angle_vals * dt_vals[:, None]

        # Cumsum within chunk + add state from previous chunks
        chunk_cumsum = tl.cumsum(vals, axis=0)
        out_vals = chunk_cumsum + state[None, :]

        # Apply mod 2*pi for rotary angle normalization
        out_vals = out_vals - TWO_PI * tl.floor(out_vals / TWO_PI)

        # Store output
        out_ptrs = OUT + (chunk_start + seq_range[:, None]) * stride_out_seq + dim_range[None, :] * stride_out_dim
        tl.store(out_ptrs, out_vals, mask=seq_mask[:, None] & dim_mask[None, :])

        # Update state: add chunk sum and apply mod 2*pi
        chunk_sum = tl.sum(vals, axis=0)
        state = state + chunk_sum
        state = state - TWO_PI * tl.floor(state / TWO_PI)

    # Store final state if requested
    if RETURN_OUTPUT_STATE:
        output_state_ptrs = OUTPUT_STATE + seq_idx * stride_output_state_seq + pid_h * stride_output_state_head + dim_range * stride_output_state_dim
        tl.store(output_state_ptrs, state, mask=dim_mask)


def angle_dt_fwd(
    angle: Tensor,
    dt: Tensor,
    init_state: Optional[Tensor] = None,
    chunk_size: int = 64,
    return_output_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Forward pass for angle * dt cumsum.

    Args:
        angle: Angle tensor             (batch, seqlen, nheads, dim)
        dt: Time delta tensor           (batch, nheads, seqlen)
        init_state: Initial state       (num_sequences, nheads, dim) or None
        chunk_size: Chunk size for chunked computation
        return_output_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths (num_sequences + 1,) for varlen mode

    Returns:
        If return_output_state=False:
            out: Cumulative output      (batch, seqlen, nheads, dim)
        If return_output_state=True:
            Tuple of:
                out: Cumulative output      (batch, seqlen, nheads, dim)
                output_state: Final state   (num_sequences, nheads, dim)
    """
    batch, seqlen, nheads, dim = angle.shape
    is_varlen = cu_seqlens is not None
    
    # Determine number of sequences
    if is_varlen:
        assert batch == 1, "Varlen mode requires batch=1"
        num_sequences = cu_seqlens.shape[0] - 1
    else:
        num_sequences = batch
    
    assert dt.shape == (batch, nheads, seqlen), f"dt shape mismatch: {dt.shape}"
    if init_state is not None:
        assert init_state.shape == (num_sequences, nheads, dim), f"init_state shape mismatch: {init_state.shape}"

    out = torch.empty_like(angle)
    BLOCK_D = triton.next_power_of_2(dim)

    # Handle None init_state for kernel
    HAS_INIT_STATE = init_state is not None
    if not HAS_INIT_STATE:
        init_state = angle  # dummy, won't be accessed
        stride_init = (0, 0, 0)
    else:
        stride_init = init_state.stride()

    # Handle output_state
    if return_output_state:
        output_state = torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=angle.dtype)
        stride_output_state = output_state.stride()
    else:
        output_state = out  # dummy, won't be accessed
        stride_output_state = (0, 0, 0)

    # Handle cu_seqlens
    if cu_seqlens is not None:
        stride_cu_seqlen = cu_seqlens.stride(0)
    else:
        cu_seqlens = angle  # dummy, won't be accessed
        stride_cu_seqlen = 0

    # Grid setup
    if is_varlen:
        grid = (nheads, batch, num_sequences)
    else:
        grid = (nheads, batch)

    angle_dt_fwd_kernel[grid](
        out, output_state,
        angle, dt, init_state, cu_seqlens,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stride_output_state[0], stride_output_state[1], stride_output_state[2],
        angle.stride(0), angle.stride(1), angle.stride(2), angle.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2),
        stride_init[0], stride_init[1], stride_init[2],
        stride_cu_seqlen,
        seqlen, dim,
        CHUNK_SIZE=chunk_size,
        BLOCK_D=BLOCK_D,
        HAS_INIT_STATE=HAS_INIT_STATE,
        RETURN_OUTPUT_STATE=return_output_state,
        IS_VARLEN=is_varlen,
    )

    if return_output_state:
        return out, output_state
    return out


# -----------------------------------------------------------------------------
# Backward kernel
# -----------------------------------------------------------------------------
