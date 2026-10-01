# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/jgcb00/mamba/blob/9a3daf5c488d9bf01d71441988de0293fd60ef0b/mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py
# (Mamba-3, Dao AI Lab / Goombalab, Apache-2.0); forward/inference parts only.

from typing import Optional, Tuple

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def rotary_qk_inference_kernel(
    OUT_Q,  # Pointers to matrices
    OUT_K,
    OUT_ANGLE_STATE,
    Q,
    K,
    ANGLE_STATE,
    ANGLE_PROJ,
    DT,
    BIAS_Q,
    BIAS_K,
    STATE_BATCH_INDICES,  # (batch,) int32 pool rows, or None
    angle_pool_rows,      # rows of the angle-state pool (bound check)
    nheads,
    headdim,
    stride_out_q,           # (batch, mimo_dim, nheads, headdim)
    stride_out_k,           # (batch, mimo_dim, nheads, headdim)
    stride_out_angle_state, # (batch, nheads, rotary_dim // 2)
    stride_q,               # (batch, mimo_dim, nheads, headdim)
    stride_k,               # (batch, mimo_dim, nheads, headdim)
    stride_angle_state,     # (batch, nheads, rotary_dim // 2)
    stride_angle_proj,      # (batch, nheads, rotary_dim // 2)
    stride_dt,              # (batch, nheads)
    stride_bias_q,          # (mimo_dim, nheads, headdim)
    stride_bias_k,          # (mimo_dim, nheads, headdim)
    # Meta-parameters
    ROTARY_DIM: tl.constexpr,
    CONJUGATE: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    MIMO_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr, # headdim, no chunking
    ROTATE_PAIRWISE: tl.constexpr, # If true, rotate every pair of dimensions together. Otherwise, rotate the first half and second half separately (like in the original RoPE paper)
):
    pid_nheads = tl.program_id(axis=0) # heads
    pid_batch = tl.program_id(axis=1)

    Q = Q + pid_batch * stride_q[0] + pid_nheads * stride_q[2]
    K = K + pid_batch * stride_k[0] + pid_nheads * stride_k[2]
    ANGLE_PROJ = ANGLE_PROJ + pid_batch * stride_angle_proj[0] + pid_nheads * stride_angle_proj[1]      # FIX: [1]
    DT = DT + pid_batch * stride_dt[0] + pid_nheads * stride_dt[1]

    OUT_Q = OUT_Q + pid_batch * stride_out_q[0] + pid_nheads * stride_out_q[2]
    OUT_K = OUT_K + pid_batch * stride_out_k[0] + pid_nheads * stride_out_k[2]

    # Angle state may live in a slot-indexed pool (paged inference): read and
    # write row state_batch_indices[b] in place instead of row b. Negative
    # rows (padding tokens) skip the token — q/k outputs zeroed, angle state
    # untouched (selective_state_update semantics). Implemented return-free:
    # loads are clamped to row 0 and every store is masked by ``valid``.
    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(STATE_BATCH_INDICES + pid_batch)
        # Out-of-range rows (CUDA-graph capture dummies) = padding too.
        valid = (state_batch_idx >= 0) & (state_batch_idx < angle_pool_rows)
        # int64: the pool may be a page-strided view, so int32
        # row * stride overflows once slot ids grow with server uptime.
        angle_batch = tl.minimum(tl.maximum(state_batch_idx, 0),
                                 angle_pool_rows - 1).to(tl.int64)
    else:
        valid = True
        angle_batch = pid_batch.to(tl.int64)
    ANGLE_STATE = ANGLE_STATE + angle_batch * stride_angle_state[0] + pid_nheads * stride_angle_state[1]  # FIX: [1]
    OUT_ANGLE_STATE = OUT_ANGLE_STATE + angle_batch * stride_out_angle_state[0] + pid_nheads * stride_out_angle_state[1]  # FIX: [1]

    rm = tl.arange(0, MIMO_DIM)
    rd = tl.arange(0, BLOCK_D)
    rd_half = tl.arange(0, BLOCK_D // 2)

    # Load angle and compute cos/sin (same for both q and k)
    ANGLE_STATE = ANGLE_STATE + rd_half * stride_angle_state[2]  # (rotary_dim // 2)
    mask_angle = rd_half < ROTARY_DIM // 2
    angle_state = tl.load(ANGLE_STATE, mask=mask_angle, other=0.0).to(tl.float32)

    ANGLE_PROJ = ANGLE_PROJ + rd_half * stride_angle_proj[2]     # (rotary_dim // 2)
    angle_proj = tl.load(ANGLE_PROJ, mask=mask_angle, other=0.0).to(tl.float32)

    dt = tl.load(DT, mask=True, other=0.0).to(tl.float32)

    # Match angle_dt: tanh(angle_proj) * dt * pi
    angle_proj = tl.sigmoid(2.0 * angle_proj) * 2.0 - 1.0  # tanh
    angle = angle_state + angle_proj * dt * 3.141592653589793  # (rotary_dim // 2)

    OUT_ANGLE_STATE = OUT_ANGLE_STATE + rd_half * stride_out_angle_state[2]
    # NOTE: the angle store is deferred to the END of the kernel. When the
    # update is in place (state_batch_indices), the compiler may rematerialize
    # the angle_state load inside the multi-warp q/k section instead of
    # keeping it in registers; storing here would race those reloads with
    # this program's own write (flaky double-stepped rotations).
    angle_out = angle

    angle = angle[None, :]  # (1, rotary_dim // 2) for mimo_dim broadcasting
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    if CONJUGATE:
        sin = -sin
    if HAS_STATE_BATCH_INDICES:
        # Padding tokens: zeroed cos/sin make every q/k output zero.
        cos = tl.where(valid, cos, 0.0)
        sin = tl.where(valid, sin, 0.0)

    # Process Q tensor
    Q = Q + (rm[:, None] * stride_q[1] + rd[None, :] * stride_q[3])
    OUT_Q = OUT_Q + (rm[:, None] * stride_out_q[1] + rd[None, :] * stride_out_q[3])
    mask = rd[None, :] < headdim
    q = tl.load(Q, mask=mask, other=0.0).to(tl.float32)  # (mimo_dim, headdim)

    # Add bias to Q if present
    if HAS_BIAS_Q:
        BIAS_Q = BIAS_Q + pid_nheads * stride_bias_q[1]                                                   
        BIAS_Q = BIAS_Q + (rm[:, None] * stride_bias_q[0] + rd[None, :] * stride_bias_q[2])
        bias_q = tl.load(BIAS_Q, mask=mask, other=0.0).to(tl.float32)
        q = q + bias_q

    if ROTATE_PAIRWISE:
        # Apply rotary to Q
        q0, q1 = tl.split(tl.reshape(q, [MIMO_DIM, BLOCK_D // 2, 2]))
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        qo = tl.reshape(tl.join(qo0, qo1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)
    else:
        # Apply rotary to Q
        q_reshaped = tl.reshape(q, [MIMO_DIM, 2, BLOCK_D // 2])
        q_permuted = tl.permute(q_reshaped, (0, 2, 1))  # (mimo_dim, block_d // 2, 2)
        q0, q1 = tl.split(q_permuted)
        qo0 = q0 * cos - q1 * sin
        qo1 = q0 * sin + q1 * cos
        q_joined = tl.join(qo0, qo1)
        q_final = tl.permute(q_joined, (0, 2, 1))  # (mimo_dim, 2, block_d // 2)
        qo = tl.reshape(q_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_Q, qo, mask=mask)

    # Process K tensor
    K = K + (rm[:, None] * stride_k[1] + rd[None, :] * stride_k[3])
    OUT_K = OUT_K + (rm[:, None] * stride_out_k[1] + rd[None, :] * stride_out_k[3])
    k = tl.load(K, mask=mask, other=0.0).to(tl.float32)

    # Add bias to K if present
    if HAS_BIAS_K:
        BIAS_K = BIAS_K + pid_nheads * stride_bias_k[1]                                                 
        BIAS_K = BIAS_K + (rm[:, None] * stride_bias_k[0] + rd[None, :] * stride_bias_k[2])
        bias_k = tl.load(BIAS_K, mask=mask, other=0.0).to(tl.float32)
        k = k + bias_k

    if ROTATE_PAIRWISE:
        # Apply rotary to K
        k0, k1 = tl.split(tl.reshape(k, [MIMO_DIM, BLOCK_D // 2, 2]))
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        ko = tl.reshape(tl.join(ko0, ko1), [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)
    else:
        # Apply rotary to K
        k_reshaped = tl.reshape(k, [MIMO_DIM, 2, BLOCK_D // 2])
        k_permuted = tl.permute(k_reshaped, (0, 2, 1))  # (mimo_dim, block_d // 2, 2)
        k0, k1 = tl.split(k_permuted)
        ko0 = k0 * cos - k1 * sin
        ko1 = k0 * sin + k1 * cos
        k_joined = tl.join(ko0, ko1)
        k_final = tl.permute(k_joined, (0, 2, 1))  # (mimo_dim, 2, block_d // 2)
        ko = tl.reshape(k_final, [MIMO_DIM, BLOCK_D])
        tl.store(OUT_K, ko, mask=mask)

    # Angle store last (see NOTE above): safe for in-place pool updates.
    tl.store(OUT_ANGLE_STATE, angle_out, mask=mask_angle & valid)

def apply_rotary_qk_inference_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    angle_state: torch.Tensor,
    angle_proj: torch.Tensor,
    dt: torch.Tensor,
    bias_q: Optional[torch.Tensor] = None,
    bias_k: Optional[torch.Tensor] = None,
    inplace=False,
    conjugate=False,
    rotate_pairwise=True,
    state_batch_indices: Optional[torch.Tensor] = None,
    # 2 warps ≈ 2.1x faster than 8 for the plain rotary (programs own only
    # MIMO_DIM x BLOCK_D elements; 8 warps was for a qk_sum variant).
    num_warps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding to both q and k tensors using the same angle.
    Also computes output angle state for next step.

    Arguments:
        q: (batch, mimo_dim, nheads, headdim)
        k: (batch, mimo_dim, nheads, headdim)
        angle_state: (batch, nheads, rotary_dim / 2), or a pool of
            (P, nheads, rotary_dim / 2) when state_batch_indices is given
        angle_proj: (batch, nheads, rotary_dim / 2)
        dt: (batch, nheads)
        bias_q: Optional (mimo_dim, nheads, headdim) - bias to add to q before rotary
        bias_k: Optional (mimo_dim, nheads, headdim) - bias to add to k before rotary
        state_batch_indices: Optional (batch,) int32 — row of the angle-state
            pool for each batch element. The angle state is read from and
            written back to row state_batch_indices[b] IN PLACE (mirrors
            selective_state_update). Negative rows skip the token: q/k
            outputs zeroed, state untouched.
    Returns:
        (q_out, k_out, angle_state_out): q_out and k_out are (batch, mimo_dim, nheads, headdim),
                               angle_state_out is (batch, nheads, rotary_dim / 2)
                               (= angle_state itself when state_batch_indices is given)
    """
    batch, mimo_dim, nheads, headdim = q.shape
    assert headdim % 2 == 0
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"

    rotary_dim = angle_state.shape[-1] * 2
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
        assert angle_state.shape[1:] == (nheads, rotary_dim // 2)
        assert angle_proj.shape == (batch, nheads, rotary_dim // 2)
    else:
        assert angle_state.shape == (batch, nheads, rotary_dim // 2)
        assert angle_state.shape == angle_proj.shape
    assert dt.shape == (batch, nheads)
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"

    if bias_q is not None:
        assert bias_q.shape == (mimo_dim, nheads, headdim), f"bias_q shape {bias_q.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}"
        bias_q = bias_q.contiguous()

    if bias_k is not None:
        assert bias_k.shape == (mimo_dim, nheads, headdim), f"bias_k shape {bias_k.shape} != (mimo_dim, nheads, headdim) {(mimo_dim, nheads, headdim)}"
        bias_k = bias_k.contiguous()

    output_q = torch.empty_like(q) if not inplace else q
    output_k = torch.empty_like(k) if not inplace else k
    # With state_batch_indices the angle pool is always updated in place.
    output_angle_state = (
        angle_state if (inplace or state_batch_indices is not None)
        else torch.empty_like(angle_state)
    )

    grid = lambda META: (nheads, batch)  # noqa
    with torch.cuda.device(q.device.index):
        torch.library.wrap_triton(rotary_qk_inference_kernel)[grid](
            output_q,  # data ptrs
            output_k,
            output_angle_state,
            q,
            k,
            angle_state,
            angle_proj,
            dt,
            bias_q,
            bias_k,
            state_batch_indices,
            angle_state.shape[0],
            nheads,
            headdim,
            output_q.stride(),  # output strides tuples
            output_k.stride(),
            output_angle_state.stride(),
            q.stride(),  # input strides tuples
            k.stride(),
            angle_state.stride(),
            angle_proj.stride(),
            dt.stride(),
            bias_q.stride() if bias_q is not None else (0, 0, 0),
            bias_k.stride() if bias_k is not None else (0, 0, 0),
            rotary_dim,
            conjugate,
            bias_q is not None,
            bias_k is not None,
            state_batch_indices is not None,
            MIMO_DIM=mimo_dim,
            BLOCK_D=triton.next_power_of_2(headdim),
            num_warps=num_warps,  # 8 was important when computing qk_sum
            ROTATE_PAIRWISE=rotate_pairwise,
        )
    return output_q, output_k, output_angle_state
