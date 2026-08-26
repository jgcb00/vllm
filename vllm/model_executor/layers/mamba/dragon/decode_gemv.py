# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token GEMV for Dragon's decode projections.

cuBLAS's batch-1 kernels sit on a ~4.2 us floor whatever the matrix size,
so the many small projections of a decode step (latent MoE, in_proj_dyn,
TPA factors) run at well under 1 TB/s. This streaming kernel — each program
owns BN output rows and walks K in BK chunks — is 2.5-3 us on those shapes
and also ahead of cuBLAS on the large ones (measured under CUDA graphs on
GH200). For M > 1 the per-row loop re-reads the weight tile, so cuBLAS
remains the better choice and ``decode_gemv`` falls back to F.linear.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gemv_m1_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros([BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        x = tl.load(x_ptr + offs_k)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def decode_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``x: (M, K)``, ``weight: (N, K)``; triton at M == 1."""
    if x.shape[0] != 1 or not x.is_cuda or x.dtype != weight.dtype:
        return F.linear(x, weight)
    N, K = weight.shape
    if K % 128 != 0:
        return F.linear(x, weight)
    y = torch.empty(1, N, dtype=x.dtype, device=x.device)
    bk = 256 if N >= 4096 else (512 if K % 512 == 0 else 128)
    _gemv_m1_kernel[(triton.cdiv(N, 8),)](
        x, weight, y, N, K, weight.stride(0),
        BN=8, BK=bk, num_warps=2 if N >= 4096 else (4 if K <= 1536 else 8),
    )
    return y


def _dragon_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return decode_gemv(x, weight)


def _dragon_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)


# Opaque to torch.compile so the M == 1 dispatch is re-evaluated per cudagraph
# capture instead of being specialized by the first (large) compile.
direct_register_custom_op(
    op_name="dragon_linear",
    op_func=_dragon_linear,
    mutates_args=[],
    fake_impl=_dragon_linear_fake,
)


def dragon_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.dragon_linear(x, weight)
