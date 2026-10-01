# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-built CUDA Mamba-3 MIMO decode step (see csrc/mamba3_step.cu).

Persistent CTAs (one head each, several tokens) stream the SSM state through
a 2-stage cp.async pipeline; the rank-4 state update and the C-contraction
run on tensor cores. Numerically equivalent to the CuteDSL step within bf16
rounding (the state is bf16-rounded before the C dot). GH200, B=64: 40 us
per layer vs 57 us (CuteDSL); B=16: 11.4 vs 15.1; B=1: 3.8 vs 6.1.

An fp32 SSM pool runs the update and the C dot as fp32 FMAs on the unrounded
state (matches the CuteDSL fp32 step to ~1e-4 relative). GH200, B=64: 73 us
vs 80 us (CuteDSL); B=16: 17.5 vs 18.5.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.olala.fallback import warn_slow_path

logger = init_logger(__name__)

_ENV = "OLALA_MAMBA3_STEP"


@lru_cache(maxsize=1)
def cuda_step_enabled() -> bool:
    """OLALA_MAMBA3_STEP=cuda (default), on Hopper only (sm_90 SASS), and only
    if the JIT build succeeds; otherwise the CuteDSL step is used."""
    if os.environ.get(_ENV, "cuda") != "cuda":
        logger.info("Olala: CUDA Mamba-3 step disabled by %s.", _ENV)
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        warn_slow_path(
            "CUDA Mamba-3 decode step disabled",
            f"the kernel is built for sm_90 (Hopper) only; this GPU is sm_{cap}",
            "CuteDSL decode step: ~1.4x slower Mamba-3 decode step (-10 to -15% decode throughput)",
            "run on H100/H200/GH200 (nothing to fix elsewhere; output is correct)",
        )
        return False
    try:
        _load()
    except Exception as e:  # noqa: BLE001 - any build/load failure falls back
        warn_slow_path(
            "CUDA Mamba-3 decode step disabled",
            f"JIT build of mamba3_step.cu failed: {e}",
            "CuteDSL decode step: ~1.4x slower Mamba-3 decode step (-10 to -15% decode throughput)",
            "the JIT needs an nvcc of torch's CUDA major (torch.version.cuda); "
            "set OLALA_JIT_CUDA_HOME to such a toolkit",
        )
        return False
    return True


@lru_cache(maxsize=1)
def _load():
    from vllm.model_executor.layers.mamba.olala.jit import load_extension

    return load_extension(
        "olala_mamba3_step",
        "mamba3_step.cu",
        ["-O3", "-std=c++17", "-gencode=arch=compute_90,code=sm_90"],
    )


def cuda_step_supported(ssm_pool, k_pool, v_pool, angle_pool, B, C, angle, num_heads, head_dim, d_state, rank, num_angles) -> bool:
    return (
        num_heads == 48 and head_dim == 64 and d_state == 128 and rank == 4 and num_angles == 32
        and ssm_pool.dtype in (torch.float32, torch.bfloat16) and k_pool.dtype == torch.bfloat16 and v_pool.dtype == torch.bfloat16
        and angle_pool.dtype == torch.float32
        and B.dim() == 3 and B.stride(1) == d_state and B.stride(2) == 1 and (B.stride(0) * 2) % 16 == 0
        and C.stride() == B.stride() and angle.stride(1) == 1 and (angle.stride(0) * 2) % 16 == 0
    )


_F32_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _fp32(w: torch.Tensor) -> torch.Tensor:
    """fp32 copy of a small constant parameter, cached by storage (D may be a bf16 parameter)."""
    if w.dtype == torch.float32:
        return w
    key = w.data_ptr()
    e = _F32_CACHE.get(key)
    if e is None:
        e = (w, w.detach().float().contiguous())
        _F32_CACHE[key] = e
    return e[1]


def refresh_f32_cache() -> None:
    """Recompute the cached fp32 copies after an in-place weight reload
    (their addresses are baked into captured CUDA graphs)."""
    with torch.inference_mode():
        for src, dst in _F32_CACHE.values():
            dst.copy_(src.detach().float())


def _as_f32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float32 else t.float()


def mamba3_step_cuda(ssm_pool, k_pool, v_pool, angle_pool, A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                     bias_q, bias_k, angle_proj, slots, y, ctas_per_sm: int = 4) -> None:
    """B, C: (N, R, S) views (any batch stride); angle_proj: (N, 32) view; everything else contiguous."""
    Dp = _fp32(Dp)
    A, dt, trap = _as_f32(A), _as_f32(dt), _as_f32(trap)
    _load().step(ssm_pool, k_pool, v_pool, angle_pool, A, B, C, Dp, x, dt, trap, xproj, zproj, outproj, z,
                 bias_q, bias_k, angle_proj, slots, y, ctas_per_sm)
