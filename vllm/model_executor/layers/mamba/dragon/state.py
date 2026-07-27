# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request recurrent state layouts for the Dragon mixers.

Dragon carries two kinds of paged recurrent state, both allocated through
``MambaSpec`` so vLLM's state manager owns the per-request slots:

* Mamba3 MIMO (``M`` layers) — four temporal tensors.
* Differential-TPA token shift (``V`` layers) — a one-token ``(k, v)``
  buffer holding the previous token's raw (pre-shift) key/value.

These live here rather than in the shared ``mamba_utils`` calculators so the
fork touches one fewer upstream file.
"""

import torch

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)


def mamba3_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
    mamba_ssm_cache_dtype: MambaDType,
) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
    """Dtypes of ``(angle, ssm, k, v)``.

    The angle state accumulates rotary phase across the whole sequence, so it
    is always fp32 — a bf16 accumulator visibly drifts over long contexts.
    The SSM state honors ``--mamba-ssm-cache-dtype``; under the default
    ``auto`` it follows the model dtype (bf16), which measured
    quality-neutral on gsm8k/humaneval/RULER and worth +16% decode throughput
    at high concurrency versus fp32 storage. Kernel compute stays fp32 either
    way. Pass ``--mamba-ssm-cache-dtype float32`` to store fp32 instead.
    """
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    if mamba_ssm_cache_dtype == "auto":
        ssm_dtype = state_dtype
    else:
        ssm_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
    return (torch.float32, ssm_dtype, state_dtype, state_dtype)


def mamba3_state_shape(
    tp_world_size: int,
    num_heads: int,
    head_dim: int,
    d_state: int,
    mimo_dim: int,
    num_rope_angles: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    """Shapes of ``(angle, ssm, k, v)`` per request slot."""
    nh = divide(num_heads, tp_world_size)
    return (
        (nh, num_rope_angles),
        (nh, head_dim, d_state),
        (mimo_dim, nh, d_state),
        (nh, head_dim),
    )


def token_shift_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
) -> tuple[torch.dtype, torch.dtype]:
    """Dtypes of the Differential-TPA ``(k_last, v_last)`` buffer."""
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    return (state_dtype, state_dtype)


def token_shift_state_shape(
    tp_world_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Shapes of the Differential-TPA ``(k_last, v_last)`` buffer."""
    kv = divide(num_kv_heads, tp_world_size)
    return (kv, head_dim), (kv, head_dim)
