# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request recurrent state layouts for the Olala mixers.

Olala carries two kinds of paged recurrent state, both allocated through
``MambaSpec`` so vLLM's state manager owns the per-request slots:

* Mamba3 MIMO (``M`` layers) — four temporal tensors.
* Differential-TPA token shift (``V`` layers) — a one-token ``(k, v)``
  buffer holding the previous token's raw (pre-shift) key/value.

These live here rather than in the shared ``mamba_utils`` calculators so the
fork touches one fewer upstream file.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)


@dataclass(frozen=True)
class ReplayConfig:
    """ReplaySSM decode for the Mamba3 layers (``mamba3_replay_step.cu``).

    ``OLALA_REPLAY_SSM=on`` (or ``=<L>``, 2 <= L <= 32; unset/0 = off) keeps
    the SSM state as a checkpoint plus a ring of the last tokens' inputs: the
    full state is rebuilt in fp32 and written back only once every ``L``
    decode steps. ``OLALA_REPLAY_READ=fp8`` (default) adds an fp8 e4m3 read
    copy of the checkpoint (one fp32 scale per 128-wide state row) that serves
    the other steps; ``none`` reads the anchor itself. The anchor is bf16,
    stochastically rounded once per window, unless ``--mamba-ssm-cache-dtype``
    asks for another dtype (float32).

    The defaults (L = 8, bf16 anchor, fp8 read) are the measured optimum on
    Olala 7A1B / H100: L = 6..8 beats the paper's 16 with a bf16 anchor (a
    flush is cheap, and the window replayed each step is shorter), and over
    24k teacher-forced tokens the output KL to an fp32 state stays within
    ~2x of the fp32-vs-fp32 kernel noise, where a bf16 state rounded every
    step drifts to ~85x.
    """

    buffer_len: int
    read_fp8: bool


DEFAULT_REPLAY_BUFFER_LEN = 8


@lru_cache(maxsize=1)
def replay_config() -> ReplayConfig | None:
    v = os.environ.get("OLALA_REPLAY_SSM", "0").strip().lower() or "0"
    if v in ("on", "true", "yes", "1"):
        v = str(DEFAULT_REPLAY_BUFFER_LEN)
    L = int(v)
    if L <= 0:
        return None
    if not 2 <= L <= 32:
        raise ValueError(f"OLALA_REPLAY_SSM must be in [2, 32], got {L}")
    read = os.environ.get("OLALA_REPLAY_READ", "fp8")
    if read not in ("fp8", "none"):
        raise ValueError(f"OLALA_REPLAY_READ must be fp8 or none, got {read!r}")
    return ReplayConfig(buffer_len=L, read_fp8=read == "fp8")


def mamba3_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
    mamba_ssm_cache_dtype: MambaDType,
) -> tuple[torch.dtype, ...]:
    """Dtypes of ``(angle, ssm, k, v)``, then the ReplaySSM tensors if enabled
    (see ``mamba3_state_shape``).

    The angle state accumulates rotary phase across the whole sequence, so it
    is always fp32 — a bf16 accumulator visibly drifts over long contexts.
    The SSM state honors ``--mamba-ssm-cache-dtype``; under the default
    ``auto`` it is stored fp32: re-rounding the state to bf16 after every
    decode step compounds over long generations and was linked to
    repetition loops (bf16 storage is ~16% faster at high concurrency; opt in
    with ``--mamba-ssm-cache-dtype bfloat16``). With ReplaySSM, ``auto`` is a
    bf16 anchor, which that compounding does not apply to.
    """
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    rc = replay_config()
    if mamba_ssm_cache_dtype == "auto":
        # ReplaySSM rounds its anchor once per window, not every step: bf16
        # holds fp32 quality there (see ReplayConfig)
        ssm_dtype = torch.bfloat16 if rc is not None else torch.float32
    else:
        ssm_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
    dtypes = (torch.float32, ssm_dtype, state_dtype, state_dtype)
    if rc is not None:
        dtypes += (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32, torch.int32)
        if rc.read_fp8:
            dtypes += (torch.float8_e4m3fn, torch.float32)
    return dtypes


def mamba3_state_shape(
    tp_world_size: int,
    num_heads: int,
    head_dim: int,
    d_state: int,
    mimo_dim: int,
    num_rope_angles: int,
) -> tuple[tuple[int, ...], ...]:
    """Shapes of ``(angle, ssm, k, v)`` per request slot.

    With ReplaySSM, ``ssm`` is the checkpoint anchor and the ring follows:
    ``x (nh, L, head_dim)``, pre-bias/rotation ``B (L, mimo_dim, d_state)``
    (shared by the heads: one group), the rotary phase ``(nh, L, angles)``,
    ``(A*dt, gamma, beta, -) (nh, L, 4)`` and ``(count, flush target)
    (nh, 2)``; then the fp8 read copy ``(nh, head_dim, d_state)`` and its
    per-row scales ``(nh, head_dim)`` if enabled.
    """
    nh = divide(num_heads, tp_world_size)
    shapes: tuple[tuple[int, ...], ...] = (
        (nh, num_rope_angles),
        (nh, head_dim, d_state),
        (mimo_dim, nh, d_state),
        (nh, head_dim),
    )
    rc = replay_config()
    if rc is not None:
        L = rc.buffer_len
        shapes += (
            (nh, L, head_dim),
            (L, mimo_dim, d_state),
            (nh, L, num_rope_angles),
            (nh, L, 4),
            (nh, 2),
        )
        if rc.read_fp8:
            shapes += ((nh, head_dim, d_state), (nh, head_dim))
    return shapes


def token_shift_state_dtype(
    model_dtype: ModelDType | torch.dtype,
    mamba_cache_dtype: MambaDType,
    rank: int = 0,
) -> tuple[torch.dtype, ...]:
    """Dtypes of the Differential-TPA ``(k_last, v_last[, A_k_last, A_v_last])`` buffer."""
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    return (state_dtype,) * (4 if rank else 2)


def token_shift_state_shape(
    tp_world_size: int,
    num_kv_heads: int,
    head_dim: int,
    rank: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Shapes of the Differential-TPA ``(k_last, v_last)`` buffer.

    With ``rank`` > 0 (TPA-factorized KV cache) the previous token's rank
    coefficients ``(A_k_last, A_v_last)`` are kept as well: the factor row of
    token ``t`` carries ``alpha_t * A[t-1]`` for the token-shift term.
    """
    kv = divide(num_kv_heads, tp_world_size)
    shapes: tuple[tuple[int, int], ...] = ((kv, head_dim), (kv, head_dim))
    if rank:
        shapes += ((kv, rank), (kv, rank))
    return shapes
