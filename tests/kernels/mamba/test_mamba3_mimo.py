# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba-3 MIMO inference kernels (ops/mamba3) at the Olala shapes."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="Mamba-3 MIMO kernels need CUDA sm90+",
)

H, N, P, R, A = 48, 128, 64, 4, 32
CHUNK = 64 // R


def _inputs(lens: list[int], seed: int = 0) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, scale=1.0):
        return torch.randn(*s, device="cuda", generator=g) * scale

    def ru(*s, scale=1.0):
        return torch.rand(*s, device="cuda", generator=g) * scale

    T = sum(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    return dict(
        Q=rn(1, T, R, 1, N).bfloat16(),
        K=rn(1, T, R, 1, N).bfloat16(),
        V=rn(1, T, H, P).bfloat16(),
        ADT=-ru(1, H, T, scale=0.1),
        DT=ru(1, H, T, scale=0.1),
        Trap=rn(1, H, T).bfloat16(),
        Q_bias=rn(H, R, N),
        K_bias=rn(H, R, N),
        MIMO_V=rn(H, R, P),
        MIMO_Z=rn(H, R, P),
        MIMO_Out=rn(H, R, P),
        Angles=rn(1, T, H, A),
        D=rn(H),
        Z=rn(1, T, H, P).bfloat16(),
        chunk_size=CHUNK,
        rotary_dim_divisor=4,
        dtype=torch.bfloat16,
        cu_seqlens=cu.cuda(),
    )


def _slice(kw: dict, a: int, b: int) -> dict:
    """Tokens [a, b) of a single packed sequence."""
    out = dict(kw)
    for k in ("Q", "K", "V", "Angles", "Z"):
        out[k] = kw[k][:, a:b]
    for k in ("ADT", "DT", "Trap"):
        out[k] = kw[k][:, :, a:b]
    out["cu_seqlens"] = torch.tensor([0, b - a], dtype=torch.int32, device="cuda")
    return out


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float) -> None:
    err = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
    assert err < rtol, f"relative max error {err:.3e} >= {rtol}"


@torch.inference_mode()
def test_input_states_continuation():
    """Prefill in two chunks (second resumes from the first's final state)
    matches the one-shot prefill: the chunked-prefill / prefix-cache contract."""
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo import mamba3_mimo

    L, split = 700, 333
    kw = _inputs([L])
    full = mamba3_mimo(**kw, return_state=True)
    first = mamba3_mimo(**_slice(kw, 0, split), return_state=True)
    second = mamba3_mimo(
        **_slice(kw, split, L), return_state=True, Input_States=tuple(first[1:])
    )
    _close(torch.cat([first[0], second[0]], dim=1), full[0], 2e-2)
    for got, ref in zip(second[1:], full[1:]):
        _close(got, ref, 2e-2)


@torch.inference_mode()
def test_grouped_prefill_matches_single_pass():
    """Group-parallel prefill (long sequences split into virtual groups) has
    the exact contract of the single-pass kernel."""
    from vllm.model_executor.layers.mamba.ops.mamba3.mimo import (
        mamba3_mimo,
        mamba3_mimo_varlen_grouped,
    )

    kw = _inputs([5000, 200, 4200], seed=1)
    ref = mamba3_mimo(**kw, return_state=True)
    got = mamba3_mimo_varlen_grouped(**kw, cu_seqlens_cpu=kw["cu_seqlens"].cpu())
    for g, r in zip(got, ref):
        _close(g, r, 2e-2)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cuda_step_matches_cute_step(state_dtype: torch.dtype):
    """The persistent CUDA decode step agrees with the CuteDSL reference step
    (fused bias + rotary, in-place pool update, scattered slots)."""
    from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
        mamba3_step_cuda,
    )
    from vllm.model_executor.layers.mamba.ops.mamba3.step_cute import mamba3_step_fn

    g = torch.Generator(device="cuda").manual_seed(2)
    Bs, pool = 16, 20

    def rn(*s):
        return torch.randn(*s, device="cuda", generator=g)

    pools = dict(
        ssm=rn(pool, H, P, N).to(state_dtype),
        kp=rn(pool, R, H, N).bfloat16(),
        vp=rn(pool, H, P).bfloat16(),
        ang=rn(pool, H, A),
    )
    a = -torch.rand(Bs, H, device="cuda", generator=g) - 0.1
    b, c = rn(Bs, R, N).bfloat16(), rn(Bs, R, N).bfloat16()
    d, x = rn(H), rn(Bs, H, P).bfloat16()
    dt = torch.rand(Bs, H, device="cuda", generator=g)
    trap = torch.rand(Bs, H, device="cuda", generator=g)
    xpj, opj, zpj, z = rn(R, H, P), rn(R, H, P), rn(R, H, P), rn(Bs, H, P).bfloat16()
    bq, bk, angp = rn(R, H, N), rn(R, H, N), rn(Bs, A).bfloat16()
    slots = torch.randperm(pool, device="cuda", generator=g)[:Bs].to(torch.int32)

    ref = {k: v.clone() for k, v in pools.items()}
    y_ref = torch.empty(Bs, H, P, device="cuda", dtype=torch.bfloat16)
    mamba3_step_fn(
        ref["ssm"], ref["kp"], ref["vp"], a,
        b.unsqueeze(2).expand(Bs, R, H, N), c.unsqueeze(2).expand(Bs, R, H, N),
        d, x, dt, trap, xpj, opj, None, y_ref,
        z=z, zproj=zpj, state_batch_indices=slots, update_kv_state=True,
        tile_D=64, num_warps=4, rotary_dim=2 * A, rotary_bias_q=bq,
        rotary_bias_k=bk, rotary_angle_proj=angp.unsqueeze(1).expand(Bs, H, A),
        rotary_angle_state=ref["ang"],
    )  # fmt: skip
    got = {k: v.clone() for k, v in pools.items()}
    y = torch.empty_like(y_ref)
    mamba3_step_cuda(
        got["ssm"], got["kp"], got["vp"], got["ang"], a, b, c, d, x, dt, trap,
        xpj, zpj, opj, z, bq, bk, angp, slots, y,
    )  # fmt: skip
    # An fp32 state is never rounded, so y and the state agree tightly; the
    # bf16 pools (and a bf16 state) carry bf16 rounding of the rotated B.
    tight = 1e-3 if state_dtype == torch.float32 else 2e-2
    _close(y, y_ref, tight)
    _close(got["ssm"], ref["ssm"], tight)
    _close(got["ang"], ref["ang"], 1e-5)
    for k in ("kp", "vp"):
        _close(got[k], ref[k], 2e-2)
