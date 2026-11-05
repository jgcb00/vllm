# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala ReplaySSM decode step (csrc/mamba3_replay_step.cu) against the plain
CUDA Mamba-3 step on an fp32 state, over enough steps to wrap the ring several
times with staggered flush targets and padded batch rows (PAD_SLOT_ID and the
null block 0 that CUDA-graph padding uses)."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.mamba3_step_cuda import (
    mamba3_replay_step,
    mamba3_step_cuda,
)
from vllm.model_executor.layers.mamba.olala.replay import replay_reset

requires_replay = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs an sm_89+ GPU (fp8)",
)

H, D, S, R, NA = 48, 64, 128, 4, 32


def _pools(P, L, anchor_dtype, read_fp8, ang0, ssm0, k0, v0):
    bf = torch.bfloat16
    pools = [
        ang0.clone(), ssm0.to(anchor_dtype).clone(), k0.clone(), v0.clone(),
        torch.zeros(P, H, L, D, device="cuda", dtype=bf),
        torch.zeros(P, L, R, S, device="cuda", dtype=bf),
        torch.zeros(P, H, L, NA, device="cuda"),
        torch.zeros(P, H, L, 4, device="cuda"),
        torch.zeros(P, H, 2, device="cuda", dtype=torch.int32),
    ]
    if read_fp8:
        pools += [
            torch.zeros(P, H, D, S, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(P, H, D, device="cuda"),
        ]
    return pools


@requires_replay
@pytest.mark.parametrize("L", [8, 16])
@pytest.mark.parametrize("anchor_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("read_fp8", [False, True])
def test_replay_step_matches_plain_step(L, anchor_dtype, read_fp8):
    torch.manual_seed(0)
    N, P, T = 24, 40, 3 * L + 5
    bf = torch.bfloat16
    slots = (torch.randperm(P - 1, device="cuda")[:N] + 1).to(torch.int32)
    slots[3] = -1  # padded row
    slots[5] = 0  # null block (CUDA-graph padding)
    valid = slots > 0
    ang0 = torch.rand(P, H, NA, device="cuda") * 6.28
    ssm0 = torch.randn(P, H, D, S, device="cuda") * 0.5
    k0 = torch.randn(P, R, H, S, device="cuda").to(bf)
    v0 = torch.randn(P, H, D, device="cuda").to(bf)
    xpj = torch.rand(R, H, D, device="cuda") * 0.5
    zpj = torch.rand(R, H, D, device="cuda")
    opj = torch.rand(R, H, D, device="cuda") * 0.5
    bq = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    bk = torch.randn(R, H, S, device="cuda") * 0.1 + 1
    Dp = torch.randn(H, device="cuda")

    pools = _pools(P, L, anchor_dtype, read_fp8, ang0, ssm0, k0, v0)
    # the reference starts from the same (possibly bf16-rounded) state, in fp32
    ref = [ang0.clone(), pools[1].float().clone(), k0.clone(), v0.clone()]
    replay_reset(tuple(pools), slots)

    # fp8 read copy: per-row e4m3 rounding of the checkpoint, scaled by its decay
    y_tol = 2e-2 if read_fp8 else 2e-3
    for _ in range(T):
        A = -torch.rand(N, H, device="cuda") * 2 - 0.05
        dt = torch.rand(N, H, device="cuda") * 0.3 + 0.01
        trap = torch.rand(N, H, device="cuda")
        BC = torch.randn(N, 2, R, S, device="cuda").to(bf)
        x = torch.randn(N, H, D, device="cuda").to(bf)
        z = torch.randn(N, H, D, device="cuda").to(bf)
        angp = torch.randn(N, NA, device="cuda").to(bf)
        y0 = torch.empty(N, H, D, device="cuda", dtype=bf)
        y1 = torch.empty_like(y0)
        mamba3_step_cuda(ref[1], ref[2], ref[3], ref[0], A, BC[:, 0], BC[:, 1], Dp, x, dt,
                         trap, xpj, zpj, opj, z, bq, bk, angp, slots, y0)
        mamba3_replay_step(tuple(pools), A, BC[:, 0], BC[:, 1], Dp, x, dt, trap, xpj, zpj,
                           opj, z, bq, bk, angp, slots, y1)
        a, b = y0[valid].float(), y1[valid].float()
        assert ((a - b).norm() / a.norm()).item() < y_tol
        assert torch.equal(y1[~valid], torch.zeros_like(y1[~valid]))
        torch.testing.assert_close(pools[0][slots[valid].long()], ref[0][slots[valid].long()])
        # right after a flush the anchor is the full state
        sl = slots[valid].long()
        flushed = sl[(pools[8][sl, :, 0] == 0).all(dim=1)]
        if flushed.numel():
            st0, st1 = ref[1][flushed], pools[1][flushed].float()
            # fp32 anchor: the flush GEMM is ~fp32 accurate; bf16: one rounding of the state
            s_tol = 1e-4 if anchor_dtype == torch.float32 else 5e-3
            assert ((st0 - st1).norm() / st0.norm()).item() < s_tol
            torch.testing.assert_close(pools[2][flushed], ref[2][flushed], atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(pools[3][flushed], ref[3][flushed])
        if anchor_dtype != torch.float32:
            # the reference keeps the fp32 recurrence the replay anchor approximates
            # at each flush: re-sync it so per-step errors do not compound in the check
            ref[1][flushed] = pools[1][flushed].float()
