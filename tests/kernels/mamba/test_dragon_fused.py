# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the fused Dragon decode kernels.

- ``_geodesic_norm_kernel``: fused geodesic residual update vs the eager
  reference chain (and accuracy vs an fp32 ground truth).
- ``_dragon_decode_preamble_kernel`` / ``_dragon_bc_norm_kernel``: fused
  decode preamble vs the eager op-by-op computation.
"""
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.dragon_mamba3 import (
    _dragon_bc_norm_kernel,
    _dragon_decode_preamble_kernel,
)
from vllm.model_executor.models.dragon import DragonGeodesicNorm

H, D, R, S = 48, 64, 4, 128  # Dragon 7A1B dims
A_FLOOR = 1e-4
EPS = 1e-5

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("layer_idx", [0, 3, 17, 35])
@pytest.mark.parametrize("n", [1, 7, 257])
def test_geodesic_fp32_parity(layer_idx: int, n: int):
    """In fp32 the fused kernel must match the eager reference closely."""
    torch.manual_seed(0)
    m = DragonGeodesicNorm(layer_idx).to("cuda", torch.float32)
    with torch.no_grad():
        m.scale.copy_(torch.tensor(1.37))
        m.bias.copy_(torch.tensor(-0.21))
    x = torch.randn(n, 1536, device="cuda") * 3.0
    g = torch.randn(n, 1536, device="cuda")
    with torch.no_grad():
        out_f = m(x, g)
        out_r = m._forward_ref(x, g)
    rel = ((out_f - out_r).abs() / out_r.abs().clamp_min(1e-3)).max()
    assert rel.item() < 1e-3


@requires_cuda
def test_geodesic_bf16_closer_to_fp32_truth():
    """bf16 diffs vs eager are eager's own rounding: the fused kernel keeps
    fp32 intermediates and must be at least as close to fp32 ground truth."""
    torch.manual_seed(0)
    m16 = DragonGeodesicNorm(0).to("cuda", torch.bfloat16)
    m32 = DragonGeodesicNorm(0).to("cuda", torch.float32)
    x = torch.randn(4096, 1536, device="cuda") * 3.0
    g = torch.randn(4096, 1536, device="cuda")
    xb, gb = x.bfloat16(), g.bfloat16()
    with torch.no_grad():
        truth = m32._forward_ref(xb.float(), gb.float())
        fused = m16(xb, gb).float()
        eager = m16._forward_ref(xb, gb).float()
    assert (fused - truth).abs().mean() <= (eager - truth).abs().mean() * 1.05


@requires_cuda
@pytest.mark.parametrize("n", [1, 17, 256])
def test_decode_preamble_parity(n: int):
    torch.manual_seed(0)
    zxdt = torch.randn(n, H * (2 * D + 3), device="cuda",
                       dtype=torch.bfloat16) * 2
    dt_bias = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    x_f = torch.empty(n, H, D, device="cuda", dtype=torch.bfloat16)
    z_f = torch.empty_like(x_f)
    a_f = torch.empty(n, H, device="cuda", dtype=torch.float32)
    dt_f = torch.empty_like(a_f)
    tr_f = torch.empty_like(a_f)
    _dragon_decode_preamble_kernel[(n * H,)](
        zxdt, x_f, z_f, a_f, dt_f, tr_f, dt_bias, A_FLOOR, H,
        zxdt.stride(0), D=D)

    per_head = zxdt.view(n, H, 2 * D + 3)
    assert torch.equal(z_f, per_head[..., 0:D])
    assert torch.equal(x_f, per_head[..., D:2 * D])
    a_ref = torch.clamp(
        -F.softplus(per_head[..., 2 * D + 1].to(torch.float32)),
        max=-A_FLOOR)
    dt_ref = F.softplus(per_head[..., 2 * D].to(torch.float32)
                        + dt_bias.to(torch.float32))
    tr_ref = torch.sigmoid(per_head[..., 2 * D + 2].to(torch.float32))
    assert (a_f - a_ref).abs().max().item() < 3e-6
    assert (dt_f - dt_ref).abs().max().item() < 3e-6
    assert (tr_f - tr_ref).abs().max().item() < 3e-6


@requires_cuda
@pytest.mark.parametrize("n", [1, 17, 256])
def test_bc_norm_parity(n: int):
    torch.manual_seed(0)
    n_ang = 32
    bc = torch.randn(n, 2 * R * S + n_ang, device="cuda",
                     dtype=torch.bfloat16)
    wb = torch.rand(S, device="cuda", dtype=torch.bfloat16) + 0.5
    wc = torch.rand(S, device="cuda", dtype=torch.bfloat16) + 0.5
    out = torch.empty(n, 2, R, S, device="cuda", dtype=torch.bfloat16)
    _dragon_bc_norm_kernel[(n * 2 * R,)](
        bc, out, wb, wc, EPS, R * S, R, bc.stride(0),
        ZERO_CENTERED=False, S=S)

    def rms_ref(v, w):
        v32 = v.to(torch.float32)
        y = v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + EPS)
        return y * w.to(torch.float32)

    b_r = bc[:, :R * S].view(n, R, S)
    c_r = bc[:, R * S:2 * R * S].view(n, R, S)
    # fused keeps fp32 through the weight mul; eager rounds to bf16 between
    # ops — agree within bf16 ULP.
    assert (out[:, 0].float() - rms_ref(b_r, wb)).abs().max().item() < 2e-2
    assert (out[:, 1].float() - rms_ref(c_r, wc)).abs().max().item() < 2e-2
