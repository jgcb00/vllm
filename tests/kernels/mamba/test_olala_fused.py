# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the fused Olala decode kernels.

Each fused kernel replaces a chain of small eager ops on the decode hot path;
these tests pin them to the eager reference (and, for the geodesic update, to
an fp32 ground truth, since the fused kernel deliberately keeps fp32
intermediates where the eager chain rounds per op).
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.olala.mamba3 import (
    _bc_norm_kernel,
    _decode_preamble_kernel,
)
from vllm.model_executor.models.olala import OlalaGeodesicNorm

H, D, R, S = 48, 64, 4, 128  # Olala 7A1B dims
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
    m = OlalaGeodesicNorm(layer_idx).to("cuda", torch.float32)
    with torch.no_grad():
        m.scale.copy_(torch.tensor(1.37))
        m.bias.copy_(torch.tensor(-0.21))
    x = torch.randn(n, 1536, device="cuda") * 3.0
    g = torch.randn(n, 1536, device="cuda")
    with torch.no_grad():
        out_f = m(x, g)
        out_r = m.forward_native(x, g)
    rel = ((out_f - out_r).abs() / out_r.abs().clamp_min(1e-3)).max()
    assert rel.item() < 1e-3


@requires_cuda
def test_geodesic_bf16_closer_to_fp32_truth():
    """bf16 diffs against eager are eager's own rounding: the fused kernel
    keeps fp32 intermediates and must be at least as close to fp32 truth."""
    torch.manual_seed(0)
    m16 = OlalaGeodesicNorm(0).to("cuda", torch.bfloat16)
    m32 = OlalaGeodesicNorm(0).to("cuda", torch.float32)
    x = torch.randn(4096, 1536, device="cuda") * 3.0
    g = torch.randn(4096, 1536, device="cuda")
    xb, gb = x.bfloat16(), g.bfloat16()
    with torch.no_grad():
        truth = m32.forward_native(xb.float(), gb.float())
        fused = m16(xb, gb).float()
        eager = m16.forward_native(xb, gb).float()
    assert (fused - truth).abs().mean() <= (eager - truth).abs().mean() * 1.05


@requires_cuda
@pytest.mark.parametrize("n", [1, 17, 256])
def test_decode_preamble_parity(n: int):
    torch.manual_seed(0)
    zxdt = torch.randn(n, H * (2 * D + 3), device="cuda", dtype=torch.bfloat16) * 2
    dt_bias = torch.randn(H, device="cuda", dtype=torch.bfloat16)

    x_f = torch.empty(n, H, D, device="cuda", dtype=torch.bfloat16)
    z_f = torch.empty_like(x_f)
    a_f = torch.empty(n, H, device="cuda", dtype=torch.float32)
    dt_f = torch.empty_like(a_f)
    tr_f = torch.empty_like(a_f)
    _decode_preamble_kernel[(n * H,)](
        zxdt, x_f, z_f, a_f, dt_f, tr_f, dt_bias, A_FLOOR, H, zxdt.stride(0), D=D
    )

    per_head = zxdt.view(n, H, 2 * D + 3)
    assert torch.equal(z_f, per_head[..., 0:D])
    assert torch.equal(x_f, per_head[..., D : 2 * D])
    a_ref = torch.clamp(
        -F.softplus(per_head[..., 2 * D + 1].to(torch.float32)), max=-A_FLOOR
    )
    dt_ref = F.softplus(
        per_head[..., 2 * D].to(torch.float32) + dt_bias.to(torch.float32)
    )
    tr_ref = torch.sigmoid(per_head[..., 2 * D + 2].to(torch.float32))
    assert (a_f - a_ref).abs().max().item() < 3e-6
    assert (dt_f - dt_ref).abs().max().item() < 3e-6
    assert (tr_f - tr_ref).abs().max().item() < 3e-6


@requires_cuda
@pytest.mark.parametrize("n", [1, 17, 256])
def test_bc_norm_parity(n: int):
    torch.manual_seed(0)
    n_ang = 32
    bc = torch.randn(n, 2 * R * S + n_ang, device="cuda", dtype=torch.bfloat16)
    wb = torch.rand(S, device="cuda", dtype=torch.bfloat16) + 0.5
    wc = torch.rand(S, device="cuda", dtype=torch.bfloat16) + 0.5
    out = torch.empty(n, 2, R, S, device="cuda", dtype=torch.bfloat16)
    _bc_norm_kernel[(n * 2 * R,)](
        bc, out, wb, wc, EPS, R * S, R, bc.stride(0), ZERO_CENTERED=False, S=S
    )

    def rms_ref(v, w):
        v32 = v.to(torch.float32)
        y = v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + EPS)
        return y * w.to(torch.float32)

    b_r = bc[:, : R * S].view(n, R, S)
    c_r = bc[:, R * S : 2 * R * S].view(n, R, S)
    # The fused kernel keeps fp32 through the gain multiply where the eager
    # chain rounds to bf16 between ops; agree within bf16 ULP.
    assert (out[:, 0].float() - rms_ref(b_r, wb)).abs().max().item() < 2e-2
    assert (out[:, 1].float() - rms_ref(c_r, wc)).abs().max().item() < 2e-2


@requires_cuda
@pytest.mark.parametrize("n,npad", [(1, 0), (16, 0), (8, 3), (64, 8)])
def test_token_shift_decode_parity(n: int, npad: int):
    """Fused decode token-shift against the eager gather/blend/scatter chain.

    Cudagraph padding lanes carry NULL_BLOCK_ID (row 0, the reserved null
    block), so they must leave every live request's row untouched — writing to
    row 0 itself is harmless because no request is ever allocated that block.
    """
    from vllm.model_executor.layers.mamba.olala.diff_tpa import (
        _token_shift_decode_kernel,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    torch.manual_seed(0)
    dev = "cuda"
    Hkv, Dh, P = 12, 128, 128
    real = n - npad
    # Live requests never get the null block, so draw their slots from 1..P-1.
    slots = (torch.randperm(P - 1, device=dev)[:n] + 1).to(torch.int32)
    slots[real:] = NULL_BLOCK_ID
    k = torch.randn(n, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    ak = torch.randn(n, Hkv, device=dev, dtype=torch.bfloat16)
    av = torch.randn(n, Hkv, device=dev, dtype=torch.bfloat16)
    kp0 = torch.randn(P, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    vp0 = torch.randn(P, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    pos = torch.randint(1, 999, (n,), device=dev)
    if real > 0:
        pos[0] = 0  # document start disables the shift

    # Eager reference, blending in fp32 to match the kernel's accumulation.
    kpool_r, vpool_r = kp0.clone(), vp0.clone()
    rs = slots[:real].long()
    kprev = torch.zeros_like(k, dtype=torch.float32)
    vprev = torch.zeros_like(v, dtype=torch.float32)
    kprev[:real] = kpool_r[rs].float()
    vprev[:real] = vpool_r[rs].float()
    kpool_r[rs] = k[:real]
    vpool_r[rs] = v[:real]
    a_k = torch.sigmoid(ak.float()).unsqueeze(-1)
    a_v = torch.sigmoid(av.float()).unsqueeze(-1)
    doc = (pos == 0).view(-1, 1, 1)
    kprev = kprev.masked_fill(doc, 0)
    vprev = vprev.masked_fill(doc, 0)
    a_k = a_k.masked_fill(doc, 0)
    a_v = a_v.masked_fill(doc, 0)
    ks_r = (a_k * kprev + (1 - a_k) * k.float()).to(torch.bfloat16)
    vs_r = (a_v * vprev + (1 - a_v) * v.float()).to(torch.bfloat16)

    kpool_f, vpool_f = kp0.clone(), vp0.clone()
    ks_f = torch.empty_like(k)
    vs_f = torch.empty_like(v)
    _token_shift_decode_kernel[(n * Hkv,)](
        k,
        v,
        ks_f,
        vs_f,
        kpool_f,
        vpool_f,
        ak,
        av,
        slots,
        pos,
        Hkv,
        P,
        k.stride(0),
        k.stride(1),
        kpool_f.stride(0),
        kpool_f.stride(1),
        ak.stride(0),
        D=Dh,
    )
    torch.cuda.synchronize()

    # Blended outputs agree within 1 bf16 ULP (tl.sigmoid vs torch.sigmoid).
    assert (ks_f[:real].float() - ks_r[:real].float()).abs().max() <= 1e-2
    assert (vs_f[:real].float() - vs_r[:real].float()).abs().max() <= 1e-2
    # Pool stores are bit-exact for every live row (raw k/v, no arithmetic).
    assert torch.equal(kpool_f[1:], kpool_r[1:])
    assert torch.equal(vpool_f[1:], vpool_r[1:])
    if npad:
        # Padding lanes read the null block and blend against whatever it
        # holds; only their non-participation in live rows matters, asserted
        # above. Their own outputs are discarded by the caller.
        assert torch.isfinite(ks_f[real:].float()).all()
