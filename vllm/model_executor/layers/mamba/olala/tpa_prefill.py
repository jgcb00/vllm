# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused prefill glue of the TPA-factorized DiffTPA layer.

Replaces the per-token torch chain of ``OlalaDiffTPAAttention._factor_prefill`` (k/v rank
contractions, previous-token seeds from the batch or the shift pools, shift gates, k RMS
norm, factor rows, row scatter, pool updates) and of ``_q_for_attention`` (q RMS norm and
scalable-softmax scale) by three kernels, with the same bf16 roundings as the torch code.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tpa_rows_kernel(
    Ak_ptr, Av_ptr, Bk_ptr, Bv_ptr, ak_ptr, av_ptr, rs,  # projection slices, common row stride
    pos_ptr, slots_ptr, starts_ptr, ends_ptr, pslots_ptr, init_ptr,
    kpool_ptr, vpool_ptr, akpool_ptr, avpool_ptr, s_kp, s_vp, s_akp, s_avp,
    kw_ptr, eps,
    kplane_ptr, vplane_ptr, bstride, block_size,
    H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
):
    """(request, BT-token tile): K and V factor rows [B' | C | D] of every token, written to the
    paged planes at the token's slot (slots < 0 skipped)."""
    req = tl.program_id(0)
    t0 = tl.load(starts_ptr + req)
    t_end = tl.load(ends_ptr + req)
    t = t0 + tl.program_id(1) * BT + tl.arange(0, BT)
    if tl.program_id(1) * BT >= t_end - t0:
        return
    tm = t < t_end
    first = t == t0
    inbatch = tm & (t > t0)
    pslot = tl.load(pslots_ptr + req).to(tl.int64)
    hinit = tl.load(init_ptr + req) != 0
    seed = first & hinit
    hh = tl.arange(0, HP)
    hm = hh < H
    d = tl.arange(0, D)
    t64 = t.to(tl.int64)
    pos = tl.load(pos_ptr + t, mask=tm, other=1)
    keep = (pos != 0)[:, None]
    m2 = tm[:, None] & hm[None, :]
    a_k = tl.where(keep, tl.sigmoid(tl.load(ak_ptr + t64[:, None] * rs + hh[None, :], mask=m2, other=0.0).to(tl.float32)), 0.0)
    a_v = tl.where(keep, tl.sigmoid(tl.load(av_ptr + t64[:, None] * rs + hh[None, :], mask=m2, other=0.0).to(tl.float32)), 0.0)
    # k = (A_k @ B_k) / R for this token and the previous one (bf16 rounding of the bmm, then /R exact)
    k = tl.zeros([BT, HP, D], dtype=tl.float32)
    kprev = tl.zeros([BT, HP, D], dtype=tl.float32)
    for r in tl.static_range(R):
        a_r = tl.load(Ak_ptr + t64[:, None] * rs + hh[None, :] * R + r, mask=m2, other=0.0).to(tl.float32)
        b_r = tl.load(Bk_ptr + t64[:, None] * rs + r * D + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        k += a_r[:, :, None] * b_r[:, None, :]
        ap_r = tl.load(Ak_ptr + (t64 - 1)[:, None] * rs + hh[None, :] * R + r, mask=inbatch[:, None] & hm[None, :], other=0.0).to(tl.float32)
        bp_r = tl.load(Bk_ptr + (t64 - 1)[:, None] * rs + r * D + d[None, :], mask=inbatch[:, None], other=0.0).to(tl.float32)
        kprev += ap_r[:, :, None] * bp_r[:, None, :]
    k = k.to(tl.bfloat16).to(tl.float32) / R
    kprev = kprev.to(tl.bfloat16).to(tl.float32) / R
    # pool seeds are cast to the activation dtype first, as pool[pslots].to(cur.dtype)
    kseed = tl.load(kpool_ptr + pslot * s_kp + hh[:, None] * D + d[None, :], mask=hm[:, None] & hinit, other=0.0).to(tl.bfloat16).to(tl.float32)
    kprev = tl.where(seed[:, None, None], kseed[None, :, :], kprev)
    a3 = a_k[:, :, None]
    ks = (a3 * kprev + (1.0 - a3) * k).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=2) / D + eps)  # [BT, HP]
    # K row
    kw = (tl.load(kw_ptr + d).to(tl.float32) + 1.0).to(tl.bfloat16).to(tl.float32)
    slot = tl.load(slots_ptr + t, mask=tm, other=-1).to(tl.int64)
    ok = tm & (slot >= 0)
    kbase = kplane_ptr + (slot // block_size) * bstride + (slot % block_size) * (R * D + 2 * H * R)
    vbase = vplane_ptr + (slot // block_size) * bstride + (slot % block_size) * (R * D + 2 * H * R)
    for r in tl.static_range(R):
        b_r = tl.load(Bk_ptr + t64[:, None] * rs + r * D + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        tl.store(kbase[:, None] + r * D + d[None, :], (b_r * kw[None, :]).to(kplane_ptr.dtype.element_ty), mask=ok[:, None])
        bv_r = tl.load(Bv_ptr + t64[:, None] * rs + r * D + d[None, :], mask=tm[:, None], other=0.0)
        tl.store(vbase[:, None] + r * D + d[None, :], bv_r.to(vplane_ptr.dtype.element_ty), mask=ok[:, None])
        a_r = tl.load(Ak_ptr + t64[:, None] * rs + hh[None, :] * R + r, mask=m2, other=0.0).to(tl.float32)
        ap_r = tl.load(Ak_ptr + (t64 - 1)[:, None] * rs + hh[None, :] * R + r, mask=inbatch[:, None] & hm[None, :], other=0.0).to(tl.float32)
        ap_r = tl.where(seed[:, None], tl.load(akpool_ptr + pslot * s_akp + hh * R + r, mask=hm & hinit, other=0.0).to(tl.bfloat16).to(tl.float32)[None, :], ap_r)
        scale = (1.0 - a_k) / R * inv
        dscale = a_k / R * inv
        cofs = R * D + hh[None, :] * R + r
        tl.store(kbase[:, None] + cofs, (a_r * scale).to(kplane_ptr.dtype.element_ty), mask=ok[:, None] & hm[None, :])
        tl.store(kbase[:, None] + cofs + H * R, (ap_r * dscale).to(kplane_ptr.dtype.element_ty), mask=ok[:, None] & hm[None, :])
        av_r = tl.load(Av_ptr + t64[:, None] * rs + hh[None, :] * R + r, mask=m2, other=0.0).to(tl.float32)
        avp_r = tl.load(Av_ptr + (t64 - 1)[:, None] * rs + hh[None, :] * R + r, mask=inbatch[:, None] & hm[None, :], other=0.0).to(tl.float32)
        avp_r = tl.where(seed[:, None], tl.load(avpool_ptr + pslot * s_avp + hh * R + r, mask=hm & hinit, other=0.0).to(tl.bfloat16).to(tl.float32)[None, :], avp_r)
        tl.store(vbase[:, None] + cofs, (av_r * ((1.0 - a_v) / R)).to(vplane_ptr.dtype.element_ty), mask=ok[:, None] & hm[None, :])
        tl.store(vbase[:, None] + cofs + H * R, (avp_r * (a_v / R)).to(vplane_ptr.dtype.element_ty), mask=ok[:, None] & hm[None, :])


@triton.jit
def _tpa_pools_kernel(
    Ak_ptr, Av_ptr, Bk_ptr, Bv_ptr, rs, ends_ptr, pslots_ptr,
    kpool_ptr, vpool_ptr, akpool_ptr, avpool_ptr, s_kp, s_vp, s_akp, s_avp,
    H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
):
    """Shift pools of one request <- its last token's k, v, A_k, A_v."""
    req = tl.program_id(0)
    l = (tl.load(ends_ptr + req) - 1).to(tl.int64)
    pslot = tl.load(pslots_ptr + req).to(tl.int64)
    hh = tl.arange(0, HP)
    hm = hh < H
    d = tl.arange(0, D)
    k = tl.zeros([HP, D], dtype=tl.float32)
    v = tl.zeros([HP, D], dtype=tl.float32)
    for r in tl.static_range(R):
        ak = tl.load(Ak_ptr + l * rs + hh * R + r, mask=hm, other=0.0)
        av = tl.load(Av_ptr + l * rs + hh * R + r, mask=hm, other=0.0)
        k += ak.to(tl.float32)[:, None] * tl.load(Bk_ptr + l * rs + r * D + d).to(tl.float32)[None, :]
        v += av.to(tl.float32)[:, None] * tl.load(Bv_ptr + l * rs + r * D + d).to(tl.float32)[None, :]
        tl.store(akpool_ptr + pslot * s_akp + hh * R + r, ak.to(akpool_ptr.dtype.element_ty), mask=hm)
        tl.store(avpool_ptr + pslot * s_avp + hh * R + r, av.to(avpool_ptr.dtype.element_ty), mask=hm)
    k = k.to(tl.bfloat16).to(tl.float32) / R
    v = v.to(tl.bfloat16).to(tl.float32) / R
    tl.store(kpool_ptr + pslot * s_kp + hh[:, None] * D + d[None, :], k.to(kpool_ptr.dtype.element_ty), mask=hm[:, None])
    tl.store(vpool_ptr + pslot * s_vp + hh[:, None] * D + d[None, :], v.to(vpool_ptr.dtype.element_ty), mask=hm[:, None])


@triton.jit
def _tpa_q_kernel(
    q_ptr, rs, pos_ptr, w_ptr, scaler_ptr, out_ptr, eps, wsize,
    HQ: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, HAS_WS: tl.constexpr,
):
    """q <- (rms_norm(q) * (1 + w)) * (scaler[h] * log(clamp(pos + 1, 1, wsize))), torch roundings."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    hm = h < HQ
    d = tl.arange(0, D)
    x = tl.load(q_ptr + t * rs + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    w = (tl.load(w_ptr + d).to(tl.float32) + 1.0).to(tl.bfloat16).to(tl.float32)
    y = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(pos_ptr + t).to(tl.float32) + 1.0
    pos = tl.maximum(pos, 1.0)
    if HAS_WS:
        pos = tl.minimum(pos, wsize)
    lp = tl.log(pos).to(tl.bfloat16).to(tl.float32)
    sc = (tl.load(scaler_ptr + h, mask=hm, other=0.0).to(tl.bfloat16).to(tl.float32) * lp).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t * (HQ * D) + h[:, None] * D + d[None, :], (y * sc[:, None]).to(out_ptr.dtype.element_ty), mask=hm[:, None])


def tpa_prefill_rows(A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions, slots, starts, ends, pslots, init,
                     pools, k_norm_w, eps, kplane, vplane, H, R, D):
    """Factor rows of every prefill token into the paged planes, then the shift pools."""
    n = A_k.shape[0]
    nk = starts.shape[0]
    if n == 0 or nk == 0:
        return
    rs = A_k.stride(0)
    for x in (A_v, B_k, B_v, alpha_k, alpha_v):
        assert x.stride(0) == rs and x.stride(-1) == 1
    k_pool, v_pool, ak_pool, av_pool = pools
    assert kplane.stride(1) == R * D + 2 * H * R and kplane.stride(2) == 1 and vplane.stride(0) == kplane.stride(0)
    BT = 4
    HP = triton.next_power_of_2(H)
    max_len = n  # upper bound on any request's tokens in this batch (grid rows past a request return)
    _tpa_rows_kernel[(nk, triton.cdiv(max_len, BT))](
        A_k, A_v, B_k, B_v, alpha_k, alpha_v, rs, positions, slots, starts, ends, pslots, init,
        k_pool, v_pool, ak_pool, av_pool, k_pool.stride(0), v_pool.stride(0), ak_pool.stride(0), av_pool.stride(0),
        k_norm_w, eps, kplane, vplane, kplane.stride(0), kplane.shape[1],
        H=H, HP=HP, R=R, D=D, BT=BT, num_warps=8,
    )
    _tpa_pools_kernel[(nk,)](
        A_k, A_v, B_k, B_v, rs, ends, pslots,
        k_pool, v_pool, ak_pool, av_pool, k_pool.stride(0), v_pool.stride(0), ak_pool.stride(0), av_pool.stride(0),
        H=H, HP=HP, R=R, D=D, num_warps=4,
    )


def tpa_prefill_q(q_all, positions, q_norm_w, scaler, eps, wsize, HQ, D):
    n = q_all.shape[0]
    out = torch.empty(n, HQ * D, device=q_all.device, dtype=q_all.dtype)
    if n == 0:
        return out.view(n, HQ, D)
    HB = 8
    _tpa_q_kernel[(n, triton.cdiv(HQ, HB))](
        q_all, q_all.stride(0), positions, q_norm_w, scaler, out, eps, float(wsize),
        HQ=HQ, HB=HB, D=D, HAS_WS=wsize > 0, num_warps=4,
    )
    return out.view(n, HQ, D)
