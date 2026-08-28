# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode preamble / epilogue for Dragon's Differential-TPA layers.

For a pure-decode batch the eager path between the input projections and the
attention call is ~20 small kernels per layer: rank-``R`` K/V reconstruction
(bmm + div), the token-shift blend against the one-token pool, q/k RMSNorm,
and the scalable-softmax scaling of q. One program per (token, kv-head) does
all of it — the kv head's K/V rows and its ``snr + 1`` query heads — keeping
the eager bf16 rounding points. A second kernel folds the differential
recombination ``sig - sigmoid(lam) * noise`` after attention.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tpa_decode_qkv_kernel(
    q_ptr, sq_n, sq_h,                 # q_all (N, Hq*D) view: row stride, head stride
    ak_ptr, av_ptr, sa_n,              # A_k / A_v (N, Hkv*R): row stride
    bk_ptr, bv_ptr, sb_n,              # B_k / B_v (N, R*D): row stride
    alk_ptr, alv_ptr, sal_n,           # shift logits (N, Hkv): row stride
    pos_ptr, slots_ptr,
    kpool_ptr, vpool_ptr, sp_n, sp_h, P,
    qw_ptr, kw_ptr, eps,               # norm gains (D,), zero-centered
    scaler_ptr, wsize,                 # softmax scaler (Hq,), window clamp (0 = none)
    qo_ptr, ko_ptr, vo_ptr,            # outputs (N, Hq*D) / (N, Hkv*D) contiguous
    HKV: tl.constexpr,
    GQ: tl.constexpr,                  # query heads per kv head (= snr + 1)
    R: tl.constexpr,
    D: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)

    # ---- rank-R K/V reconstruction: bf16(sum_r A[r] * B[r]) / R ----------
    k = tl.zeros([D], dtype=tl.float32)
    v = tl.zeros([D], dtype=tl.float32)
    for r in range(R):
        ak_r = tl.load(ak_ptr + n * sa_n + h * R + r).to(tl.float32)
        av_r = tl.load(av_ptr + n * sa_n + h * R + r).to(tl.float32)
        bk_r = tl.load(bk_ptr + n * sb_n + r * D + offs).to(tl.float32)
        bv_r = tl.load(bv_ptr + n * sb_n + r * D + offs).to(tl.float32)
        k += ak_r * bk_r
        v += av_r * bv_r
    k = (k.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)
    v = (v.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)

    # ---- token shift against the one-token pool ---------------------------
    slot = tl.load(slots_ptr + n)
    srow = tl.minimum(tl.maximum(slot, 0), P - 1).to(tl.int64)
    kp = tl.load(kpool_ptr + srow * sp_n + h * sp_h + offs).to(tl.float32)
    vp = tl.load(vpool_ptr + srow * sp_n + h * sp_h + offs).to(tl.float32)
    a_k = tl.sigmoid(tl.load(alk_ptr + n * sal_n + h).to(tl.float32))
    a_v = tl.sigmoid(tl.load(alv_ptr + n * sal_n + h).to(tl.float32))
    pos = tl.load(pos_ptr + n)
    a_k = tl.where(pos == 0, 0.0, a_k)
    a_v = tl.where(pos == 0, 0.0, a_v)
    ks = (a_k * kp + (1.0 - a_k) * k).to(tl.bfloat16).to(tl.float32)
    vs = (a_v * vp + (1.0 - a_v) * v).to(tl.bfloat16)
    tl.store(kpool_ptr + srow * sp_n + h * sp_h + offs, k.to(kpool_ptr.dtype.element_ty))
    tl.store(vpool_ptr + srow * sp_n + h * sp_h + offs, v.to(vpool_ptr.dtype.element_ty))
    tl.store(vo_ptr + n * (HKV * D) + h * D + offs, vs)

    # ---- k norm (zero-centered gain), same rounding as DragonRMSNorm -------
    kw = (1.0 + tl.load(kw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=0) / D + eps)
    kn = (ks * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(ko_ptr + n * (HKV * D) + h * D + offs, (kn * kw).to(tl.bfloat16))

    # ---- q heads of this kv group: norm + scalable-softmax scale -----------
    qw = (1.0 + tl.load(qw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    p1 = tl.maximum(pos.to(tl.float32) + 1.0, 1.0)
    if wsize > 0:
        p1 = tl.minimum(p1, wsize)
    log_pos = tl.log(p1).to(tl.bfloat16).to(tl.float32)
    for j in range(GQ):
        hq = h * GQ + j
        q = tl.load(q_ptr + n * sq_n + hq * sq_h + offs).to(tl.float32)
        inv_q = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / D + eps)
        qn = (q * inv_q).to(tl.bfloat16).to(tl.float32)
        qn = (qn * qw).to(tl.bfloat16).to(tl.float32)
        sc = tl.load(scaler_ptr + hq).to(tl.bfloat16).to(tl.float32)
        scale = (sc * log_pos).to(tl.bfloat16).to(tl.float32)
        tl.store(qo_ptr + n * (HKV * GQ * D) + hq * D + offs, (qn * scale).to(tl.bfloat16))


@triton.jit
def _tpa_diff_combine_kernel(
    attn_ptr,          # (N, Hkv*(SNR+1)*D) contiguous
    lam_ptr, sl_n,     # lambda logits (N, Hkv): row stride
    out_ptr, so_n,     # (N, Hkv*SNR*D): row stride
    SNR: tl.constexpr,
    D: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)
    base = attn_ptr + n * tl.num_programs(1) * (SNR + 1) * D + h * (SNR + 1) * D
    noi = tl.load(base + SNR * D + offs).to(tl.float32)
    lam = tl.load(lam_ptr + n * sl_n + h).to(tl.float32)
    s = tl.sigmoid(lam).to(tl.bfloat16).to(tl.float32)
    t = (s * noi).to(tl.bfloat16).to(tl.float32)
    for j in range(SNR):
        sig = tl.load(base + j * D + offs).to(tl.float32)
        tl.store(out_ptr + n * so_n + (h * SNR + j) * D + offs, (sig - t).to(out_ptr.dtype.element_ty))


def tpa_decode_qkv(q_all, A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions, slots,
                   k_pool, v_pool, q_w, k_w, eps, scaler, wsize, hkv, gq, rank, d):
    n = q_all.shape[0]
    dt = q_all.dtype
    q_out = torch.empty(n, hkv * gq * d, dtype=dt, device=q_all.device)
    k_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    v_out = torch.empty(n, hkv * d, dtype=dt, device=q_all.device)
    _tpa_decode_qkv_kernel[(n, hkv)](
        q_all, q_all.stride(0), d,
        A_k, A_v, A_k.stride(0),
        B_k, B_v, B_k.stride(0),
        alpha_k, alpha_v, alpha_k.stride(0),
        positions, slots,
        k_pool, v_pool, k_pool.stride(0), k_pool.stride(1), k_pool.shape[0],
        q_w, k_w, eps, scaler, float(wsize),
        q_out, k_out, v_out,
        HKV=hkv, GQ=gq, R=rank, D=d, num_warps=2,
    )
    return q_out, k_out, v_out


def tpa_diff_combine(attn_out, lam_all, out, hkv, snr, d):
    n = attn_out.shape[0]
    _tpa_diff_combine_kernel[(n, hkv)](
        attn_out, lam_all, lam_all.stride(0), out, out.stride(0),
        SNR=snr, D=d, num_warps=2,
    )
