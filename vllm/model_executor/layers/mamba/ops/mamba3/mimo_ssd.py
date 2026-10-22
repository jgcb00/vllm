# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-parallel (SSD-style) Mamba-3 MIMO prefill for Olala.

Same math as the TileLang ``mamba_mimo_fwd`` varlen kernel (reduceO, Z gate, D skip,
initial states), restructured so a prefill chunk is not walked serially per head:

  per head h, token t (ranks r, state N x P):
    S_t = exp(ADT_t) S_{t-1} + ts_t * sum_r khat[t,r]^T PsiV[t,r]
    y[t,r] = qhat[t,r] . (exp(ADT_t) S_{t-1} + gamma_t * sum_r' khat[t,r']^T PsiV[t,r']) + D PsiV[t,r]
    out[t] = sum_r y[t,r] * Phi[r] * silu(Z[t] * Zeta[r])
  with gamma_t = DT_t sigmoid(Trap_t), ts_t = gamma_t + DT_{t+1} sigmoid(-Trap_{t+1}) (0 past the
  sequence end), PsiV[t,r] = V[t] * Psi[r], qhat/khat = (Q/K + bias) rotated on the (n, n + N/2),
  n < N/4 pairs by theta_t = sum_{s<=t} tanh(angle_s) pi DT_s (mod 2 pi).

Pipeline of ``mamba3_mimo_hybrid_pools`` (chunks of LS = 128 tokens never cross a sequence; rows of a
tile are token-major t*R + r):
  0. ``_ssd_prep``        (chunk, head): per-token chunk-local log-decay cumsum (and its 16-token-group
     variants), trapezoid scales, chunk-local phase cumsum; chunk totals.
  1. chunk-local states: CUDA ``chunk_state`` (olala/csrc/mamba3_scan_prefill.cu, sm90) or the Triton
     ``_ssd_chunk_state``.
  2. ``_ssd_state_pass_pool`` (head, sequence): serial over chunks, S_in[c] = state entering chunk c,
     read from / written back to the request's pool slot; a chunk's local state is rotated by the
     phase prefix (rot(a + b) = rot(a) rot(b)).
  3. outputs: every chunk is an independent virtual sequence starting from S_in[c]: CUDA ``scan``
     (sm90, wgmma) or the TileLang ``mamba_mimo_fwd_v2`` kernel (``OLALA_TL_V2=0``: v1).
  4. ``_ssd_pool_tail``: final angle / K / V of every sequence into its pool slot.
Rotary columns are four NQ = N/4 wide groups: g0/g2 rotate together, g1/g3 do not.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

import os as _os

_TL_V2 = _os.environ.get("OLALA_TL_V2", "1") == "1"
# sm_90 CUDA kernels (wgmma) for the output pass and the chunk-local states; TileLang / Triton elsewhere
_SCAN_CUDA = _os.environ.get("OLALA_SCAN_CUDA", "1") == "1"
_CS_CUDA = _os.environ.get("OLALA_CS_CUDA", "1") == "1"
_SCAN_EXT = None


def _scan_cuda_ok(dev) -> bool:
    return _SCAN_CUDA and torch.cuda.get_device_capability(dev) == (9, 0)


def _scan_ext():
    """CUDA output pass of the hybrid scan (olala/csrc/mamba3_scan_prefill.cu), built JIT."""
    global _SCAN_EXT
    if _SCAN_EXT is None:
        from vllm.model_executor.layers.mamba.olala.jit import load_extension
        _SCAN_EXT = load_extension("olala_mamba3_scan_prefill", "mamba3_scan_prefill.cu",
                                   ["-O3", "-std=c++17", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"])
    return _SCAN_EXT


@triton.jit
def _tanh(x):
    return tl.sigmoid(2.0 * x) * 2.0 - 1.0


@triton.jit
def _mod2pi(x):
    return x - 6.283185307179586 * tl.floor(x / 6.283185307179586)


@triton.jit
def _ssd_prep(
    ang_ptr, DT_ptr, ADT_ptr, TRAP_ptr, ch_start_ptr, ch_len_ptr, ch_send_ptr,
    dacs_ptr, ts_ptr, gam_ptr, thl_ptr, dtot_ptr, asum_ptr, cl_ptr, sl_ptr,
    T, H, NQ: tl.constexpr, LS: tl.constexpr,
    dcs16_ptr=None, dcsr16_ptr=None, G16: tl.constexpr = 0, WRITE_CS: tl.constexpr = True,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    send = tl.load(ch_send_ptr + c)
    i = tl.arange(0, LS)
    tm = i < cl
    t = cs + i
    adt = tl.load(ADT_ptr + h * T + t, mask=tm, other=0.0)
    dtv = tl.load(DT_ptr + h * T + t, mask=tm, other=0.0)
    trv = tl.load(TRAP_ptr + h * T + t, mask=tm, other=0.0).to(tl.float32)
    nm = tm & (t + 1 < send)
    dtn = tl.load(DT_ptr + h * T + t + 1, mask=nm, other=0.0)
    trn = tl.load(TRAP_ptr + h * T + t + 1, mask=nm, other=0.0).to(tl.float32)
    gam = dtv * tl.sigmoid(trv)
    tl.store(dacs_ptr + h * T + t, tl.cumsum(adt, axis=0), mask=tm)
    if G16 > 0:
        # TileLang chunk-of-G16 cumsums (compute_dacs_segsum_triton_varlen's da_cs / da_cs_rev, clipped <= 0)
        a2 = tl.reshape(adt, (LS // G16, G16))
        cs2 = tl.cumsum(a2, axis=1)
        rev2 = tl.cumsum(a2, axis=1, reverse=True) - a2
        tl.store(dcs16_ptr + h * T + t, tl.reshape(tl.minimum(cs2, 0.0), (LS,)), mask=tm)
        tl.store(dcsr16_ptr + h * T + t, tl.reshape(tl.minimum(rev2, 0.0), (LS,)), mask=tm)
    tl.store(ts_ptr + h * T + t, gam + dtn * tl.sigmoid(-trn), mask=tm)
    tl.store(gam_ptr + h * T + t, gam, mask=tm)
    tl.store(dtot_ptr + c * H + h, tl.sum(adt, axis=0))
    nq = tl.arange(0, NQ)
    a = tl.load(ang_ptr + t[:, None] * NQ + nq[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    inc = _tanh(a) * 3.141592653589793 * dtv[:, None]
    thl = tl.cumsum(inc, axis=0)
    tl.store(thl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], thl, mask=tm[:, None])
    if WRITE_CS:  # fp16 cos / sin tables for the Triton chunk-state kernels
        tl.store(cl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], tl.cos(thl).to(tl.float16), mask=tm[:, None])
        tl.store(sl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], tl.sin(thl).to(tl.float16), mask=tm[:, None])
    tl.store(asum_ptr + (c * H + h) * NQ + nq, tl.sum(inc, axis=0))


@triton.jit
def _rot_coef_tab(cl_ptr, sl_ptr, ca0, sa0, h, tok0, m, rm, H, R: tl.constexpr, NQ: tl.constexpr):
    """cos/sin of a0 + theta_local from the fp16 local tables: the angle-sum identities."""
    nq = tl.arange(0, NQ)
    off = ((tok0 + m // R)[:, None] * H + h) * NQ + nq[None, :]
    c = tl.load(cl_ptr + off, mask=rm[:, None], other=1.0).to(tl.float32)
    s = tl.load(sl_ptr + off, mask=rm[:, None], other=0.0).to(tl.float32)
    return ca0[None, :] * c - sa0[None, :] * s, sa0[None, :] * c + ca0[None, :] * s


@triton.jit
def _load_rot2(X_ptr, b_ptr, cr, sr, h, tok0, m, rm, R: tl.constexpr, N: tl.constexpr, NQ: tl.constexpr):
    nq = tl.arange(0, NQ)
    xo = (tok0 * R + m)[:, None] * N + nq[None, :]
    bo = (h * R + m % R)[:, None] * N + nq[None, :]
    x0 = (tl.load(X_ptr + xo, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo).to(tl.bfloat16)).to(tl.float32)
    x2 = (tl.load(X_ptr + xo + 2 * NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + 2 * NQ).to(tl.bfloat16)).to(tl.float32)
    x1 = tl.load(X_ptr + xo + NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + NQ).to(tl.bfloat16)
    x3 = tl.load(X_ptr + xo + 3 * NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + 3 * NQ).to(tl.bfloat16)
    return (cr * x0 - sr * x2).to(tl.bfloat16), x1, (sr * x0 + cr * x2).to(tl.bfloat16), x3


@triton.jit
def _load_psiv(V_ptr, psi_ptr, h, tok0, m, rm, H, R: tl.constexpr, P: tl.constexpr, PB: tl.constexpr = 0, p0=0):
    if PB == 0:
        p = tl.arange(0, P)
    else:
        p = p0 + tl.arange(0, PB)
    v = tl.load(V_ptr + ((tok0 + m // R)[:, None] * H + h) * P + p[None, :], mask=rm[:, None], other=0.0)
    ps = tl.load(psi_ptr + (h * R + m % R)[:, None] * P + p[None, :])
    return (v.to(tl.float32) * ps).to(tl.bfloat16)


@triton.jit
def _ssd_chunk_state(
    K_ptr, V_ptr, kb_ptr, psi_ptr, cl_ptr, sl_ptr, dacs_ptr, ts_ptr, dtot_ptr,
    ch_start_ptr, ch_len_ptr, Sloc_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr, PB: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    p0 = tl.program_id(2) * PB
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    dlast = tl.load(dtot_ptr + c * H + h)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    nq = tl.arange(0, NQ)
    p = p0 + tl.arange(0, PB)
    ones_a = tl.full([NQ], 1.0, tl.float32)
    zero_a = tl.zeros([NQ], dtype=tl.float32)
    s0 = tl.zeros([NQ, PB], dtype=tl.float32)
    s1 = tl.zeros([NQ, PB], dtype=tl.float32)
    s2 = tl.zeros([NQ, PB], dtype=tl.float32)
    s3 = tl.zeros([NQ, PB], dtype=tl.float32)
    for j0 in range(0, cl, LB):
        rm = j0 + m // R < cl
        tok0 = cs + j0
        cr, sr = _rot_coef_tab(cl_ptr, sl_ptr, ones_a, zero_a, h, tok0, m, rm, H, R, NQ)
        k0, k1, k2, k3 = _load_rot2(K_ptr, kb_ptr, cr, sr, h, tok0, m, rm, R, N, NQ)
        tokr = tok0 + m // R
        w = tl.where(rm, tl.load(ts_ptr + h * T + tokr, mask=rm, other=0.0)
                     * tl.exp(dlast - tl.load(dacs_ptr + h * T + tokr, mask=rm, other=0.0)), 0.0)
        psiv = _load_psiv(V_ptr, psi_ptr, h, tok0, m, rm, H, R, P, PB, p0)
        s0 = tl.dot(tl.trans((k0.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv, s0)
        s1 = tl.dot(tl.trans((k1.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv, s1)
        s2 = tl.dot(tl.trans((k2.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv, s2)
        s3 = tl.dot(tl.trans((k3.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv, s3)
    base = Sloc_ptr + (c * H + h).to(tl.int64) * N * P
    tl.store(base + nq[:, None] * P + p[None, :], s0)
    tl.store(base + (NQ + nq[:, None]) * P + p[None, :], s1)
    tl.store(base + (2 * NQ + nq[:, None]) * P + p[None, :], s2)
    tl.store(base + (3 * NQ + nq[:, None]) * P + p[None, :], s3)


@triton.jit
def _ssd_abs_angles(thl_ptr, Ain_ptr, ch_start_ptr, ch_len_ptr, out_ptr, H, NQ: tl.constexpr, LS: tl.constexpr):
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    i = tl.arange(0, LS)
    nq = tl.arange(0, NQ)
    off = ((cs + i)[:, None] * H + h) * NQ + nq[None, :]
    a = tl.load(thl_ptr + off, mask=(i < cl)[:, None], other=0.0) + tl.load(Ain_ptr + (c * H + h) * NQ + nq)[None, :]
    tl.store(out_ptr + off, _mod2pi(a), mask=(i < cl)[:, None])


@triton.jit
def _ssd_state_pass_pool(
    Sloc_ptr, dtot_ptr, asum_ptr, seq_c0_ptr, seq_nc_ptr, seq_t0_ptr,
    slots_ptr, hinit_ptr,
    ang_pool, ssm_pool, k_pool, v_pool, s_ang, s_ssm, s_k, s_v,
    psi_ptr, DT_ptr, TRAP_ptr,
    SinT_ptr, Ain_ptr, fin_ang_ptr,
    T, H,
    HAS_INIT: tl.constexpr, R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr,
    PB: tl.constexpr,
):
    """_ssd_state_pass reading the entering state straight from the request's pool slot (zero when
    has_initial_state is false) and writing the final SSM state back into it; the final angle goes
    to fin_ang (written to the pool later: other programs of this sequence still read it)."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    p0 = tl.program_id(2) * PB
    c0 = tl.load(seq_c0_ptr + s)
    nc = tl.load(seq_nc_ptr + s)
    slot = tl.load(slots_ptr + s).to(tl.int64)
    nq = tl.arange(0, NQ)
    p = p0 + tl.arange(0, PB)
    sb = ssm_pool + slot * s_ssm + h * P * N  # (H, P, N) per slot
    if HAS_INIT:
        hi = tl.load(hinit_ptr + s) != 0
        st0 = tl.load(sb + p[None, :] * N + nq[:, None], mask=hi, other=0.0).to(tl.float32)
        st1 = tl.load(sb + p[None, :] * N + NQ + nq[:, None], mask=hi, other=0.0).to(tl.float32)
        st2 = tl.load(sb + p[None, :] * N + 2 * NQ + nq[:, None], mask=hi, other=0.0).to(tl.float32)
        st3 = tl.load(sb + p[None, :] * N + 3 * NQ + nq[:, None], mask=hi, other=0.0).to(tl.float32)
        t0 = tl.load(seq_t0_ptr + s)
        bscale = tl.load(DT_ptr + h * T + t0) * tl.sigmoid(-tl.load(TRAP_ptr + h * T + t0).to(tl.float32))
        bscale = tl.where(hi, bscale, 0.0)
        vv = tl.load(v_pool + slot * s_v + h * P + p, mask=hi, other=0.0).to(tl.float32)
        for r in tl.static_range(R):
            ps = tl.load(psi_ptr + (h * R + r) * P + p)
            kb = k_pool + slot * s_k + (r * H + h) * N
            pv = vv * ps * bscale
            st0 += tl.load(kb + nq, mask=hi, other=0.0).to(tl.float32)[:, None] * pv[None, :]
            st1 += tl.load(kb + NQ + nq, mask=hi, other=0.0).to(tl.float32)[:, None] * pv[None, :]
            st2 += tl.load(kb + 2 * NQ + nq, mask=hi, other=0.0).to(tl.float32)[:, None] * pv[None, :]
            st3 += tl.load(kb + 3 * NQ + nq, mask=hi, other=0.0).to(tl.float32)[:, None] * pv[None, :]
        th = tl.load(ang_pool + slot * s_ang + h * NQ + nq, mask=hi, other=0.0)
    else:
        st0 = tl.zeros([NQ, PB], dtype=tl.float32)
        st1 = tl.zeros([NQ, PB], dtype=tl.float32)
        st2 = tl.zeros([NQ, PB], dtype=tl.float32)
        st3 = tl.zeros([NQ, PB], dtype=tl.float32)
        th = tl.zeros([NQ], dtype=tl.float32)
    for j in range(0, nc):
        c = c0 + j
        ob = SinT_ptr + (c * H + h).to(tl.int64) * N * P
        tl.store(ob + p[None, :] * N + nq[:, None], st0)
        tl.store(ob + p[None, :] * N + NQ + nq[:, None], st1)
        tl.store(ob + p[None, :] * N + 2 * NQ + nq[:, None], st2)
        tl.store(ob + p[None, :] * N + 3 * NQ + nq[:, None], st3)
        tl.store(Ain_ptr + (c * H + h) * NQ + nq, th)
        lb = Sloc_ptr + (c * H + h).to(tl.int64) * N * P
        l0 = tl.load(lb + nq[:, None] * P + p[None, :])
        l1 = tl.load(lb + (NQ + nq[:, None]) * P + p[None, :])
        l2 = tl.load(lb + (2 * NQ + nq[:, None]) * P + p[None, :])
        l3 = tl.load(lb + (3 * NQ + nq[:, None]) * P + p[None, :])
        cr = tl.cos(th)[:, None]
        sr = tl.sin(th)[:, None]
        dec = tl.exp(tl.load(dtot_ptr + c * H + h))
        st0 = dec * st0 + (cr * l0 - sr * l2)
        st2 = dec * st2 + (sr * l0 + cr * l2)
        st1 = dec * st1 + l1
        st3 = dec * st3 + l3
        th = _mod2pi(th + tl.load(asum_ptr + (c * H + h) * NQ + nq))
    dt_ = ssm_pool.dtype.element_ty
    tl.store(sb + p[None, :] * N + nq[:, None], st0.to(dt_))
    tl.store(sb + p[None, :] * N + NQ + nq[:, None], st1.to(dt_))
    tl.store(sb + p[None, :] * N + 2 * NQ + nq[:, None], st2.to(dt_))
    tl.store(sb + p[None, :] * N + 3 * NQ + nq[:, None], st3.to(dt_))
    if p0 == 0:
        tl.store(fin_ang_ptr + (s * H + h) * NQ + nq, th)


@triton.jit
def _ssd_pool_tail(
    fin_ang_ptr, kfin_ptr, last_rows_ptr, X_ptr, ends_ptr, slots_ptr,
    ang_pool, k_pool, v_pool, s_ang, s_k, s_v,
    H, R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr,
):
    """Per (sequence, head): final angle, final rotated K (last virtual chunk) and the last token's
    x into the request's pool slot."""
    s = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(slots_ptr + s).to(tl.int64)
    nq = tl.arange(0, NQ)
    tl.store(ang_pool + slot * s_ang + h * NQ + nq, tl.load(fin_ang_ptr + (s * H + h) * NQ + nq))
    lr = tl.load(last_rows_ptr + s).to(tl.int64)
    n = tl.arange(0, N)
    for r in tl.static_range(R):
        tl.store(k_pool + slot * s_k + (r * H + h) * N + n, tl.load(kfin_ptr + ((lr * R + r) * H + h) * N + n))
    p = tl.arange(0, P)
    e = tl.load(ends_ptr + s).to(tl.int64)
    tl.store(v_pool + slot * s_v + h * P + p, tl.load(X_ptr + ((e - 1) * H + h) * P + p))


_LAYOUT_CACHE: dict = {}


def _layout_meta(cu: tuple, chunk_tokens: int, dev):
    """Chunk layout of a batch as one device int32 tensor, cached: every Mamba layer of a step
    (and steps with the same layout) reuse it instead of rebuilding and uploading it."""
    key = (cu, chunk_tokens, dev)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    cu_l = list(cu)
    starts, lens, sends, lasts, seq_c0, seq_nc = _chunk_layout(cu_l, chunk_tokens)
    NC, NS = len(starts), len(cu_l) - 1
    last_rows = [seq_c0[s] + seq_nc[s] - 1 for s in range(NS)]
    meta = torch.tensor(starts + lens + sends + lasts + seq_c0 + seq_nc + cu_l[:-1]
                        + starts + [cu_l[-1]] + last_rows + cu_l[1:],
                        dtype=torch.int32).to(dev, non_blocking=True)
    if len(_LAYOUT_CACHE) > 64:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = (NC, NS, meta)
    return NC, NS, meta


_ZERO_KV: dict = {}


def _zero_kv(nc, R, H, N, P, dev, dtype):
    """Zero boundary K/V for the TileLang pass (read-only, so cached and grown on demand)."""
    key = (R, H, N, P, dev, dtype)
    zk, zv = _ZERO_KV.get(key, (None, None))
    if zk is None or zk.shape[0] < nc:
        n = max(nc, 64)
        zk = torch.zeros(n, R, H, N, device=dev, dtype=dtype)
        zv = torch.zeros(n, H, P, device=dev, dtype=dtype)
        _ZERO_KV[key] = (zk, zv)
    return zk[:nc], zv[:nc]


@torch.no_grad()
def mamba3_mimo_hybrid_pools(
    Q, K, V, Z, angles, DT, ADT, Trap, q_bias, k_bias, psi, phi, zeta, D,
    cu_seqlens_cpu: list[int],
    pools, slots: torch.Tensor, has_init: torch.Tensor | None, out: torch.Tensor,
    chunk_tokens: int = 128,
    tl_chunk: int = 16,
    tune: dict | None = None,
):
    """:func:`mamba3_mimo_hybrid` reading the entering states from and writing the final states to
    the request pool slots in place (``has_init`` None: every sequence starts from zero)."""
    from .mimo_fwd_varlen import mamba_mimo_forward_varlen
    from .segsum import compute_dacs_segsum_triton_varlen
    angle_pool, ssm_pool, k_pool, v_pool = pools
    T, R, N = Q.shape
    H, P = V.shape[1], V.shape[2]
    NQ = N // 4
    dev = Q.device
    NC, NS, meta = _layout_meta(tuple(cu_seqlens_cpu), chunk_tokens, dev)
    o = 0
    def take(k):
        nonlocal o
        v = meta[o:o + k]
        o += k
        return v
    m_start, m_len, m_send, m_last = take(NC), take(NC), take(NC), take(NC)
    s_c0, s_nc, s_t0 = take(NS), take(NS), take(NS)
    v_cu, m_lrow, m_end = take(NC + 1), take(NS), take(NS)
    f32 = dict(device=dev, dtype=torch.float32)
    dacs, ts, gam = torch.empty(H, T, **f32), torch.empty(H, T, **f32), torch.empty(H, T, **f32)
    cs_cuda = _CS_CUDA and _scan_cuda_ok(dev)
    cl_t = sl_t = None  # fp16 cos / sin tables, only for the Triton chunk-state kernels
    if not cs_cuda:
        cl_t = torch.empty(T, H, NQ, device=dev, dtype=torch.float16)
        sl_t = torch.empty(T, H, NQ, device=dev, dtype=torch.float16)
    thl = torch.empty(T, H, NQ, **f32)
    dtot, asum = torch.empty(NC, H, **f32), torch.empty(NC, H, NQ, **f32)
    Sloc = torch.empty(NC, H, N, P, **f32)
    SinT = torch.empty(NC, H, P, N, **f32)
    Ain = torch.empty(NC, H, NQ, **f32)
    fin_ang = torch.empty(NS, H, NQ, **f32)
    tune = tune or {}
    DA_CS = torch.empty(1, H, T, **f32)
    DA_CS_REV = torch.empty(1, H, T, **f32)
    _ssd_prep[(NC, H)](angles, DT, ADT, Trap, m_start, m_len, m_send, dacs, ts, gam, thl, dtot, asum, cl_t, sl_t,
                       T, H, NQ=NQ, LS=chunk_tokens, dcs16_ptr=DA_CS, dcsr16_ptr=DA_CS_REV, G16=tl_chunk,
                       WRITE_CS=not cs_cuda, num_warps=tune.get("prep_nw", 8))
    if cs_cuda:
        _scan_ext().chunk_state(K, V, k_bias, psi, thl, dacs, ts, dtot, m_start, m_len, Sloc, NC)
    else:
        _ssd_chunk_state[(NC, H, 1)](K, V, k_bias, psi, cl_t, sl_t, dacs, ts, dtot, m_start, m_len, Sloc, T, H,
                                     R=R, N=N, P=P, NQ=NQ, LB=16, PB=P, num_warps=8)
    pb = tune.get("pass_pb", 16)
    _ssd_state_pass_pool[(H, NS, P // pb)](
        Sloc, dtot, asum, s_c0, s_nc, s_t0, slots, has_init if has_init is not None else slots,
        angle_pool, ssm_pool, k_pool, v_pool,
        angle_pool.stride(0), ssm_pool.stride(0), k_pool.stride(0), v_pool.stride(0),
        psi, DT, Trap, SinT, Ain, fin_ang, T, H,
        HAS_INIT=has_init is not None, R=R, N=N, P=P, NQ=NQ, PB=pb, num_warps=tune.get("pass_nw", 2),
    )
    if _scan_cuda_ok(dev):
        k_fin = torch.empty(NC, R, H, N, device=dev, dtype=Q.dtype)
        _scan_ext().scan(Q, K, V, Z, q_bias, k_bias, psi, phi, zeta, D, thl, Ain, DA_CS, DA_CS_REV, ts, gam,
                         m_start, m_len, m_last, SinT, out, k_fin, NC)
    elif _TL_V2:
        zk, zv = _zero_kv(NC, R, H, N, P, dev, Q.dtype)
        # v2: chunk-local phases + per-chunk offset in-kernel, segsum inline, no qk_dot GEMM
        from .mimo_fwd_varlen_v2 import mamba_mimo_forward_varlen_v2
        _, _, k_fin = mamba_mimo_forward_varlen_v2(
            Q.view(1, T, R, 1, N), K.view(1, T, R, 1, N), V.view(1, T, H, P), q_bias, k_bias, psi, phi,
            Z.view(1, T, H, P), D, zeta, thl.view(1, T, H, NQ), DA_CS, DA_CS_REV, DT.view(1, H, T), Trap.view(1, H, T),
            Ain, tl_chunk, 4, Q.dtype, cu_seqlens=v_cu, return_state=True, initial_states=(SinT, zk, zv),
            threads=128, num_stages=0, out=out.view(1, T, H, P),
        )
    else:
        zk, zv = _zero_kv(NC, R, H, N, P, dev, Q.dtype)
        _, _, Segsum = compute_dacs_segsum_triton_varlen(ADT.view(1, H, T), tl_chunk, cu_seqlens=v_cu)
        ang_abs = torch.empty(T, H, NQ, **f32)
        _ssd_abs_angles[(NC, H)](thl, Ain, m_start, m_len, ang_abs, H, NQ=NQ, LS=chunk_tokens, num_warps=tune.get("abs_nw", 4))
        _, _, k_fin = mamba_mimo_forward_varlen(
            Q.view(1, T, R, 1, N), K.view(1, T, R, 1, N), V.view(1, T, H, P), q_bias, k_bias, psi, phi,
            Z.view(1, T, H, P), D, zeta, ang_abs.view(1, T, H, NQ), DA_CS, DA_CS_REV, DT.view(1, H, T), Trap.view(1, H, T),
            Segsum, tl_chunk, 4, Q.dtype, cu_seqlens=v_cu, return_state=True, initial_states=(SinT, zk, zv),
            threads=128, num_stages=0, out=out.view(1, T, H, P),
        )
    _ssd_pool_tail[(NS, H)](fin_ang, k_fin, m_lrow, V, m_end, slots, angle_pool, k_pool, v_pool,
                            angle_pool.stride(0), k_pool.stride(0), v_pool.stride(0),
                            H, R=R, N=N, P=P, NQ=NQ, num_warps=1)


def _chunk_layout(cu: list[int], ls: int):
    starts, lens, sends, lasts, seq_c0, seq_nc = [], [], [], [], [], []
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        seq_c0.append(len(starts))
        n = 0
        for x in range(a, b, ls):
            starts.append(x)
            lens.append(min(ls, b - x))
            sends.append(b)
            lasts.append(s if x + ls >= b else -1)
            n += 1
        seq_nc.append(n)
    return starts, lens, sends, lasts, seq_c0, seq_nc

