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

Kernels (chunks of LS tokens never cross a sequence; rows of a tile are token-major t*R + r):
  0. ``_ssd_prep``        (chunk, head): per-token chunk-local log-decay cumsum, trapezoid scales,
     chunk-local phase cumsum; chunk totals.
  1. ``_ssd_chunk_state`` (chunk, head): local chunk state with chunk-local phases.
  2. ``_ssd_state_pass``  (head, sequence): serial over chunks, S_in[c] = state entering chunk c;
     a chunk's local state is rotated by the phase prefix (rot(a + b) = rot(a) rot(b)).
  3. ``_ssd_chunk_scan``  (chunk x token tile, head): outputs of one tile from S_in and the
     earlier tiles of the chunk.
Rotary columns are four NQ = N/4 wide groups: g0/g2 rotate together, g1/g3 do not.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


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
    tl.store(ts_ptr + h * T + t, gam + dtn * tl.sigmoid(-trn), mask=tm)
    tl.store(gam_ptr + h * T + t, gam, mask=tm)
    tl.store(dtot_ptr + c * H + h, tl.sum(adt, axis=0))
    nq = tl.arange(0, NQ)
    a = tl.load(ang_ptr + t[:, None] * NQ + nq[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    inc = _tanh(a) * 3.141592653589793 * dtv[:, None]
    thl = tl.cumsum(inc, axis=0)
    tl.store(thl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], thl, mask=tm[:, None])
    tl.store(cl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], tl.cos(thl).to(tl.float16), mask=tm[:, None])
    tl.store(sl_ptr + (t[:, None] * H + h) * NQ + nq[None, :], tl.sin(thl).to(tl.float16), mask=tm[:, None])
    tl.store(asum_ptr + (c * H + h) * NQ + nq, tl.sum(inc, axis=0))


@triton.jit
def _load_rot(X_ptr, b_ptr, thl_ptr, a0, h, tok0, m, rm, T, H,
              R: tl.constexpr, N: tl.constexpr, NQ: tl.constexpr):
    """Rows tok0*R + m of a (T*R, N) bf16 matrix, + bias[h, r] (in bf16), rotated by
    theta = a0 + thl[token, h]; four bf16 [LBR, NQ] column groups."""
    nq = tl.arange(0, NQ)
    tok = tok0 + m // R
    r = m % R
    xo = (tok0 * R + m)[:, None] * N + nq[None, :]
    bo = (h * R + r)[:, None] * N + nq[None, :]
    th = a0[None, :] + tl.load(thl_ptr + (tok[:, None] * H + h) * NQ + nq[None, :], mask=rm[:, None], other=0.0)
    cr = tl.cos(th)
    sr = tl.sin(th)
    x0 = (tl.load(X_ptr + xo, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo).to(tl.bfloat16)).to(tl.float32)
    x2 = (tl.load(X_ptr + xo + 2 * NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + 2 * NQ).to(tl.bfloat16)).to(tl.float32)
    x1 = tl.load(X_ptr + xo + NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + NQ).to(tl.bfloat16)
    x3 = tl.load(X_ptr + xo + 3 * NQ, mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + 3 * NQ).to(tl.bfloat16)
    y0 = (cr * x0 - sr * x2).to(tl.bfloat16)
    y2 = (sr * x0 + cr * x2).to(tl.bfloat16)
    return y0, x1, y2, x3


@triton.jit
def _rot_coef(thl_ptr, a0, h, tok0, m, rm, H, R: tl.constexpr, NQ: tl.constexpr):
    nq = tl.arange(0, NQ)
    th = a0[None, :] + tl.load(thl_ptr + ((tok0 + m // R)[:, None] * H + h) * NQ + nq[None, :], mask=rm[:, None], other=0.0)
    return tl.cos(th), tl.sin(th)


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
def _ssd_state_pass(
    Sloc_ptr, dtot_ptr, asum_ptr, seq_c0_ptr, seq_nc_ptr, seq_t0_ptr,
    in_ssm_ptr, in_k_ptr, in_v_ptr, in_ang_ptr, psi_ptr, DT_ptr, TRAP_ptr,
    Sin_ptr, Ain_ptr, fin_ssm_ptr, fin_ang_ptr,
    T, H,
    HAS_INIT: tl.constexpr, R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr,
    SIN_T: tl.constexpr = False, PB: tl.constexpr = 64,
):
    h = tl.program_id(0)
    s = tl.program_id(1)
    p0 = tl.program_id(2) * PB
    c0 = tl.load(seq_c0_ptr + s)
    nc = tl.load(seq_nc_ptr + s)
    nq = tl.arange(0, NQ)
    p = p0 + tl.arange(0, PB)
    if HAS_INIT:
        # INIT_SSM is (NS, H, P, N); fold the previous chunk's last token with weight DT[t0] sigmoid(-Trap[t0])
        ib = in_ssm_ptr + (s * H + h).to(tl.int64) * P * N
        st0 = tl.load(ib + p[None, :] * N + nq[:, None])
        st1 = tl.load(ib + p[None, :] * N + NQ + nq[:, None])
        st2 = tl.load(ib + p[None, :] * N + 2 * NQ + nq[:, None])
        st3 = tl.load(ib + p[None, :] * N + 3 * NQ + nq[:, None])
        t0 = tl.load(seq_t0_ptr + s)
        bscale = tl.load(DT_ptr + h * T + t0) * tl.sigmoid(-tl.load(TRAP_ptr + h * T + t0).to(tl.float32))
        vv = tl.load(in_v_ptr + (s * H + h) * P + p).to(tl.float32)
        for r in tl.static_range(R):
            ps = tl.load(psi_ptr + (h * R + r) * P + p)
            kb = in_k_ptr + ((s * R + r) * H + h) * N
            pv = vv * ps * bscale
            st0 += tl.load(kb + nq).to(tl.float32)[:, None] * pv[None, :]
            st1 += tl.load(kb + NQ + nq).to(tl.float32)[:, None] * pv[None, :]
            st2 += tl.load(kb + 2 * NQ + nq).to(tl.float32)[:, None] * pv[None, :]
            st3 += tl.load(kb + 3 * NQ + nq).to(tl.float32)[:, None] * pv[None, :]
        th = tl.load(in_ang_ptr + (s * H + h) * NQ + nq)
    else:
        st0 = tl.zeros([NQ, PB], dtype=tl.float32)
        st1 = tl.zeros([NQ, PB], dtype=tl.float32)
        st2 = tl.zeros([NQ, PB], dtype=tl.float32)
        st3 = tl.zeros([NQ, PB], dtype=tl.float32)
        th = tl.zeros([NQ], dtype=tl.float32)
    for j in range(0, nc):
        c = c0 + j
        ob = Sin_ptr + (c * H + h).to(tl.int64) * N * P
        if SIN_T:  # (NC, H, P, N) fp32: the TileLang kernel's INIT_SSM_STATE layout
            tl.store(ob + p[None, :] * N + nq[:, None], st0)
            tl.store(ob + p[None, :] * N + NQ + nq[:, None], st1)
            tl.store(ob + p[None, :] * N + 2 * NQ + nq[:, None], st2)
            tl.store(ob + p[None, :] * N + 3 * NQ + nq[:, None], st3)
        else:
            tl.store(ob + nq[:, None] * P + p[None, :], st0.to(tl.bfloat16))
            tl.store(ob + (NQ + nq[:, None]) * P + p[None, :], st1.to(tl.bfloat16))
            tl.store(ob + (2 * NQ + nq[:, None]) * P + p[None, :], st2.to(tl.bfloat16))
            tl.store(ob + (3 * NQ + nq[:, None]) * P + p[None, :], st3.to(tl.bfloat16))
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
    fb = fin_ssm_ptr + (s * H + h).to(tl.int64) * P * N  # (NS, H, P, N)
    tl.store(fb + p[None, :] * N + nq[:, None], st0)
    tl.store(fb + p[None, :] * N + NQ + nq[:, None], st1)
    tl.store(fb + p[None, :] * N + 2 * NQ + nq[:, None], st2)
    tl.store(fb + p[None, :] * N + 3 * NQ + nq[:, None], st3)
    tl.store(fin_ang_ptr + (s * H + h) * NQ + nq, th)


@triton.jit
def _ssd_chunk_scan(
    Q_ptr, K_ptr, V_ptr, Z_ptr, qb_ptr, kb_ptr, psi_ptr, phi_ptr, zeta_ptr, D_ptr,
    thl_ptr, dacs_ptr, ts_ptr, gam_ptr,
    ch_start_ptr, ch_len_ptr, ch_last_ptr,
    Sin_ptr, Ain_ptr, O_ptr, fin_k_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr, NT: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    c = pid // NT
    ti = pid % NT
    cl = tl.load(ch_len_ptr + c)
    if ti * LB >= cl:
        return
    cs = tl.load(ch_start_ptr + c)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    nq = tl.arange(0, NQ)
    p = tl.arange(0, P)
    tok0 = cs + ti * LB
    rm = ti * LB + m // R < cl
    tokr = tok0 + m // R
    a0 = tl.load(Ain_ptr + (c * H + h) * NQ + nq)
    q0, q1, q2, q3 = _load_rot(Q_ptr, qb_ptr, thl_ptr, a0, h, tok0, m, rm, T, H, R, N, NQ)
    dacs_i = tl.load(dacs_ptr + h * T + tokr, mask=rm, other=0.0)
    # inter-chunk: exp(dacs_t) qhat . S_in  (S_in in bf16, as the original kernel)
    sb = Sin_ptr + (c * H + h).to(tl.int64) * N * P
    o = tl.dot(q0, tl.load(sb + nq[:, None] * P + p[None, :]))
    o = tl.dot(q1, tl.load(sb + (NQ + nq[:, None]) * P + p[None, :]), o)
    o = tl.dot(q2, tl.load(sb + (2 * NQ + nq[:, None]) * P + p[None, :]), o)
    o = tl.dot(q3, tl.load(sb + (3 * NQ + nq[:, None]) * P + p[None, :]), o)
    o = o * tl.exp(dacs_i)[:, None]
    # intra-chunk: earlier tiles (strictly causal, trapezoid scale), own tile adds the diagonal (gamma)
    ti_r = m // R
    for tj in range(0, ti + 1):
        tokj = cs + tj * LB
        rmj = tj * LB + m // R < cl
        k0, k1, k2, k3 = _load_rot(K_ptr, kb_ptr, thl_ptr, a0, h, tokj, m, rmj, T, H, R, N, NQ)
        g = tl.dot(q0, tl.trans(k0))
        g = tl.dot(q1, tl.trans(k1), g)
        g = tl.dot(q2, tl.trans(k2), g)
        g = tl.dot(q3, tl.trans(k3), g)
        tokjr = tokj + m // R
        dacs_j = tl.load(dacs_ptr + h * T + tokjr, mask=rmj, other=0.0)
        ts_j = tl.load(ts_ptr + h * T + tokjr, mask=rmj, other=0.0)
        if tj < ti:
            coef = ts_j[None, :] * tl.exp(dacs_i[:, None] - dacs_j[None, :])
        else:
            gam_i = tl.load(gam_ptr + h * T + tokr, mask=rm, other=0.0)
            coef = tl.where(ti_r[:, None] > ti_r[None, :],
                            ts_j[None, :] * tl.exp(tl.minimum(dacs_i[:, None] - dacs_j[None, :], 0.0)),
                            tl.where(ti_r[:, None] == ti_r[None, :], gam_i[:, None], 0.0))
        coef = tl.where(rmj[None, :], coef, 0.0)
        psiv = _load_psiv(V_ptr, psi_ptr, h, tokj, m, rmj, H, R, P)
        o = tl.dot((g * coef).to(tl.bfloat16), psiv, o)
        if tj == ti:
            o += tl.load(D_ptr + h) * psiv.to(tl.float32)
            last_seq = tl.load(ch_last_ptr + c)
            if last_seq >= 0:
                if (ti + 1) * LB >= cl:
                    lastrow = (cl - 1 - ti * LB) * R
                    sel = (m >= lastrow) & (m < lastrow + R)
                    fb = fin_k_ptr + ((last_seq * R + (m % R)) * H + h)[:, None].to(tl.int64) * N
                    tl.store(fb + nq[None, :], k0, mask=sel[:, None])
                    tl.store(fb + NQ + nq[None, :], k1, mask=sel[:, None])
                    tl.store(fb + 2 * NQ + nq[None, :], k2, mask=sel[:, None])
                    tl.store(fb + 3 * NQ + nq[None, :], k3, mask=sel[:, None])
    # gate + MIMO down-projection (sum over ranks)
    z = tl.load(Z_ptr + (tokr[:, None] * H + h) * P + p[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    zg = z * tl.load(zeta_ptr + (h * R + m % R)[:, None] * P + p[None, :])
    o = o * tl.load(phi_ptr + (h * R + m % R)[:, None] * P + p[None, :]) * (zg * tl.sigmoid(zg))
    out = tl.sum(tl.reshape(o, (LB, R, P)), axis=1)
    i = tl.arange(0, LB)
    tm = ti * LB + i < cl
    tl.store(O_ptr + ((tok0 + i)[:, None] * H + h) * P + p[None, :], out.to(O_ptr.dtype.element_ty), mask=tm[:, None])


@triton.jit
def _ssd_chunk_scan_seq(
    Q_ptr, K_ptr, V_ptr, Z_ptr, qb_ptr, kb_ptr, psi_ptr, phi_ptr, zeta_ptr, D_ptr,
    cl_ptr, sl_ptr, dacs_ptr, ts_ptr, gam_ptr,
    ch_start_ptr, ch_len_ptr, ch_last_ptr,
    Sin_ptr, Ain_ptr, O_ptr, fin_k_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr, PB: tl.constexpr, ABL: tl.constexpr = 0,
):
    """One program per (chunk, head, P tile): walks the chunk's LB-token sub-blocks serially from S_in,
    the state held in registers (four [NQ, P] fp32 groups)."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    p0 = tl.program_id(2) * PB
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    last_seq = tl.load(ch_last_ptr + c)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    nq = tl.arange(0, NQ)
    p = p0 + tl.arange(0, PB)
    ti_r = m // R
    sb_ = Sin_ptr + (c * H + h).to(tl.int64) * N * P
    st0 = tl.load(sb_ + nq[:, None] * P + p[None, :]).to(tl.float32)
    st1 = tl.load(sb_ + (NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    st2 = tl.load(sb_ + (2 * NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    st3 = tl.load(sb_ + (3 * NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    a0 = tl.load(Ain_ptr + (c * H + h) * NQ + nq)
    ca0 = tl.cos(a0)
    sa0 = tl.sin(a0)
    dskip = tl.load(D_ptr + h)
    base = 0.0
    for j0 in range(0, cl, LB):
        tok0 = cs + j0
        rm = j0 + ti_r < cl
        tokr = tok0 + ti_r
        if ABL & 1:
            cr = tl.full([LBR, NQ], 1.0, tl.float32)
            sr = tl.zeros([LBR, NQ], dtype=tl.float32)
        else:
            cr, sr = _rot_coef_tab(cl_ptr, sl_ptr, ca0, sa0, h, tok0, m, rm, H, R, NQ)
        q0, q1, q2, q3 = _load_rot2(Q_ptr, qb_ptr, cr, sr, h, tok0, m, rm, R, N, NQ)
        k0, k1, k2, k3 = _load_rot2(K_ptr, kb_ptr, cr, sr, h, tok0, m, rm, R, N, NQ)
        # sub-block-relative log-decay cumsum
        dacs = tl.load(dacs_ptr + h * T + tokr, mask=rm, other=0.0) - base
        last_i = tl.minimum(cl - j0, LB) - 1
        dlast = tl.load(dacs_ptr + h * T + tok0 + last_i) - base
        base = base + dlast
        # inter: exp(dacs) qhat . S (state in bf16 for the MMA, as the original kernel)
        if ABL & 8:
            o = tl.zeros([LBR, PB], dtype=tl.float32)
        else:
            o = tl.dot(q0, st0.to(tl.bfloat16))
            o = tl.dot(q1, st1.to(tl.bfloat16), o)
            o = tl.dot(q2, st2.to(tl.bfloat16), o)
            o = tl.dot(q3, st3.to(tl.bfloat16), o)
            o = o * tl.exp(dacs)[:, None]
        ts_r = tl.load(ts_ptr + h * T + tokr, mask=rm, other=0.0)
        gam_r = tl.load(gam_ptr + h * T + tokr, mask=rm, other=0.0)
        psiv = _load_psiv(V_ptr, psi_ptr, h, tok0, m, rm, H, R, P, PB, p0)
        if not (ABL & 2):
            g = tl.dot(q0, tl.trans(k0))
            g = tl.dot(q1, tl.trans(k1), g)
            g = tl.dot(q2, tl.trans(k2), g)
            g = tl.dot(q3, tl.trans(k3), g)
            coef = tl.where(ti_r[:, None] > ti_r[None, :],
                            ts_r[None, :] * tl.exp(tl.minimum(dacs[:, None] - dacs[None, :], 0.0)),
                            tl.where(ti_r[:, None] == ti_r[None, :], gam_r[:, None], 0.0))
            coef = tl.where(rm[None, :], coef, 0.0)
            o = tl.dot((g * coef).to(tl.bfloat16), psiv, o)
        o += dskip * psiv.to(tl.float32)
        z = tl.load(Z_ptr + (tokr[:, None] * H + h) * P + p[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        zg = z * tl.load(zeta_ptr + (h * R + m % R)[:, None] * P + p[None, :])
        o = o * tl.load(phi_ptr + (h * R + m % R)[:, None] * P + p[None, :]) * (zg * tl.sigmoid(zg))
        out = tl.sum(tl.reshape(o, (LB, R, PB)), axis=1)
        i = tl.arange(0, LB)
        tl.store(O_ptr + ((tok0 + i)[:, None] * H + h) * P + p[None, :], out.to(O_ptr.dtype.element_ty),
                 mask=(j0 + i < cl)[:, None])
        if (last_seq >= 0) & (p0 == 0):
            if j0 + LB >= cl:
                lastrow = (cl - 1 - j0) * R
                sel = (m >= lastrow) & (m < lastrow + R)
                fb = fin_k_ptr + ((last_seq * R + (m % R)) * H + h)[:, None].to(tl.int64) * N
                tl.store(fb + nq[None, :], k0, mask=sel[:, None])
                tl.store(fb + NQ + nq[None, :], k1, mask=sel[:, None])
                tl.store(fb + 2 * NQ + nq[None, :], k2, mask=sel[:, None])
                tl.store(fb + 3 * NQ + nq[None, :], k3, mask=sel[:, None])
        w = tl.where(rm, ts_r * tl.exp(dlast - dacs), 0.0)
        dec = tl.exp(dlast)
        if ABL & 4:
            w = w * 0.0
        st0 = dec * st0 + tl.dot(tl.trans((k0.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st1 = dec * st1 + tl.dot(tl.trans((k1.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st2 = dec * st2 + tl.dot(tl.trans((k2.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st3 = dec * st3 + tl.dot(tl.trans((k3.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)


@triton.jit
def _rot_full(X_ptr, b_ptr, thl_ptr, a0_ptr, h, tok0, m, rm, H,
              R: tl.constexpr, N: tl.constexpr, NQ: tl.constexpr, HAS_A0: tl.constexpr):
    """Rows tok0*R + m of a (T*R, N) bf16 matrix, + bias[h, r] (bf16 add), rotated on the
    (n, n + N/2), n < NQ pairs by theta = a0 + thl[token, h]: full-width [LBR, N] bf16 tile,
    xhat = x * C + x[partner] * S (partner(n) = n +- N/2 on rotated columns, else n)."""
    n = tl.arange(0, N)
    rot = (n < NQ) | ((n >= N // 2) & (n < N // 2 + NQ))
    partner = tl.where(n < NQ, n + N // 2, tl.where(rot, n - N // 2, n))
    aidx = n % NQ
    sgn = tl.where(n < NQ, -1.0, 1.0)
    tok = tok0 + m // R
    row = (tok0 * R + m)[:, None] * N
    bo = (h * R + m % R)[:, None] * N
    x = tl.load(X_ptr + row + n[None, :], mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + n[None, :]).to(tl.bfloat16)
    xp = tl.load(X_ptr + row + partner[None, :], mask=rm[:, None], other=0.0) + tl.load(b_ptr + bo + partner[None, :]).to(tl.bfloat16)
    th = tl.load(thl_ptr + (tok[:, None] * H + h) * NQ + aidx[None, :], mask=rm[:, None], other=0.0)
    if HAS_A0:
        th += tl.load(a0_ptr + aidx)[None, :]
    cr = tl.where(rot[None, :], tl.cos(th), 1.0)
    sr = tl.where(rot[None, :], tl.sin(th) * sgn[None, :], 0.0)
    return (x.to(tl.float32) * cr + xp.to(tl.float32) * sr).to(tl.bfloat16)


@triton.jit
def _ssd_chunk_state_full(
    K_ptr, V_ptr, kb_ptr, psi_ptr, thl_ptr, dacs_ptr, ts_ptr, dtot_ptr,
    ch_start_ptr, ch_len_ptr, Sloc_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr,
):
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    dlast = tl.load(dtot_ptr + c * H + h)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    n = tl.arange(0, N)
    p = tl.arange(0, P)
    s = tl.zeros([N, P], dtype=tl.float32)
    for j0 in range(0, cl, LB):
        rm = j0 + m // R < cl
        tok0 = cs + j0
        k = _rot_full(K_ptr, kb_ptr, thl_ptr, thl_ptr, h, tok0, m, rm, H, R, N, NQ, False)
        tokr = tok0 + m // R
        w = tl.where(rm, tl.load(ts_ptr + h * T + tokr, mask=rm, other=0.0)
                     * tl.exp(dlast - tl.load(dacs_ptr + h * T + tokr, mask=rm, other=0.0)), 0.0)
        psiv = _load_psiv(V_ptr, psi_ptr, h, tok0, m, rm, H, R, P)
        s = tl.dot(tl.trans((k.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv, s)
    tl.store(Sloc_ptr + (c * H + h).to(tl.int64) * N * P + n[:, None] * P + p[None, :], s)


@triton.jit
def _ssd_chunk_scan_full(
    Q_ptr, K_ptr, V_ptr, Z_ptr, qb_ptr, kb_ptr, psi_ptr, phi_ptr, zeta_ptr, D_ptr,
    thl_ptr, dacs_ptr, ts_ptr, gam_ptr,
    ch_start_ptr, ch_len_ptr, ch_last_ptr,
    Sin_ptr, Ain_ptr, O_ptr, fin_k_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr,
):
    """One program per (chunk, head): the chunk's LB-token sub-blocks serially from S_in
    (state [N, P] fp32 in registers), full-width rotated q/k tiles."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    last_seq = tl.load(ch_last_ptr + c)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    n = tl.arange(0, N)
    p = tl.arange(0, P)
    ti_r = m // R
    st = tl.load(Sin_ptr + (c * H + h).to(tl.int64) * N * P + n[:, None] * P + p[None, :]).to(tl.float32)
    a0_ptr = Ain_ptr + (c * H + h) * NQ
    dskip = tl.load(D_ptr + h)
    base = 0.0
    for j0 in range(0, cl, LB):
        tok0 = cs + j0
        rm = j0 + ti_r < cl
        tokr = tok0 + ti_r
        q = _rot_full(Q_ptr, qb_ptr, thl_ptr, a0_ptr, h, tok0, m, rm, H, R, N, NQ, True)
        k = _rot_full(K_ptr, kb_ptr, thl_ptr, a0_ptr, h, tok0, m, rm, H, R, N, NQ, True)
        dacs = tl.load(dacs_ptr + h * T + tokr, mask=rm, other=0.0) - base
        last_i = tl.minimum(cl - j0, LB) - 1
        dlast = tl.load(dacs_ptr + h * T + tok0 + last_i) - base
        base = base + dlast
        o = tl.dot(q, st.to(tl.bfloat16)) * tl.exp(dacs)[:, None]
        g = tl.dot(q, tl.trans(k))
        ts_r = tl.load(ts_ptr + h * T + tokr, mask=rm, other=0.0)
        gam_r = tl.load(gam_ptr + h * T + tokr, mask=rm, other=0.0)
        coef = tl.where(ti_r[:, None] > ti_r[None, :],
                        ts_r[None, :] * tl.exp(tl.minimum(dacs[:, None] - dacs[None, :], 0.0)),
                        tl.where(ti_r[:, None] == ti_r[None, :], gam_r[:, None], 0.0))
        coef = tl.where(rm[None, :], coef, 0.0)
        psiv = _load_psiv(V_ptr, psi_ptr, h, tok0, m, rm, H, R, P)
        o = tl.dot((g * coef).to(tl.bfloat16), psiv, o)
        o += dskip * psiv.to(tl.float32)
        z = tl.load(Z_ptr + (tokr[:, None] * H + h) * P + p[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        zg = z * tl.load(zeta_ptr + (h * R + m % R)[:, None] * P + p[None, :])
        o = o * tl.load(phi_ptr + (h * R + m % R)[:, None] * P + p[None, :]) * (zg * tl.sigmoid(zg))
        out = tl.sum(tl.reshape(o, (LB, R, P)), axis=1)
        i = tl.arange(0, LB)
        tl.store(O_ptr + ((tok0 + i)[:, None] * H + h) * P + p[None, :], out.to(O_ptr.dtype.element_ty),
                 mask=(j0 + i < cl)[:, None])
        if last_seq >= 0:
            if j0 + LB >= cl:
                lastrow = (cl - 1 - j0) * R
                sel = (m >= lastrow) & (m < lastrow + R)
                fb = fin_k_ptr + ((last_seq * R + (m % R)) * H + h)[:, None].to(tl.int64) * N
                tl.store(fb + n[None, :], k, mask=sel[:, None])
        w = tl.where(rm, ts_r * tl.exp(dlast - dacs), 0.0)
        st = tl.exp(dlast) * st + tl.dot(tl.trans((k.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)


@triton.jit
def _bp_rows(ptr, T, H, NQ_: tl.constexpr, tok0, h, LB: tl.constexpr, R: tl.constexpr, DT_: tl.constexpr):
    """[LB*R, NQ_] per-row copy of a (T, H, NQ_) per-token table at head h (token-major rows)."""
    bp = tl.make_block_ptr(ptr + h * NQ_, shape=(T, NQ_), strides=(H * NQ_, 1), offsets=(tok0, 0),
                           block_shape=(LB, NQ_), order=(1, 0))
    x = tl.load(bp, boundary_check=(0,), padding_option="zero").to(DT_)
    return tl.reshape(tl.broadcast_to(x[:, None, :], (LB, R, NQ_)), (LB * R, NQ_))


@triton.jit
def _bp_qk(X_ptr, T, tok0, g, LB: tl.constexpr, R: tl.constexpr, N: tl.constexpr, NQ: tl.constexpr):
    """[LB*R, NQ] column group g of rows tok0*R .. of a (T, R, N) tensor (block pointer, zero padded)."""
    bp = tl.make_block_ptr(X_ptr, shape=(T * R, N), strides=(N, 1), offsets=(tok0 * R, g * NQ),
                           block_shape=(LB * R, NQ), order=(1, 0))
    return tl.load(bp, boundary_check=(0,), padding_option="zero")


@triton.jit
def _ssd_chunk_scan_bp(
    Q_ptr, K_ptr, V_ptr, Z_ptr, qbr_ptr, kbr_ptr, psir_ptr, phir_ptr, zetar_ptr, D_ptr,
    cl_ptr, sl_ptr, dacs_ptr, ts_ptr, gam_ptr,
    ch_start_ptr, ch_len_ptr, ch_last_ptr,
    Sin_ptr, Ain_ptr, O_ptr, fin_k_ptr,
    T, H,
    R: tl.constexpr, N: tl.constexpr, P: tl.constexpr, NQ: tl.constexpr, LB: tl.constexpr,
):
    """(chunk, head) program, sub-blocks serially from S_in; block-pointer loads (no per-element
    pointer tensors); per-rank constants come pre-expanded to rows: *r_ptr is (H, LB*R, X)."""
    c = tl.program_id(0)
    h = tl.program_id(1)
    cs = tl.load(ch_start_ptr + c)
    cl = tl.load(ch_len_ptr + c)
    last_seq = tl.load(ch_last_ptr + c)
    LBR: tl.constexpr = LB * R
    m = tl.arange(0, LBR)
    nq = tl.arange(0, NQ)
    p = tl.arange(0, P)
    ti_r = m // R
    sbase = Sin_ptr + (c * H + h).to(tl.int64) * N * P
    st0 = tl.load(sbase + nq[:, None] * P + p[None, :]).to(tl.float32)
    st1 = tl.load(sbase + (NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    st2 = tl.load(sbase + (2 * NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    st3 = tl.load(sbase + (3 * NQ + nq[:, None]) * P + p[None, :]).to(tl.float32)
    a0 = tl.load(Ain_ptr + (c * H + h) * NQ + nq)
    ca0 = tl.cos(a0)[None, :]
    sa0 = tl.sin(a0)[None, :]
    dskip = tl.load(D_ptr + h)
    # per-head row constants
    rb = (h * LBR + m)[:, None]
    qb0 = tl.load(qbr_ptr + rb * N + nq[None, :]).to(tl.bfloat16)
    qb1 = tl.load(qbr_ptr + rb * N + NQ + nq[None, :]).to(tl.bfloat16)
    qb2 = tl.load(qbr_ptr + rb * N + 2 * NQ + nq[None, :]).to(tl.bfloat16)
    qb3 = tl.load(qbr_ptr + rb * N + 3 * NQ + nq[None, :]).to(tl.bfloat16)
    base = 0.0
    for j0 in range(0, cl, LB):
        tok0 = cs + j0
        rm = j0 + ti_r < cl
        c_l = _bp_rows(cl_ptr, T, H, NQ, tok0, h, LB, R, tl.float32)
        s_l = _bp_rows(sl_ptr, T, H, NQ, tok0, h, LB, R, tl.float32)
        cr = ca0 * c_l - sa0 * s_l
        sr = sa0 * c_l + ca0 * s_l
        x0 = (_bp_qk(Q_ptr, T, tok0, 0, LB, R, N, NQ) + qb0).to(tl.float32)
        x2 = (_bp_qk(Q_ptr, T, tok0, 2, LB, R, N, NQ) + qb2).to(tl.float32)
        q0 = (cr * x0 - sr * x2).to(tl.bfloat16)
        q2 = (sr * x0 + cr * x2).to(tl.bfloat16)
        q1 = _bp_qk(Q_ptr, T, tok0, 1, LB, R, N, NQ) + qb1
        q3 = _bp_qk(Q_ptr, T, tok0, 3, LB, R, N, NQ) + qb3
        x0 = (_bp_qk(K_ptr, T, tok0, 0, LB, R, N, NQ) + tl.load(kbr_ptr + rb * N + nq[None, :]).to(tl.bfloat16)).to(tl.float32)
        x2 = (_bp_qk(K_ptr, T, tok0, 2, LB, R, N, NQ) + tl.load(kbr_ptr + rb * N + 2 * NQ + nq[None, :]).to(tl.bfloat16)).to(tl.float32)
        k0 = (cr * x0 - sr * x2).to(tl.bfloat16)
        k2 = (sr * x0 + cr * x2).to(tl.bfloat16)
        k1 = _bp_qk(K_ptr, T, tok0, 1, LB, R, N, NQ) + tl.load(kbr_ptr + rb * N + NQ + nq[None, :]).to(tl.bfloat16)
        k3 = _bp_qk(K_ptr, T, tok0, 3, LB, R, N, NQ) + tl.load(kbr_ptr + rb * N + 3 * NQ + nq[None, :]).to(tl.bfloat16)
        tvec = tl.make_block_ptr(dacs_ptr + h * T, shape=(T,), strides=(1,), offsets=(tok0,), block_shape=(LB,), order=(0,))
        dacs_t = tl.load(tvec, boundary_check=(0,), padding_option="zero") - base
        tsv = tl.make_block_ptr(ts_ptr + h * T, shape=(T,), strides=(1,), offsets=(tok0,), block_shape=(LB,), order=(0,))
        ts_t = tl.load(tsv, boundary_check=(0,), padding_option="zero")
        gv = tl.make_block_ptr(gam_ptr + h * T, shape=(T,), strides=(1,), offsets=(tok0,), block_shape=(LB,), order=(0,))
        gam_t = tl.load(gv, boundary_check=(0,), padding_option="zero")
        i = tl.arange(0, LB)
        tmask = j0 + i < cl
        ts_t = tl.where(tmask, ts_t, 0.0)
        last_i = tl.minimum(cl - j0, LB) - 1
        dlast = tl.sum(tl.where(i == last_i, dacs_t, 0.0), axis=0)
        base = base + dlast
        dacs = tl.reshape(tl.broadcast_to(dacs_t[:, None], (LB, R)), (LBR,))
        ts_r = tl.reshape(tl.broadcast_to(ts_t[:, None], (LB, R)), (LBR,))
        gam_r = tl.reshape(tl.broadcast_to(gam_t[:, None], (LB, R)), (LBR,))
        o = tl.dot(q0, st0.to(tl.bfloat16))
        o = tl.dot(q1, st1.to(tl.bfloat16), o)
        o = tl.dot(q2, st2.to(tl.bfloat16), o)
        o = tl.dot(q3, st3.to(tl.bfloat16), o)
        o = o * tl.exp(dacs)[:, None]
        g = tl.dot(q0, tl.trans(k0))
        g = tl.dot(q1, tl.trans(k1), g)
        g = tl.dot(q2, tl.trans(k2), g)
        g = tl.dot(q3, tl.trans(k3), g)
        coef = tl.where(ti_r[:, None] > ti_r[None, :],
                        ts_r[None, :] * tl.exp(tl.minimum(dacs[:, None] - dacs[None, :], 0.0)),
                        tl.where(ti_r[:, None] == ti_r[None, :], gam_r[:, None], 0.0))
        vbp = tl.make_block_ptr(V_ptr + h * P, shape=(T, P), strides=(H * P, 1), offsets=(tok0, 0),
                                block_shape=(LB, P), order=(1, 0))
        v = tl.load(vbp, boundary_check=(0,), padding_option="zero").to(tl.float32)
        psiv = (tl.reshape(tl.broadcast_to(v[:, None, :], (LB, R, P)), (LBR, P))
                * tl.load(psir_ptr + rb * P + p[None, :])).to(tl.bfloat16)
        o = tl.dot((g * coef).to(tl.bfloat16), psiv, o)
        o += dskip * psiv.to(tl.float32)
        zbp = tl.make_block_ptr(Z_ptr + h * P, shape=(T, P), strides=(H * P, 1), offsets=(tok0, 0),
                                block_shape=(LB, P), order=(1, 0))
        z = tl.load(zbp, boundary_check=(0,), padding_option="zero").to(tl.float32)
        zg = tl.reshape(tl.broadcast_to(z[:, None, :], (LB, R, P)), (LBR, P)) * tl.load(zetar_ptr + rb * P + p[None, :])
        o = o * tl.load(phir_ptr + rb * P + p[None, :]) * (zg * tl.sigmoid(zg))
        out = tl.sum(tl.reshape(o, (LB, R, P)), axis=1)
        obp = tl.make_block_ptr(O_ptr + h * P, shape=(T, P), strides=(H * P, 1), offsets=(tok0, 0),
                                block_shape=(LB, P), order=(1, 0))
        # rows of the next chunk / sequence must not be written: mask through the token count
        tl.store(O_ptr + ((tok0 + i)[:, None] * H + h) * P + p[None, :], out.to(O_ptr.dtype.element_ty), mask=tmask[:, None])
        if last_seq >= 0:
            if j0 + LB >= cl:
                lastrow = (cl - 1 - j0) * R
                sel = (m >= lastrow) & (m < lastrow + R)
                fb = fin_k_ptr + ((last_seq * R + (m % R)) * H + h)[:, None].to(tl.int64) * N
                tl.store(fb + nq[None, :], k0, mask=sel[:, None])
                tl.store(fb + NQ + nq[None, :], k1, mask=sel[:, None])
                tl.store(fb + 2 * NQ + nq[None, :], k2, mask=sel[:, None])
                tl.store(fb + 3 * NQ + nq[None, :], k3, mask=sel[:, None])
        w = tl.where(rm, ts_r * tl.exp(dlast - dacs), 0.0)
        dec = tl.exp(dlast)
        st0 = dec * st0 + tl.dot(tl.trans((k0.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st1 = dec * st1 + tl.dot(tl.trans((k1.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st2 = dec * st2 + tl.dot(tl.trans((k2.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)
        st3 = dec * st3 + tl.dot(tl.trans((k3.to(tl.float32) * w[:, None]).to(tl.bfloat16)), psiv)


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


@torch.no_grad()
def mamba3_mimo_hybrid(
    Q, K, V, Z, angles, DT, ADT, Trap, q_bias, k_bias, psi, phi, zeta, D,
    cu_seqlens_cpu: list[int],
    init_states=None,
    chunk_tokens: int = 128,
    tl_chunk: int = 16,
    tl_threads: int = 128,
    tl_stages: int = 0,
):
    """Chunk-parallel exact prefill: the Triton phases give every LS-token chunk its entering state
    and absolute phases, then the TileLang chunked-scan kernel runs once over the chunks as
    independent virtual sequences (zero boundary K/V fold: the entering state already holds the
    previous token's full trapezoid contribution). Same contract as :func:`mamba3_mimo_ssd`."""
    from .mimo_fwd_varlen import mamba_mimo_forward_varlen
    from .segsum import compute_dacs_segsum_triton_varlen
    T, R, N = Q.shape
    H, P = V.shape[1], V.shape[2]
    NQ = N // 4
    dev = Q.device
    starts, lens, sends, lasts, seq_c0, seq_nc = _chunk_layout(cu_seqlens_cpu, chunk_tokens)
    NC, NS = len(starts), len(cu_seqlens_cpu) - 1
    v_cu_list = starts + [cu_seqlens_cpu[-1]]
    meta = torch.tensor([starts, lens, sends, lasts], dtype=torch.int32).to(dev, non_blocking=True)
    smeta = torch.tensor([seq_c0, seq_nc, cu_seqlens_cpu[:-1]], dtype=torch.int32).to(dev, non_blocking=True)
    v_cu = torch.tensor(v_cu_list, dtype=torch.int32).to(dev, non_blocking=True)
    f32 = dict(device=dev, dtype=torch.float32)
    dacs, ts, gam = torch.empty(H, T, **f32), torch.empty(H, T, **f32), torch.empty(H, T, **f32)
    cl_t = torch.empty(T, H, NQ, device=dev, dtype=torch.float16)
    sl_t = torch.empty(T, H, NQ, device=dev, dtype=torch.float16)
    thl = torch.empty(T, H, NQ, **f32)
    dtot, asum = torch.empty(NC, H, **f32), torch.empty(NC, H, NQ, **f32)
    Sloc = torch.empty(NC, H, N, P, **f32)
    SinT = torch.empty(NC, H, P, N, **f32)
    Ain = torch.empty(NC, H, NQ, **f32)
    fin_ssm, fin_ang = torch.empty(NS, H, P, N, **f32), torch.empty(NS, H, NQ, **f32)
    angles = angles.contiguous()
    _ssd_prep[(NC, H)](angles, DT, ADT, Trap, meta[0], meta[1], meta[2], dacs, ts, gam, thl, dtot, asum, cl_t, sl_t,
                       T, H, NQ=NQ, LS=chunk_tokens, num_warps=4)
    _ssd_chunk_state[(NC, H, 1)](K, V, k_bias, psi, cl_t, sl_t, dacs, ts, dtot, meta[0], meta[1], Sloc, T, H,
                                 R=R, N=N, P=P, NQ=NQ, LB=16, PB=P, num_warps=8)
    has_init = init_states is not None
    if has_init:
        in_ang, in_ssm, in_k, in_v = (x.contiguous() for x in init_states)
        in_ang, in_ssm = in_ang.float(), in_ssm.float()
    else:
        in_ang = in_ssm = in_k = in_v = Sloc  # unused
    _ssd_state_pass[(H, NS, P // 16)](
        Sloc, dtot, asum, smeta[0], smeta[1], smeta[2], in_ssm, in_k, in_v, in_ang, psi, DT, Trap,
        SinT, Ain, fin_ssm, fin_ang, T, H, HAS_INIT=has_init, R=R, N=N, P=P, NQ=NQ, SIN_T=True, PB=16, num_warps=2,
    )
    ang_abs = torch.empty(T, H, NQ, **f32)
    _ssd_abs_angles[(NC, H)](thl, Ain, meta[0], meta[1], ang_abs, H, NQ=NQ, LS=chunk_tokens, num_warps=4)
    DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(ADT.view(1, H, T), tl_chunk, cu_seqlens=v_cu)
    zk = torch.zeros(NC, R, H, N, device=dev, dtype=Q.dtype)
    zv = torch.zeros(NC, H, P, device=dev, dtype=V.dtype)
    o, _, k_fin = mamba_mimo_forward_varlen(
        Q.view(1, T, R, 1, N), K.view(1, T, R, 1, N), V.view(1, T, H, P), q_bias, k_bias, psi, phi,
        Z.view(1, T, H, P), D, zeta, ang_abs.view(1, T, H, NQ), DA_CS, DA_CS_REV, DT.view(1, H, T), Trap.view(1, H, T),
        Segsum, tl_chunk, 4, Q.dtype, cu_seqlens=v_cu, return_state=True, initial_states=(SinT, zk, zv),
        threads=tl_threads, num_stages=tl_stages,
    )
    last_rows = torch.tensor([seq_c0[s] + seq_nc[s] - 1 for s in range(NS)], dtype=torch.long).to(dev, non_blocking=True)
    return o.view(T, H, P), fin_ang, fin_ssm, k_fin[last_rows]


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


@torch.no_grad()
def mamba3_mimo_ssd(
    Q: torch.Tensor,  # (T, R, N) bf16 (B/C-normed C)
    K: torch.Tensor,  # (T, R, N) bf16
    V: torch.Tensor,  # (T, H, P) bf16
    Z: torch.Tensor,  # (T, H, P) bf16
    angles: torch.Tensor,  # (T, N // 4) raw per-token angle (shared by heads)
    DT: torch.Tensor,  # (H, T) fp32 softplus(dt + dt_bias)
    ADT: torch.Tensor,  # (H, T) fp32 A * DT
    Trap: torch.Tensor,  # (H, T) bf16
    q_bias: torch.Tensor, k_bias: torch.Tensor,  # (H, R, N) fp32
    psi: torch.Tensor, phi: torch.Tensor, zeta: torch.Tensor,  # (H, R, P) fp32
    D: torch.Tensor,  # (H,) fp32
    cu_seqlens_cpu: list[int],
    init_states: tuple[torch.Tensor, ...] | None = None,  # (angle (NS,H,N/4), ssm (NS,H,P,N), k (NS,R,H,N), v (NS,H,P))
    chunk_tokens: int = 128,
    sub_tokens: int = 16,
    num_warps: int = 4,
    scan_mode: str = "seq",
    p_tile: int = 32,
    ablate: int = 0,
    scan_stages: int = 1,
):
    """Returns (out (T, H, P), final_angle (NS, H, N/4), final_ssm (NS, H, P, N) fp32, final_k (NS, R, H, N))."""
    T, R, N = Q.shape
    H, P = V.shape[1], V.shape[2]
    NQ = N // 4
    dev = Q.device
    starts, lens, sends, lasts, seq_c0, seq_nc = _chunk_layout(cu_seqlens_cpu, chunk_tokens)
    NC, NS = len(starts), len(cu_seqlens_cpu) - 1
    meta = torch.tensor([starts, lens, sends, lasts], dtype=torch.int32).to(dev, non_blocking=True)
    smeta = torch.tensor([seq_c0, seq_nc, cu_seqlens_cpu[:-1]], dtype=torch.int32).to(dev, non_blocking=True)
    f32 = dict(device=dev, dtype=torch.float32)
    dacs, ts, gam = torch.empty(H, T, **f32), torch.empty(H, T, **f32), torch.empty(H, T, **f32)
    cl_t = torch.empty(T, H, N // 4, device=dev, dtype=torch.float16)
    sl_t = torch.empty(T, H, N // 4, device=dev, dtype=torch.float16)
    thl = torch.empty(T, H, NQ, **f32)
    dtot, asum = torch.empty(NC, H, **f32), torch.empty(NC, H, NQ, **f32)
    Sloc = torch.empty(NC, H, N, P, **f32)
    Sin = torch.empty(NC, H, N, P, device=dev, dtype=torch.bfloat16)
    Ain = torch.empty(NC, H, NQ, **f32)
    fin_ssm, fin_ang = torch.empty(NS, H, P, N, **f32), torch.empty(NS, H, NQ, **f32)
    fin_k = torch.empty(NS, R, H, N, device=dev, dtype=Q.dtype)
    out = torch.empty(T, H, P, device=dev, dtype=V.dtype)
    angles = angles.contiguous()
    _ssd_prep[(NC, H)](angles, DT, ADT, Trap, meta[0], meta[1], meta[2], dacs, ts, gam, thl, dtot, asum, cl_t, sl_t,
                       T, H, NQ=NQ, LS=chunk_tokens, num_warps=4)
    if scan_mode == "full_x":
        _ssd_chunk_state_full[(NC, H)](K, V, k_bias, psi, thl, dacs, ts, dtot, meta[0], meta[1], Sloc, T, H,
                                       R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, num_warps=num_warps)
    else:
      _ssd_chunk_state[(NC, H, P // p_tile)](K, V, k_bias, psi, cl_t, sl_t, dacs, ts, dtot, meta[0], meta[1], Sloc, T, H,
                              R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, PB=p_tile, num_warps=num_warps)
    has_init = init_states is not None
    if has_init:
        in_ang, in_ssm, in_k, in_v = (x.contiguous() for x in init_states)
        in_ang, in_ssm = in_ang.float(), in_ssm.float()
    else:
        in_ang = in_ssm = in_k = in_v = Sloc  # unused
    _ssd_state_pass[(H, NS)](
        Sloc, dtot, asum, smeta[0], smeta[1], smeta[2], in_ssm, in_k, in_v, in_ang, psi, DT, Trap,
        Sin, Ain, fin_ssm, fin_ang, T, H, HAS_INIT=has_init, R=R, N=N, P=P, NQ=NQ, num_warps=4,
    )
    if scan_mode == "bp":
        LBR = sub_tokens * R
        rows = lambda x: x.repeat(1, sub_tokens, 1).contiguous()  # (H, R, X) -> (H, LB*R, X), row = tok*R + r
        _ssd_chunk_scan_bp[(NC, H)](
            Q, K, V, Z, rows(q_bias), rows(k_bias), rows(psi), rows(phi), rows(zeta), D, cl_t, sl_t, dacs, ts, gam,
            meta[0], meta[1], meta[3], Sin, Ain, out, fin_k, T, H,
            R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, num_warps=num_warps, num_stages=scan_stages,
        )
    elif scan_mode == "full":
        _ssd_chunk_scan_full[(NC, H)](
            Q, K, V, Z, q_bias, k_bias, psi, phi, zeta, D, thl, dacs, ts, gam,
            meta[0], meta[1], meta[3], Sin, Ain, out, fin_k, T, H,
            R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, num_warps=num_warps, num_stages=scan_stages,
        )
    elif scan_mode == "tiles":
        NT = chunk_tokens // sub_tokens
        _ssd_chunk_scan[(NC * NT, H)](
            Q, K, V, Z, q_bias, k_bias, psi, phi, zeta, D, thl, dacs, ts, gam,
            meta[0], meta[1], meta[3], Sin, Ain, out, fin_k, T, H,
            R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, NT=NT, num_warps=num_warps,
        )
    else:
        _ssd_chunk_scan_seq[(NC, H, P // p_tile)](
            Q, K, V, Z, q_bias, k_bias, psi, phi, zeta, D, cl_t, sl_t, dacs, ts, gam,
            meta[0], meta[1], meta[3], Sin, Ain, out, fin_k, T, H,
            R=R, N=N, P=P, NQ=NQ, LB=sub_tokens, PB=p_tile, ABL=ablate, num_warps=num_warps, num_stages=scan_stages,
        )
    return out, fin_ang, fin_ssm, fin_k
