"""Phase A: decode attention over a TPA-factorized KV cache (standalone).

Cache row per token (bf16), offsets:
  [0:512)     Bk  (4 x 128)      [512:1024) Bv (4 x 128)
  [1024:1072) Ck  (12 x 4)       [1072:1120) Dk (12 x 4)
  [1120:1168) Cv  (12 x 4)       [1168:1216) Dv (12 x 4)
k_t[h] = sum_r Ck[h,r] Bk_t[r] + Dk[h,r] Bk_{t-1}[r]   (same for v)
"""
import torch, triton, triton.language as tl

ROW = 1216; OFF_BK, OFF_BV, OFF_CK, OFF_DK, OFF_CV, OFF_DV = 0, 512, 1024, 1072, 1120, 1168
HQ, HKV, R, D, GQ = 48, 12, 4, 128, 4
HQP = 64


@triton.jit
def _tpa_decode_split_kernel(
    q_ptr,            # [B, HQ, D] bf16
    cache_ptr,        # [NB, BS, ROW] bf16
    bt_ptr, bt_stride,  # block table [B, max_blocks] int32
    seqlen_ptr,       # [B] int32
    part_o_ptr, part_m_ptr, part_l_ptr,  # [B, SPLIT, HQP, D] fp32, [B, SPLIT, HQP] fp32 x2
    sm_scale,
    TOK_PER_SPLIT: tl.constexpr, T: tl.constexpr, BS: tl.constexpr,
    ROW: tl.constexpr, HQ: tl.constexpr, HQP: tl.constexpr, HKV: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    OFF_BK: tl.constexpr, OFF_BV: tl.constexpr, OFF_CK: tl.constexpr, OFF_DK: tl.constexpr,
    OFF_CV: tl.constexpr, OFF_DV: tl.constexpr,
):
    b = tl.program_id(0); s = tl.program_id(1)
    seqlen = tl.load(seqlen_ptr + b)
    start = s * TOK_PER_SPLIT
    offs_h = tl.arange(0, HQP); offs_d = tl.arange(0, D)
    hmask = offs_h < HQ
    hq = tl.minimum(offs_h, HQ - 1)
    q = tl.load(q_ptr + b * HQ * D + hq[:, None] * D + offs_d[None, :], mask=hmask[:, None], other=0.0)  # [HQP, D]
    grp = hq // (HQ // HKV)                                                            # kv group per q head
    m_i = tl.full([HQP], float("-inf"), tl.float32); l_i = tl.zeros([HQP], tl.float32)
    acc = tl.zeros([HQP, D], tl.float32)
    tr = tl.arange(0, T * R); t_of = tr // R; r_of = tr % R
    stop = tl.minimum(start + TOK_PER_SPLIT, seqlen)
    for t0 in range(start, stop, T):
        # token index per (t, r) lane and its predecessor
        tt = t0 + t_of
        vt = tt < seqlen
        vp = vt & (tt >= 1)
        tpv = tl.maximum(tt - 1, 0)
        blk = tl.load(bt_ptr + b * bt_stride + tt // BS, mask=vt, other=0).to(tl.int64)
        blkp = tl.load(bt_ptr + b * bt_stride + tpv // BS, mask=vp, other=0).to(tl.int64)
        row = blk * BS + (tt % BS)                                                     # [T*R] cache rows
        rowp = blkp * BS + (tpv % BS)
        Bk = tl.load(cache_ptr + row[:, None] * ROW + OFF_BK + r_of[:, None] * D + offs_d[None, :], mask=vt[:, None], other=0.0)   # [T*R, D]
        Bkp = tl.load(cache_ptr + rowp[:, None] * ROW + OFF_BK + r_of[:, None] * D + offs_d[None, :], mask=vp[:, None], other=0.0)
        qB = tl.dot(q, tl.trans(Bk))                                                   # [HQP, T*R] fp32
        qBp = tl.dot(q, tl.trans(Bkp))
        # group-level coefficient loads [16 groups(12 real), T*R], broadcast to the 4 heads of each group
        og = tl.arange(0, 16); gmask = og < HKV; gg = tl.minimum(og, HKV - 1)
        cCg = tl.load(cache_ptr + row[None, :] * ROW + OFF_CK + gg[:, None] * R + r_of[None, :], mask=vt[None, :] & gmask[:, None], other=0.0).to(tl.float32)
        cDg = tl.load(cache_ptr + row[None, :] * ROW + OFF_DK + gg[:, None] * R + r_of[None, :], mask=vt[None, :] & gmask[:, None], other=0.0).to(tl.float32)
        cC = tl.reshape(tl.broadcast_to(cCg[:, None, :], [16, HQP // 16, T * R]), [HQP, T * R])
        cD = tl.reshape(tl.broadcast_to(cDg[:, None, :], [16, HQP // 16, T * R]), [HQP, T * R])
        prod = qB * cC + qBp * cD                                                      # [HQP, T*R]
        sc = tl.sum(tl.reshape(prod, [HQP, T, R]), axis=2) * sm_scale                  # [HQP, T]
        offs_t = t0 + tl.arange(0, T)
        valid_t = offs_t < seqlen
        sc = tl.where(valid_t[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])                                                # [HQP, T]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        vCg = tl.load(cache_ptr + row[None, :] * ROW + OFF_CV + gg[:, None] * R + r_of[None, :], mask=vt[None, :] & gmask[:, None], other=0.0).to(tl.float32)
        vDg = tl.load(cache_ptr + row[None, :] * ROW + OFF_DV + gg[:, None] * R + r_of[None, :], mask=vt[None, :] & gmask[:, None], other=0.0).to(tl.float32)
        vC = tl.reshape(tl.broadcast_to(vCg[:, None, :], [16, HQP // 16, T * R]), [HQP, T * R])
        vD = tl.reshape(tl.broadcast_to(vDg[:, None, :], [16, HQP // 16, T * R]), [HQP, T * R])
        p_tr = tl.reshape(tl.broadcast_to(p[:, :, None], [HQP, T, R]), [HQP, T * R])
        Bv = tl.load(cache_ptr + row[:, None] * ROW + OFF_BV + r_of[:, None] * D + offs_d[None, :], mask=vt[:, None], other=0.0)
        Bvp = tl.load(cache_ptr + rowp[:, None] * ROW + OFF_BV + r_of[:, None] * D + offs_d[None, :], mask=vp[:, None], other=0.0)
        acc += tl.dot((p_tr * vC).to(tl.bfloat16), Bv) + tl.dot((p_tr * vD).to(tl.bfloat16), Bvp)
        m_i = m_new
    base = (b * tl.num_programs(1) + s) * HQP
    tl.store(part_o_ptr + base * D + offs_h[:, None] * D + offs_d[None, :], acc)
    tl.store(part_m_ptr + base + offs_h, m_i)
    tl.store(part_l_ptr + base + offs_h, l_i)


@triton.jit
def _tpa_decode_combine_kernel(part_o_ptr, part_m_ptr, part_l_ptr, o_ptr, SPLIT: tl.constexpr, HQ: tl.constexpr, HQP: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    offs_h = tl.arange(0, HQP); offs_d = tl.arange(0, D)
    hmask = offs_h < HQ
    m = tl.full([HQP], float("-inf"), tl.float32)
    for s in range(SPLIT):
        m = tl.maximum(m, tl.load(part_m_ptr + (b * SPLIT + s) * HQP + offs_h, mask=hmask, other=float("-inf")))
    l = tl.zeros([HQP], tl.float32); acc = tl.zeros([HQP, D], tl.float32)
    for s in range(SPLIT):
        ms = tl.load(part_m_ptr + (b * SPLIT + s) * HQP + offs_h, mask=hmask, other=float("-inf"))
        w = tl.where(hmask, tl.exp(ms - m), 0.0)
        l += w * tl.load(part_l_ptr + (b * SPLIT + s) * HQP + offs_h, mask=hmask, other=0.0)
        acc += w[:, None] * tl.load(part_o_ptr + ((b * SPLIT + s) * HQP + offs_h[:, None]) * D + offs_d[None, :], mask=hmask[:, None], other=0.0)
    tl.store(o_ptr + b * HQ * D + offs_h[:, None] * D + offs_d[None, :], (acc / l[:, None]).to(tl.bfloat16), mask=hmask[:, None])


def tpa_decode_attention(q, cache, block_table, seq_lens, sm_scale, bs, tok_per_split=int(__import__("os").environ.get("TPA_SPLIT", "512")), T=int(__import__("os").environ.get("TPA_T", "16")), num_stages=int(__import__("os").environ.get("TPA_NS", "1")), num_warps=int(__import__("os").environ.get("TPA_NW", "4"))):
    B = q.shape[0]
    max_len = int(seq_lens.max().item())
    SPLIT = max(1, -(-max_len // tok_per_split))
    part_o = torch.empty(B, SPLIT, HQP, D, dtype=torch.float32, device=q.device)
    part_m = torch.empty(B, SPLIT, HQP, dtype=torch.float32, device=q.device)
    part_l = torch.empty(B, SPLIT, HQP, dtype=torch.float32, device=q.device)
    _tpa_decode_split_kernel[(B, SPLIT)](
        q, cache, block_table, block_table.stride(0), seq_lens, part_o, part_m, part_l, sm_scale,
        TOK_PER_SPLIT=tok_per_split, T=T, BS=bs, ROW=ROW, HQ=HQ, HQP=HQP, HKV=HKV, R=R, D=D,
        OFF_BK=OFF_BK, OFF_BV=OFF_BV, OFF_CK=OFF_CK, OFF_DK=OFF_DK, OFF_CV=OFF_CV, OFF_DV=OFF_DV, num_warps=num_warps, num_stages=num_stages)
    o = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=q.device)
    _tpa_decode_combine_kernel[(B,)](part_o, part_m, part_l, o, SPLIT=SPLIT, HQ=HQ, HQP=HQP, D=D, num_warps=4)
    return o


# ---------------- reference: dense reconstruction + attention ----------------
def make_cache(B, L, bs, device):
    """Random factor cache + block tables; returns (cache, block_table, seq_lens, dense_k, dense_v)."""
    nblk = -(-L // bs); NB = B * nblk + 1
    cache = torch.zeros(NB, bs, ROW, dtype=torch.bfloat16, device=device)
    bt = torch.zeros(B, nblk, dtype=torch.int32, device=device)
    kd = torch.zeros(B, L, HKV, D, dtype=torch.float32, device=device); vd = torch.zeros_like(kd)
    for b in range(B):
        blocks = torch.arange(1 + b * nblk, 1 + (b + 1) * nblk, device=device, dtype=torch.int32); bt[b] = blocks
        Bk = torch.randn(L, R, D, device=device) * 0.3; Bv = torch.randn(L, R, D, device=device) * 0.3
        Ck = torch.randn(L, HKV, R, device=device) * 0.5; Dk = torch.randn(L, HKV, R, device=device) * 0.5; Dk[0] = 0
        Cv = torch.randn(L, HKV, R, device=device) * 0.5; Dv = torch.randn(L, HKV, R, device=device) * 0.5; Dv[0] = 0
        rows = torch.cat([Bk.reshape(L, -1), Bv.reshape(L, -1), Ck.reshape(L, -1), Dk.reshape(L, -1), Cv.reshape(L, -1), Dv.reshape(L, -1)], 1).to(torch.bfloat16)
        for t in range(L):
            cache[blocks[t // bs], t % bs] = rows[t]
        Bkb, Bvb = Bk.to(torch.bfloat16).float(), Bv.to(torch.bfloat16).float()
        Ckb, Dkb, Cvb, Dvb = (x.to(torch.bfloat16).float() for x in (Ck, Dk, Cv, Dv))
        Bkp = torch.cat([torch.zeros_like(Bkb[:1]), Bkb[:-1]]); Bvp = torch.cat([torch.zeros_like(Bvb[:1]), Bvb[:-1]])
        kd[b] = torch.einsum("thr,trd->thd", Ckb, Bkb) + torch.einsum("thr,trd->thd", Dkb, Bkp)
        vd[b] = torch.einsum("thr,trd->thd", Cvb, Bvb) + torch.einsum("thr,trd->thd", Dvb, Bvp)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=device)
    return cache, bt, seq_lens, kd, vd

def ref_attention(q, kd, vd, sm_scale):
    B = q.shape[0]; qf = q.float().view(B, HKV, GQ, D)               # heads grouped by kv head
    k = kd.permute(0, 2, 1, 3); v = vd.permute(0, 2, 1, 3)           # [B, HKV, L, D]
    s = torch.einsum("bhgd,bhld->bhgl", qf, k) * sm_scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhgl,bhld->bhgd", p, v).reshape(B, HQ, D)

if __name__ == "__main__":
    import time, sys
    torch.manual_seed(0); dev = "cuda"
    for (B, L) in [(2, 300), (4, 1000)]:
        cache, bt, sl, kd, vd = make_cache(B, L, 144, dev)
        q = (torch.randn(B, HQ, D, device=dev) * 0.5).bfloat16()
        out = tpa_decode_attention(q, cache, bt, sl, 1.0 / D ** 0.5, 144)
        ref = ref_attention(q, kd, vd, 1.0 / D ** 0.5)
        err = (out.float() - ref).abs().max().item(); scale = ref.abs().max().item()
        print(f"B={B} L={L}: max|diff|={err:.3e} (ref max {scale:.3f}, rel {err/scale:.2e})")
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        B, L = 64, 8192
        cache, bt, sl, kd, vd = make_cache(B, L, 144, dev)
        q = (torch.randn(B, HQ, D, device=dev) * 0.5).bfloat16()
        for _ in range(3): tpa_decode_attention(q, cache, bt, sl, 1.0 / D ** 0.5, 144)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): tpa_decode_attention(q, cache, bt, sl, 1.0 / D ** 0.5, 144)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 20 * 1e3
        gb = B * L * ROW * 2 / 1e9
        print(f"decode B={B} L={L}: {dt:.3f} ms per layer  ({gb:.2f} GB factors -> {gb/dt:.2f} TB/s; dense bf16 would be {B*L*HKV*D*2*2/1e9:.1f} GB)")
