// Olala Differential-TPA decode attention over the TPA-factorized paged KV cache (sm_80+, mma.sync edition).
// Same cache layout, launch contract and math as tpa_factor_decode.cu (the sm_90a wgmma kernel), for Ampere / Ada
// (and any GPU without wgmma / TMA). Built JIT by tpa_factor.py for the running GPU's arch.
// One CTA per (request, KV split), 4 warps: 3 consumer warps = 48 query heads (16 rows each, no padding) and one
// producer warp that streams 7-token tiles with cp.async (zero-fill outside the split) into a 2-stage pipeline, so the
// consumers spend no issue slots on address math; named barriers (full / empty per buffer) hand the buffers over; plane rows keep their plain [n][d]
// order with a 272-byte smem row stride (conflict-free ldmatrix). A tile's B rows are its 7 tokens plus the previous
// token (the D term of the token shift): that halo is copied smem -> smem from the previous tile, so only a split's
// first tile reads it from global memory. 7 tokens keep the CTA at ~40 KB of shared memory: two CTAs (8 warps, two
// per SM sub-partition) per SM on sm_86 / sm_89 (100 KB per SM), against one with 13-token tiles. qB = Q Bk^T runs as 4 n8 tiles of
// mma.m16n8k16 (B via ldmatrix), acc += W Bv as 16 n8 tiles over k = 32 (B via ldmatrix.trans). The m16n8 fragments
// are the per-warp layout of the wgmma accumulators, so the score / softmax / W code is the sm_90a kernel's.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {
constexpr int HQ = 48, HKV = 12, R = 4, D = 128;
constexpr int SPLIT_TILE = 13;                          // split granularity of tpa_factor.py (TILE there)
constexpr int T = 7, NTOK = T + 1, NCOL = NTOK * R;     // 32 columns (tokidx 0 = t0-1, the halo)
constexpr int NT8 = NCOL / 8, KS = NCOL / 16;
constexpr int ROWP = 608, OFF_C = 512, NCOEF = 96;
constexpr int NCWARPS = HQ / 16, CTHREADS = NCWARPS * 32, THREADS = CTHREADS + 32, NBUF = 2;
constexpr int BAR_FULL = 1, BAR_EMPTY = BAR_FULL + NBUF;   // named barrier ids (0 is __syncthreads)
constexpr int LDR = D + 8;                               // padded smem row (bf16): 272 B
constexpr int KROWS = NCOL;                              // W Bv runs k = 32: no zero rows
constexpr int CHUNKS_B = R * D * 2 / 16;                 // 64 x 16B per token and plane
constexpr int CHUNKS_C = NCOEF * 2 / 16;                 // 12 x 16B coefficients per token and plane

struct __align__(128) Smem {
  __nv_bfloat16 bk[NBUF][NCOL * LDR];
  __nv_bfloat16 bv[NBUF][KROWS * LDR];
  __nv_bfloat16 ck[NBUF][T * NCOEF];   // tokens 1..T: [tok-1][Ck 48 | Dk 48]
  __nv_bfloat16 cv[NBUF][T * NCOEF];
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* s, const void* g, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(s)), "l"(g), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(THREADS) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(THREADS) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Tile of tokens t0 .. t0+6 (and the halo t0-1 if `halo`) into buffer `buf`, issued by the producer warp; tokens
// outside [0, stop) are zero-filled without reading memory (stop = end of this split: later tokens belong to the next
// split). Lane j < NTOK resolves token t0-1+j's row through the block table once; the copies of each token then only
// add constant offsets to that row, so no cp.async waits on a block-table load.
__device__ __forceinline__ void load_tile(Smem& S, int buf, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl,
                                          long long bstride, const int* __restrict__ btb, int BS, int stop, int t0, bool halo, int lane) {
  const int tl = t0 - 1 + lane;
  const bool vl = lane < NTOK && tl >= 0 && tl < stop;
  const long long growl = vl ? (long long)__ldg(btb + tl / BS) * bstride + (long long)(tl % BS) * ROWP : 0;
  const unsigned vmask = __ballot_sync(0xffffffffu, vl);
#pragma unroll
  for (int tok = 0; tok < NTOK; ++tok) {
    const long long grow = __shfl_sync(0xffffffffu, growl, tok);
    if (tok == 0 && !halo) continue;
    const bool valid = (vmask >> tok) & 1u;
#pragma unroll
    for (int j = 0; j < 2 * CHUNKS_B / 32; ++j) {   // B rows: 2 planes x 64 chunks of 16 B
      const int c = lane + 32 * j, pl = c / CHUNKS_B, cc = c % CHUNKS_B, row = tok * R + cc / 16, col = (cc % 16) * 8;
      const __nv_bfloat16* src = (pl == 0 ? kpl : vpl) + grow + (cc / 16) * D + col;
      cp16((pl == 0 ? S.bk[buf] : S.bv[buf]) + row * LDR + col, src, valid);
    }
    if (tok >= 1 && lane < 2 * CHUNKS_C) {   // coefficients: 2 planes x 12 chunks; token 0's are never used
      const int pl = lane / CHUNKS_C, k = lane % CHUNKS_C;
      const __nv_bfloat16* src = (pl == 0 ? kpl : vpl) + grow + OFF_C + k * 8;
      cp16((pl == 0 ? S.ck[buf] : S.cv[buf]) + (tok - 1) * NCOEF + k * 8, src, valid);
    }
  }
}

__global__ void __launch_bounds__(THREADS)
tpa_factor_decode_sm80_kernel(const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl, long long bstride,
                              const __nv_bfloat16* __restrict__ q, const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                              float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l, int hqp,
                              float sm_scale, float softcap, const int* __restrict__ tok_per_split_ptr, int BS) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int tok_per_split = *tok_per_split_ptr;
  const float inv_cap = softcap > 0.f ? 1.f / softcap : 0.f;
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qd = lane & 3, rl = lane >> 2;
  const int row0 = warp * 16 + rl, row1 = row0 + 8;
  const int g0 = row0 / (HQ / HKV), g1 = row1 / (HQ / HKV);
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);
  const size_t base = ((size_t)b * nsplit + sp) * hqp;
  if (stop <= start) {
    for (int i = tid; i < hqp; i += THREADS) { part_m[base + i] = -CUDART_INF_F; part_l[base + i] = 0.f; }
    return;
  }
  const int* btb = bt + (size_t)b * bt_stride;
  const int ntiles = (stop - start + T - 1) / T;
  if (warp == NCWARPS) {   // ---- producer: keeps two tiles in flight, hands each buffer over through BAR_FULL
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it % NBUF;
      if (it >= NBUF) bar_sync(BAR_EMPTY + buf);   // consumers are done with tile it - NBUF
      load_tile(S, buf, kpl, vpl, bstride, btb, BS, stop, start + it * T, it == 0, lane);
      cp_commit();
      if (it >= 1) { cp_wait<1>(); bar_arrive(BAR_FULL + (it - 1) % NBUF); }
    }
    cp_wait<0>();
    bar_arrive(BAR_FULL + (ntiles - 1) % NBUF);
    return;
  }

  uint32_t qa[D / 16][4];
  {
    const __nv_bfloat16* qb = q + (size_t)b * HQ * D;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      const int col = ks * 16 + 2 * qd;
      qa[ks][0] = *reinterpret_cast<const uint32_t*>(qb + row0 * D + col);
      qa[ks][1] = *reinterpret_cast<const uint32_t*>(qb + row1 * D + col);
      qa[ks][2] = *reinterpret_cast<const uint32_t*>(qb + row0 * D + col + 8);
      qa[ks][3] = *reinterpret_cast<const uint32_t*>(qb + row1 * D + col + 8);
    }
  }
  if constexpr (KROWS > NCOL) {   // zero padding rows of W Bv's k (none with 7-token tiles)
    for (int i = tid; i < NBUF * (KROWS - NCOL) * (LDR / 8); i += CTHREADS) {
      const int buf = i / ((KROWS - NCOL) * (LDR / 8)), rem = i % ((KROWS - NCOL) * (LDR / 8));
      *reinterpret_cast<uint4*>(S.bv[buf] + NCOL * LDR + rem * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  float acc[D / 8 * 4];
#pragma unroll
  for (int i = 0; i < D / 8 * 4; ++i) acc[i] = 0.f;
  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;

  for (int it = 0; it < ntiles; ++it) {
    const int buf = it % NBUF, t0 = start + it * T;
    bar_sync(BAR_FULL + buf);   // tile it landed; the halo rows written by the consumers after tile it - 1 too
    const __nv_bfloat16* CfK = S.ck[buf] - NCOEF;   // token tok -> CfK + tok*NCOEF for tok >= 1
    const __nv_bfloat16* CfV = S.cv[buf] - NCOEF;

    // ---- qB = Q Bk^T (16 x NCOL per warp): B fragments of two k16 steps per ldmatrix.x4
    float qB[NT8 * 4];
#pragma unroll
    for (int i = 0; i < NT8 * 4; ++i) qB[i] = 0.f;
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
#pragma unroll
      for (int k2 = 0; k2 < D / 32; ++k2) {
        uint32_t bf[4];
        ldsm_x4(bf, S.bk[buf] + (nt * 8 + (lane & 7)) * LDR + k2 * 32 + (lane >> 3) * 8);
        mma16816(qB + nt * 4, qa[2 * k2], bf[0], bf[1]);
        mma16816(qB + nt * 4, qa[2 * k2 + 1], bf[2], bf[3]);
      }
    }

    // ---- scores: this thread holds cols {2qd, 2qd+1} of each n8 tile: tokidx k = 2nt + (qd>>1), ranks 2(qd&1)+{0,1}
    float sc0[NT8], sc1[NT8], ad0[NT8], ad1[NT8];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), rp = 2 * (qd & 1);
      float ac0 = 0.f, ac1 = 0.f, d0 = 0.f, d1 = 0.f;
      if (k >= 1 && k <= T) {
        const __nv_bfloat16* Ck = CfK + k * NCOEF;
        float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g0 * R + rp));
        float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g1 * R + rp));
        ac0 = c0.x * qB[nt * 4 + 0] + c0.y * qB[nt * 4 + 1];
        ac1 = c1.x * qB[nt * 4 + 2] + c1.y * qB[nt * 4 + 3];
      }
      if (k <= T - 1) {
        const __nv_bfloat16* Dk = CfK + (k + 1) * NCOEF + 48;
        float2 e0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dk + g0 * R + rp));
        float2 e1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dk + g1 * R + rp));
        d0 = e0.x * qB[nt * 4 + 0] + e0.y * qB[nt * 4 + 1];
        d1 = e1.x * qB[nt * 4 + 2] + e1.y * qB[nt * 4 + 3];
      }
      ac0 += __shfl_xor_sync(0xffffffffu, ac0, 1); ac1 += __shfl_xor_sync(0xffffffffu, ac1, 1);
      d0 += __shfl_xor_sync(0xffffffffu, d0, 1); d1 += __shfl_xor_sync(0xffffffffu, d1, 1);
      sc0[nt] = ac0; sc1[nt] = ac1; ad0[nt] = d0; ad1[nt] = d1;
    }
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      float same0 = __shfl_xor_sync(0xffffffffu, ad0[nt], 2), same1 = __shfl_xor_sync(0xffffffffu, ad1[nt], 2);
      float prev0 = (nt > 0) ? __shfl_xor_sync(0xffffffffu, ad0[nt > 0 ? nt - 1 : 0], 2) : 0.f;
      float prev1 = (nt > 0) ? __shfl_xor_sync(0xffffffffu, ad1[nt > 0 ? nt - 1 : 0], 2) : 0.f;
      const bool odd = (qd >> 1) == 1;
      sc0[nt] += odd ? same0 : prev0; sc1[nt] += odd ? same1 : prev1;
    }
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), t = k - 1;
      const bool ok = (k >= 1) && (k <= T) && (t0 + t < stop);
      float s0 = sc0[nt] * sm_scale, s1 = sc1[nt] * sm_scale;
      if (softcap > 0.f) { s0 = softcap * tanhf(s0 * inv_cap); s1 = softcap * tanhf(s1 * inv_cap); }
      sc0[nt] = ok ? s0 : -CUDART_INF_F;
      sc1[nt] = ok ? s1 : -CUDART_INF_F;
    }
    // ---- online softmax per row (4 lanes share a row)
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) { mx0 = fmaxf(mx0, sc0[nt]); mx1 = fmaxf(mx1, sc1[nt]); }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float al0 = (m0 == -CUDART_INF_F) ? 0.f : __expf(m0 - mx0);
    const float al1 = (m1 == -CUDART_INF_F) ? 0.f : __expf(m1 - mx1);
    float ls0 = 0.f, ls1 = 0.f, pr0[NT8], pr1[NT8];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      pr0[nt] = (sc0[nt] == -CUDART_INF_F) ? 0.f : __expf(sc0[nt] - mx0);
      pr1[nt] = (sc1[nt] == -CUDART_INF_F) ? 0.f : __expf(sc1[nt] - mx1);
      ls0 += pr0[nt]; ls1 += pr1[nt];
    }
    ls0 += __shfl_xor_sync(0xffffffffu, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffffu, ls1, 2);
    l0 = l0 * al0 + ls0; l1 = l1 * al1 + ls1; m0 = mx0; m1 = mx1;
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) { acc[nt * 4 + 0] *= al0; acc[nt * 4 + 1] *= al0; acc[nt * 4 + 2] *= al1; acc[nt * 4 + 3] *= al1; }

    // ---- W (A operand, 16 x KROWS per warp): W[row][n] = p[row][tokidx-1] Cv[tokidx-1] + p[row][tokidx] Dv[tokidx], n = tokidx*4 + r
    uint32_t aW[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int rp = 2 * (qd & 1);
      float w[8];
      const bool hiP = (qd >> 1) == 1;
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int tokidx = 4 * ks + (qd >> 1) + 2 * hi, ntA = 2 * ks + hi;
        const int ntB0 = ntA < NT8 ? ntA : NT8 - 1, ntB1 = ntA + 1 < NT8 ? ntA + 1 : NT8 - 1;
        const float sb0 = hiP ? pr0[ntB0] : pr0[ntB1], sb1 = hiP ? pr1[ntB0] : pr1[ntB1];
        const float pb0 = __shfl_xor_sync(0xffffffffu, sb0, 2), pb1 = __shfl_xor_sync(0xffffffffu, sb1, 2);
        const float pa0 = pr0[ntB0], pa1 = pr1[ntB0];
        float2 v0 = make_float2(0.f, 0.f), v1 = make_float2(0.f, 0.f);
        if (tokidx < NTOK) {
          if (tokidx >= 1) {
            const __nv_bfloat16* Cv = CfV + tokidx * NCOEF;
            float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g0 * R + rp));
            float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g1 * R + rp));
            v0.x += pa0 * c0.x; v0.y += pa0 * c0.y; v1.x += pa1 * c1.x; v1.y += pa1 * c1.y;
          }
          if (tokidx <= T - 1) {
            const __nv_bfloat16* Dv = CfV + (tokidx + 1) * NCOEF + 48;
            float2 d0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g0 * R + rp));
            float2 d1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g1 * R + rp));
            v0.x += pb0 * d0.x; v0.y += pb0 * d0.y; v1.x += pb1 * d1.x; v1.y += pb1 * d1.y;
          }
        }
        w[4 * hi + 0] = v0.x; w[4 * hi + 1] = v0.y; w[4 * hi + 2] = v1.x; w[4 * hi + 3] = v1.y;
      }
      aW[ks][0] = pack_bf16(w[0], w[1]); aW[ks][1] = pack_bf16(w[2], w[3]); aW[ks][2] = pack_bf16(w[4], w[5]); aW[ks][3] = pack_bf16(w[6], w[7]);
    }
    // ---- acc += W Bv: Bv rows are k (tokidx*4 + r), d contiguous -> ldmatrix.trans, two n8 tiles per x4
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int n2 = 0; n2 < D / 16; ++n2) {
        uint32_t bf[4];
        const int mi = lane >> 3;
        ldsm_x4_t(bf, S.bv[buf] + (ks * 16 + (mi & 1) * 8 + (lane & 7)) * LDR + n2 * 16 + (mi >> 1) * 8);
        mma16816(acc + (2 * n2) * 4, aW[ks], bf[0], bf[1]);
        mma16816(acc + (2 * n2 + 1) * 4, aW[ks], bf[2], bf[3]);
      }
    }
    // ---- halo of the next tile = this tile's last token: its B rows (both planes) into rows 0..R-1 of the next
    // buffer, whose cp.async fill (rows R..) may still be in flight. Published to the other consumers by BAR_FULL.
    if (it + 1 < ntiles) {
      const int nb = (it + 1) % NBUF;
      for (int i = tid; i < 2 * R * (D / 8); i += CTHREADS) {
        const int pl = i / (R * (D / 8)), r = (i / (D / 8)) % R, c = i % (D / 8);
        const __nv_bfloat16* src = (pl == 0 ? S.bk[buf] : S.bv[buf]) + ((NTOK - 1) * R + r) * LDR + c * 8;
        __nv_bfloat16* dst = (pl == 0 ? S.bk[nb] : S.bv[nb]) + r * LDR + c * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      }
    }
    if (it + NBUF < ntiles) bar_arrive(BAR_EMPTY + buf);   // the producer may refill this buffer with tile it + 2
  }
  // ---- partials: acc[nt*4 + {0,1}] -> (row0, cols nt*8 + 2qd, +1); {2,3} -> row1
#pragma unroll
  for (int nt = 0; nt < D / 8; ++nt) {
    float* o0 = part_o + (base + row0) * D + nt * 8 + 2 * qd; float* o1 = part_o + (base + row1) * D + nt * 8 + 2 * qd;
    *reinterpret_cast<float2*>(o0) = make_float2(acc[nt * 4 + 0], acc[nt * 4 + 1]);
    *reinterpret_cast<float2*>(o1) = make_float2(acc[nt * 4 + 2], acc[nt * 4 + 3]);
  }
  if (qd == 0) { part_m[base + row0] = m0; part_m[base + row1] = m1; part_l[base + row0] = l0; part_l[base + row1] = l1; }
}
}  // namespace

void tpa_factor_decode_sm80_launch(torch::Tensor q, torch::Tensor kplane, torch::Tensor vplane, torch::Tensor block_table, torch::Tensor seq_lens,
                                   torch::Tensor tok_per_split, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                                   double sm_scale, double softcap, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1), hqp = part_o.size(2);
  TORCH_CHECK(kplane.dim() == 3 && kplane.size(2) == ROWP && kplane.stride(2) == 1 && kplane.stride(1) == ROWP && q.is_contiguous());
  TORCH_CHECK(vplane.stride(0) == kplane.stride(0) && vplane.stride(1) == ROWP && kplane.size(1) == block_size && block_size % SPLIT_TILE == 0);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 && tok_per_split.dtype() == torch::kInt32);
  TORCH_CHECK(hqp >= HQ && part_o.size(3) == D && part_o.is_contiguous() && part_m.is_contiguous() && part_l.is_contiguous());
  TORCH_CHECK((kplane.stride(0) * 2) % 16 == 0 && reinterpret_cast<uintptr_t>(kplane.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(vplane.data_ptr()) % 16 == 0, "factor planes must be 16B aligned");
  static bool attr_set = false;
  const int smem = (int)sizeof(Smem);
  if (!attr_set) { cudaFuncSetAttribute(tpa_factor_decode_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr_set = true; }
  tpa_factor_decode_sm80_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(kplane.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vplane.data_ptr()), (long long)kplane.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), hqp, (float)sm_scale, (float)softcap,
      tok_per_split.data_ptr<int>(), (int)block_size);
}
int64_t tpa_factor_sm80_smem_bytes() { return sizeof(Smem); }
int64_t tpa_factor_sm80_occupancy() {
  const int smem = (int)sizeof(Smem);
  cudaFuncSetAttribute(tpa_factor_decode_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int nb = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_factor_decode_sm80_kernel, THREADS, smem);
  return nb;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch", &tpa_factor_decode_sm80_launch); m.def("smem_bytes", &tpa_factor_sm80_smem_bytes);
  m.def("occupancy", &tpa_factor_sm80_occupancy); m.attr("TILE") = SPLIT_TILE; m.attr("INNER_TILE") = T;
}
