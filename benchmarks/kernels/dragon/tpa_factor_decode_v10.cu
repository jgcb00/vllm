// Dragon Differential-TPA decode attention over the TPA-factorized paged KV cache (sm_90).
//
// Per token the cache stores two 608-element bf16 rows: K plane [Bk (4x128) | Ck (12x4) | Dk (12x4)] and
// V plane [Bv | Cv | Dv]; k_t = sum_r Ck[t,h,r] Bk[t,r] + sum_r Dk[t,h,r] Bk[t-1,r] (token shift folded in), same for v.
// Bk/Bv rows are stored with 16B chunk c of factor row r at (c&8)|((c&7)^r) so ldmatrix is bank-conflict free.
// One CTA per (request, KV split): a producer warp streams 13-token tiles with cp.async.bulk into a 2-stage
// mbarrier pipeline; 3 consumer warps (48 query heads) keep Q and the output accumulator in mma.sync fragments,
// reconstruct the scores through the rank-4 factors, run the online softmax (with optional tanh soft-cap) in
// registers and accumulate W = p*C + p'*D against Bv. Partial (o, m, l) per split are merged by a Triton combine.
// Built JIT by vllm/model_executor/layers/mamba/dragon/tpa_factor.py.
// TPA-factorized decode attention, v2: register-resident mma.sync pipeline (sm_90).
// Cache row (bf16, 1216): Bk[4x128] | Bv[4x128] | Ck[12x4] | Dk[12x4] | Cv[12x4] | Dv[12x4]
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr int HQ = 48, HQP = 64, HKV = 12, R = 4, D = 128;
constexpr int T = 13, NTOK = T + 1, NCOL = NTOK * R;   // 56 real columns (tokidx 0 = t0-1)
constexpr int NT8 = 7;                                   // n8 tiles covering 56 columns exactly (K side)
constexpr int KS = 4;                                    // k16 steps covering 64 >= 56 (V side)
constexpr int TOKBLK = 2 * R * 64;                       // per-token smem elems: [half][r][64], two swizzled 512B boxes
constexpr int ROWP = 608;                                // per-plane row: [B 4x128 | C 12x4 | D 12x4] bf16 = 1216 B
constexpr int OFF_C = 512, OFF_D = 560;
#ifndef TPA_NBUF
#define TPA_NBUF 2
#endif
#ifndef TPA_L2PROMO
#define TPA_L2PROMO CU_TENSOR_MAP_L2_PROMOTION_L2_128B
#endif
constexpr int THREADS = 128, NCONS = 96, NBUF = TPA_NBUF;   // 3 consumer warps (48 heads) + 1 producer warp
constexpr int CHUNKS_PER_TOK = 64 + 64 + 24;             // 16B chunks: Bk, Bv, 4x48 coefs

struct __align__(1024) Smem {
  __nv_bfloat16 rowsK[NBUF][NTOK * ROWP];  // raw K-plane rows [tok][Bk 4x128 | Ck | Dk], globally pre-swizzled (chunk c of r-row at (c&8)|((c&7)^r))
  __nv_bfloat16 rowsV[NBUF][NTOK * ROWP];  // raw V-plane rows [tok][Bv 4x128 | Cv | Dv]
  __nv_bfloat16 zero[512];                 // zero block for pad rows (tok >= NTOK)
  unsigned long long full[NBUF];
  unsigned long long empty[NBUF];
  int blocks[64];      // block ids covering [start-1, stop) for this split
  int blk_first;       // block index of token start-1 (clamped)
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }


__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned parity) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void tma_tensor3d_g2s(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
// swizzled smem address (bytes offset) of 16B chunk `chunk` (0..15 over d) of factor row (tok, r)
// Byte offset of 16B chunk `chunk` (0..15 over d) of factor row (tok, r) inside a raw-row plane buffer.
// Rows are 1216 B (= 64 mod 128), so consecutive tokens alternate 128B phase; XOR-ing the chunk with r inside each
// 128B half then spreads the eight ldmatrix rows (two tokens x four r) over eight distinct 16B bank groups.
__device__ __forceinline__ int fac_off(int tok, int r, int chunk) {
  return tok * (ROWP * 2) + r * 256 + ((chunk & 8) << 4) + (((chunk & 7) ^ r) << 4);
}
__device__ __forceinline__ void tma_tensor2d_g2s(void* dst, const CUtensorMap* tmap, int c0, int c1, void* mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s(void* dst, const void* src, unsigned bytes, void* mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(mbar)) : "memory");
}

__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" :: "n"(NCONS) : "memory"); }
__device__ __forceinline__ void mbar_arrive(void* mbar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* smem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// element offset of token t's row inside a plane: blocks are `bstride` elements apart, rows ROWP within a block
__device__ __forceinline__ size_t row_of(const Smem& S, int t, int BS, long long bstride) { return (size_t)S.blocks[t / BS - S.blk_first] * bstride + (size_t)(t % BS) * ROWP; }

__device__ __forceinline__ void producer_issue(Smem& S, int buf, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl,
                                               int BS, long long bstride, int seqlen, int t0, int lane) {
  char* rk = reinterpret_cast<char*>(S.rowsK[buf]);
  char* rv = reinterpret_cast<char*>(S.rowsV[buf]);
  constexpr uint32_t RB = ROWP * 2;
  const bool wide = (t0 >= 1) && (t0 + T <= seqlen);   // all NTOK tokens valid
  if (wide) {
    if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], 2 * NTOK * RB);
    __syncwarp();
    const int n1 = min(T, BS - (t0 % BS));               // rows of the main range inside the first block
    if (lane < 6) {
      const size_t gp = row_of(S, t0 - 1, BS, bstride), g0 = row_of(S, t0, BS, bstride);
      switch (lane) {
        case 0: tma_bulk_g2s(rk, kpl + gp, RB, &S.full[buf]); break;
        case 1: tma_bulk_g2s(rv, vpl + gp, RB, &S.full[buf]); break;
        case 2: tma_bulk_g2s(rk + RB, kpl + g0, n1 * RB, &S.full[buf]); break;
        case 3: tma_bulk_g2s(rv + RB, vpl + g0, n1 * RB, &S.full[buf]); break;
        case 4: if (n1 < T) tma_bulk_g2s(rk + (1 + n1) * RB, kpl + row_of(S, t0 + n1, BS, bstride), (T - n1) * RB, &S.full[buf]); break;
        case 5: if (n1 < T) tma_bulk_g2s(rv + (1 + n1) * RB, vpl + row_of(S, t0 + n1, BS, bstride), (T - n1) * RB, &S.full[buf]); break;
      }
    }
    return;
  }
  const int tok = lane, t = t0 - 1 + tok;
  const bool mine = lane < NTOK, valid = mine && (t >= 0) && (t < seqlen);
  const unsigned nvalid = __popc(__ballot_sync(0xffffffffu, valid));
  char* dk = rk + tok * RB; char* dv = rv + tok * RB;
  if (mine && !valid) {
    // Invalid tokens (before the sequence start / past its end) must read as zeros: the buffer still holds the
    // previous tile (or whatever the last CTA on this SM left in shared memory, NaNs included, and 0 * NaN = NaN
    // in the V mma). The zero stores are generic-proxy writes, so they must be fenced and complete before the
    // mbarrier arrive below releases the tile to the consumers.
    for (int c = 0; c < (int)(RB / 16); ++c) { *reinterpret_cast<uint4*>(dk + c * 16) = make_uint4(0, 0, 0, 0); *reinterpret_cast<uint4*>(dv + c * 16) = make_uint4(0, 0, 0, 0); }
    __threadfence_block();
  }
  __syncwarp();
  if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], 2 * nvalid * RB);
  __syncwarp();
  if (mine && valid) {
    const size_t grow = row_of(S, t, BS, bstride);
    tma_bulk_g2s(dk, kpl + grow, RB, &S.full[buf]);
    tma_bulk_g2s(dv, vpl + grow, RB, &S.full[buf]);
  }
}

__device__ unsigned long long g_phase[5];
__global__ void __launch_bounds__(THREADS, 3)
tpa_factor_decode_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl,
                     const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                     float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                     float sm_scale, float softcap, const int* __restrict__ tok_per_split_ptr, int BS, long long bstride) {
  const int tok_per_split = *tok_per_split_ptr;
  const float inv_cap = softcap > 0.f ? 1.f / softcap : 0.f;
  extern __shared__ __align__(1024) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qd = lane & 3, rl = lane >> 2;
  const int row0 = warp * 16 + rl, row1 = row0 + 8;             // this thread's two head rows
  const int g0 = min(row0, HQ - 1) / (HQ / HKV), g1 = min(row1, HQ - 1) / (HQ / HKV);
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);
  if (stop <= start) {
    // Empty split (static grid, dynamic split length): publish m = -inf / l = 0 only; the combine ignores o.
    if (tid < HQP) { const size_t base = ((size_t)b * nsplit + sp) * HQP; part_m[base + tid] = -CUDART_INF_F; part_l[base + tid] = 0.f; }
    return;
  }

  // Q fragments (A operand, 8 k16 steps), padded heads zero
  uint32_t qa[D / 16][4];
#ifndef TPA_NOPRO
  {
    const __nv_bfloat16* qb = q + (size_t)b * HQ * D;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      int col = ks * 16 + 2 * qd;
      qa[ks][0] = row0 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row0 * D + col) : 0u;
      qa[ks][1] = row1 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row1 * D + col) : 0u;
      qa[ks][2] = row0 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row0 * D + col + 8) : 0u;
      qa[ks][3] = row1 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row1 * D + col + 8) : 0u;
    }
  }
#else
  for (int k = 0; k < D / 16; ++k) qa[k][0] = qa[k][1] = qa[k][2] = qa[k][3] = 0u;
#endif
  for (int i = tid; i < 512 / 8; i += THREADS) *reinterpret_cast<uint4*>(S.zero + i * 8) = make_uint4(0, 0, 0, 0);
  float acc[D / 8][4];
#pragma unroll
  for (int nt = 0; nt < D / 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }
  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;

  {  // block ids for this split (tokens start-1 .. stop-1), so tile issue never waits on a global lookup
    const int bfirst = max(start - 1, 0) / BS, blast = (max(stop, 1) - 1) / BS;
    if (tid == 0) S.blk_first = bfirst;
    for (int i = tid; i <= blast - bfirst && i < 64; i += THREADS) S.blocks[i] = bt[b * bt_stride + bfirst + i];
  }
  if (tid == 0) {
    for (int i = 0; i < NBUF; ++i) { mbar_init(&S.full[i], 1); mbar_init(&S.empty[i], NCONS / 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntiles = (stop > start) ? (stop - start + T - 1) / T : 0;
#ifdef TPA_MICRO3_W0
  const int pwarp = 0;
#else
  const int pwarp = NCONS / 32;
#endif
  if (warp == pwarp) {
    // ===== producer warp: keep both buffers full, wait for consumers to release before refilling
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it % NBUF;
#if defined(TPA_MICRO)
      if (it >= NBUF) mbar_wait(&S.full[buf], ((it / NBUF) + 1) & 1);   // microbench structure: wait own previous fill
#elif !defined(TPA_FREERUN)
      if (it >= NBUF) mbar_wait(&S.empty[buf], ((it / NBUF) + 1) & 1);
#endif
      producer_issue(S, buf, kpl, vpl, BS, bstride, seqlen, start + it * T, lane);
    }
#ifdef TPA_MICRO
    for (int it = max(0, ntiles - NBUF); it < ntiles; ++it) mbar_wait(&S.full[it % NBUF], (it / NBUF) & 1);
#endif
    return;
  }
#if defined(TPA_MICRO) || defined(TPA_MICRO3)
#ifdef TPA_MICRO3_KEEP
  for (int it = max(0, ntiles - NBUF); it < ntiles; ++it) mbar_wait(&S.full[it % NBUF], (it / NBUF) & 1);   // stay resident until the end
#endif
  return;   // consumers idle
#endif

  // Hoisted ldmatrix address terms (tile-invariant). Row (tok, rr) lives at tok*ROWB + rr*256; its 16B chunk c is at
  // ((c&8)<<4) | (((c&7) ^ f)<<4) with f = (par<<2)|rr, par = parity of token t = (li>>2)^1 for every tok this lane touches.
  constexpr int ROWB = ROWP * 2;
  const int li_ = lane & 7, mi_ = lane >> 3;
  const int fK = li_ & 3;
  const int laneK = (li_ >> 2) * ROWB + (li_ & 3) * 256;              // K side: tok = 2nt + (li>>2)
  const int laneV = (mi_ & 1) * 2 * ROWB + laneK;                      // V side: tok = 4ks + 2(mi&1) + (li>>2)
  int swzK[D / 32], swzV[D / 16];
#pragma unroll
  for (int p = 0; p < D / 32; ++p) { const int c = 4 * p + mi_; swzK[p] = ((c & 8) << 4) | (((c & 7) ^ fK) << 4); }
#pragma unroll
  for (int np = 0; np < D / 16; ++np) { const int c = 2 * np + (mi_ >> 1); swzV[np] = ((c & 8) << 4) | (((c & 7) ^ fK) << 4); }
  long long ph0 = 0, ph1 = 0, ph2 = 0, ph3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  for (int it = 0; it < ntiles; ++it) {
    const int buf = it % NBUF, t0 = start + it * T;
#ifdef TPA_TIMING
    c0 = clock64();
#endif
#ifndef TPA_FREERUN
    mbar_wait(&S.full[buf], (it / NBUF) & 1);
#endif
#ifdef TPA_TIMING
    c1 = clock64(); ph0 += c1 - c0;
#endif
    const __nv_bfloat16* Bk = S.rowsK[buf]; const __nv_bfloat16* Bv = S.rowsV[buf];
    const __nv_bfloat16* CfK = S.rowsK[buf] + OFF_C; const __nv_bfloat16* CfV = S.rowsV[buf] + OFF_C;

#ifdef TPA_NOCOMPUTE
    if (it < 0) {
#endif
    // ---- qB = Q * Bk^T : 9 n8 tiles (cols = tokidx*4 + r). Per k16-pair: 9 ldmatrix back-to-back, then 18 mma
    // on distinct accumulators (no dependent chains; volatile asm keeps source order so this is the schedule).
    float qB[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) { qB[nt][0] = qB[nt][1] = qB[nt][2] = qB[nt][3] = 0.f; }
    {
      const char* BkB = reinterpret_cast<const char*>(Bk) + laneK;
      const char* Z = reinterpret_cast<const char*>(S.zero);
#pragma unroll
      for (int ks = 0; ks < D / 16; ks += 2) {
        uint32_t bfr[NT8][4];
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) {
          const bool z = (2 * nt + (li_ >> 2)) >= NTOK;
          ldmatrix_x4(bfr[nt], z ? Z : BkB + nt * (2 * ROWB) + swzK[ks >> 1]);
        }
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma16816(qB[nt], qa[ks], bfr[nt]);
#pragma unroll
        for (int nt = 0; nt < NT8; ++nt) mma16816(qB[nt], qa[ks + 1], bfr[nt] + 2);
      }
    }
#ifdef TPA_TIMING
    c2 = clock64(); ph1 += c2 - c1;
#endif
    // ---- scores. This thread holds cols {2qd, 2qd+1} of each tile: tokidx k = 2nt + (qd>>1), ranks 2(qd&1)+{0,1}
    float sc0[NT8], sc1[NT8], ad0[NT8], ad1[NT8];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), rp = 2 * (qd & 1);
      float ac0 = 0.f, ac1 = 0.f, d0 = 0.f, d1 = 0.f;
      if (k >= 1 && k <= T) {           // own-token C of token t=k-1 (smem tok index k)
        const __nv_bfloat16* Ck = CfK + k * ROWP;
        float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g0 * R + rp));
        float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g1 * R + rp));
        ac0 = c0.x * qB[nt][0] + c0.y * qB[nt][1];
        ac1 = c1.x * qB[nt][2] + c1.y * qB[nt][3];
      }
      if (k <= T - 1) {                 // D of token t=k (smem tok index k+1) applied to B of tokidx k
        const __nv_bfloat16* Dk = CfK + (k + 1) * ROWP + (OFF_D - OFF_C);
        float2 e0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dk + g0 * R + rp));
        float2 e1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dk + g1 * R + rp));
        d0 = e0.x * qB[nt][0] + e0.y * qB[nt][1];
        d1 = e1.x * qB[nt][2] + e1.y * qB[nt][3];
      }
      ac0 += __shfl_xor_sync(0xffffffffu, ac0, 1); ac1 += __shfl_xor_sync(0xffffffffu, ac1, 1);
      d0 += __shfl_xor_sync(0xffffffffu, d0, 1); d1 += __shfl_xor_sync(0xffffffffu, d1, 1);
      sc0[nt] = ac0; sc1[nt] = ac1; ad0[nt] = d0; ad1[nt] = d1;
    }
    // s(k) = Ac(k) + Ad(k-1); Ad(k-1) lives in lanes qd^2: same tile when k odd, previous tile when k even
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      float same0 = __shfl_xor_sync(0xffffffffu, ad0[nt], 2), same1 = __shfl_xor_sync(0xffffffffu, ad1[nt], 2);
      float prev0 = (nt > 0) ? __shfl_xor_sync(0xffffffffu, ad0[nt > 0 ? nt - 1 : 0], 2) : 0.f;
      float prev1 = (nt > 0) ? __shfl_xor_sync(0xffffffffu, ad1[nt > 0 ? nt - 1 : 0], 2) : 0.f;
      const bool odd = (qd >> 1) == 1;
      sc0[nt] += odd ? same0 : prev0; sc1[nt] += odd ? same1 : prev1;
    }
    // scale + validity mask (token t = k-1)
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), t = k - 1;
      bool ok = (k >= 1) && (k <= T) && (t0 + t < seqlen);
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
    float ls0 = 0.f, ls1 = 0.f, pr0[NT8], pr1[NT8];   // pr[nt] = p of token t = 2nt + (qd>>1) - 1 (0 when masked)
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      pr0[nt] = (sc0[nt] == -CUDART_INF_F) ? 0.f : __expf(sc0[nt] - mx0);
      pr1[nt] = (sc1[nt] == -CUDART_INF_F) ? 0.f : __expf(sc1[nt] - mx1);
      ls0 += pr0[nt]; ls1 += pr1[nt];
    }
    // lanes qd and qd^1 hold the same token: reduce over the two distinct token lanes only
    ls0 += __shfl_xor_sync(0xffffffffu, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffffu, ls1, 2);
    l0 = l0 * al0 + ls0; l1 = l1 * al1 + ls1; m0 = mx0; m1 = mx1;
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) { acc[nt][0] *= al0; acc[nt][1] *= al0; acc[nt][2] *= al1; acc[nt][3] *= al1; }
#ifdef TPA_TIMING
    c3 = clock64(); ph2 += c3 - c2;
#endif

    // ---- values: acc += W * Bv,  W[row][n] = p[row][tokidx-1] Cv[tokidx-1] + p[row][tokidx] Dv[tokidx]
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t a[4];
      {
        // a0:(row0, n = ks*16+2qd+{0,1}) a1:(row1, same) a2:(row0, n+8) a3:(row1, n+8)
        // n -> tokidx = 4ks + (qd>>1) + 2*hi, r = 2(qd&1)+{0,1}
        const int rp = 2 * (qd & 1);
        float w[8];
        const bool hiP = (qd >> 1) == 1;   // this lane's token parity
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
          const int tokidx = 4 * ks + (qd >> 1) + 2 * hi, ntA = 2 * ks + hi;          // p[tokidx-1] = own pr[ntA]
          const int ntB0 = ntA < NT8 ? ntA : NT8 - 1, ntB1 = ntA + 1 < NT8 ? ntA + 1 : NT8 - 1;
          // p[tokidx] has the other parity: lane qd^2 holds it at nt = ntA (if that lane is odd) or ntA+1 (if even)
          const float sb0 = hiP ? pr0[ntB0] : pr0[ntB1], sb1 = hiP ? pr1[ntB0] : pr1[ntB1];
          const float pb0 = __shfl_xor_sync(0xffffffffu, sb0, 2), pb1 = __shfl_xor_sync(0xffffffffu, sb1, 2);
          const float pa0 = pr0[ntB0], pa1 = pr1[ntB0];
          float2 v0 = make_float2(0.f, 0.f), v1 = make_float2(0.f, 0.f);
          if (tokidx < NTOK) {
            if (tokidx >= 1) {
              const __nv_bfloat16* Cv = CfV + tokidx * ROWP;
              float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g0 * R + rp));
              float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g1 * R + rp));
              v0.x += pa0 * c0.x; v0.y += pa0 * c0.y; v1.x += pa1 * c1.x; v1.y += pa1 * c1.y;
            }
            if (tokidx <= T - 1) {
              const __nv_bfloat16* Dv = CfV + (tokidx + 1) * ROWP + (OFF_D - OFF_C);
              float2 d0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g0 * R + rp));
              float2 d1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g1 * R + rp));
              v0.x += pb0 * d0.x; v0.y += pb0 * d0.y; v1.x += pb1 * d1.x; v1.y += pb1 * d1.y;
            }
          }
          w[4 * hi + 0] = v0.x; w[4 * hi + 1] = v0.y; w[4 * hi + 2] = v1.x; w[4 * hi + 3] = v1.y;
        }
        a[0] = pack_bf16(w[0], w[1]); a[1] = pack_bf16(w[2], w[3]); a[2] = pack_bf16(w[4], w[5]); a[3] = pack_bf16(w[6], w[7]);
      }
      {
        uint32_t bfr[D / 16][4];
        const bool z = (4 * ks + 2 * (mi_ & 1) + (li_ >> 2)) >= NTOK;
        const char* rowp = reinterpret_cast<const char*>(Bv) + ks * (4 * ROWB) + laneV;
        const char* Z = reinterpret_cast<const char*>(S.zero);
#pragma unroll
        for (int np = 0; np < D / 16; ++np) ldmatrix_x4_trans(bfr[np], z ? Z : rowp + swzV[np]);
#pragma unroll
        for (int np = 0; np < D / 16; ++np) { mma16816(acc[2 * np], a, bfr[np]); mma16816(acc[2 * np + 1], a, bfr[np] + 2); }
      }
    }
#ifdef TPA_NOCOMPUTE
    }
#endif
    __syncwarp();   // this warp's reads of the buffer are done
#ifdef TPA_TIMING
    ph3 += clock64() - c3;
#endif
    if (lane == 0) mbar_arrive(&S.empty[buf]);   // release the buffer to the producer
  }
#ifdef TPA_TIMING
  if (lane == 0) { atomicAdd(&g_phase[0], (unsigned long long)ph0); atomicAdd(&g_phase[1], (unsigned long long)ph1); atomicAdd(&g_phase[2], (unsigned long long)ph2); atomicAdd(&g_phase[3], (unsigned long long)ph3); atomicAdd(&g_phase[4], (unsigned long long)ntiles); }
#endif
  // ---- write partials: acc layout c0,c1 -> (row0, cols nt*8 + 2qd, +1); c2,c3 -> row1
  const size_t base = ((size_t)b * nsplit + sp) * HQP;
#pragma unroll
  for (int nt = 0; nt < D / 8; ++nt) {
    float* o0 = part_o + (base + row0) * D + nt * 8 + 2 * qd; float* o1 = part_o + (base + row1) * D + nt * 8 + 2 * qd;
    o0[0] = acc[nt][0]; o0[1] = acc[nt][1]; o1[0] = acc[nt][2]; o1[1] = acc[nt][3];
  }
  if (qd == 0) { part_m[base + row0] = m0; part_m[base + row1] = m1; part_l[base + row0] = l0; part_l[base + row1] = l1; }
}
}  // namespace

void tpa_factor_decode_launch(torch::Tensor q, torch::Tensor kplane, torch::Tensor vplane, torch::Tensor block_table, torch::Tensor seq_lens,
                              torch::Tensor tok_per_split, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                              double sm_scale, double softcap, int64_t block_size) {
  // q: (B, 48, 128) bf16; kplane/vplane: (num_blocks, block_size, 608) bf16 contiguous (globally swizzled factor rows);
  // block_table: (B, max_blocks) int32; seq_lens: (B,) int32; tok_per_split: (1,) int32 device (multiple of 13);
  // part_o: (B, SPLIT, 64, 128) fp32, part_m/part_l: (B, SPLIT, 64) fp32. Grid = B x SPLIT.
  const int B = q.size(0), SPLIT = part_o.size(1);
  // planes: (num_blocks, block_size, ROWP) with contiguous rows inside a block and an arbitrary block stride
  TORCH_CHECK(kplane.dim() == 3 && kplane.size(2) == ROWP && kplane.stride(2) == 1 && kplane.stride(1) == ROWP && q.is_contiguous());
  TORCH_CHECK(vplane.stride(0) == kplane.stride(0) && vplane.stride(1) == ROWP && vplane.stride(2) == 1 && kplane.size(1) == block_size);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 && tok_per_split.dtype() == torch::kInt32);
  static bool attr_set = false;
  const int smem = (int)sizeof(Smem) + 1024;
  if (!attr_set) { cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); attr_set = true; }
  tpa_factor_decode_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kplane.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(vplane.data_ptr()), block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (float)softcap, tok_per_split.data_ptr<int>(), (int)block_size, (long long)kplane.stride(0));
}
int64_t tpa_factor_smem_bytes() { return sizeof(Smem) + 1024; }
int64_t tpa_factor_occupancy() {
  const int smem = (int)sizeof(Smem) + 1024;
  cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int nb = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_factor_decode_kernel, THREADS, smem);
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, tpa_factor_decode_kernel);
  printf("occupancy: %d CTAs/SM (regs %d, static smem %zu, dyn %d, local %zu)\n", nb, a.numRegs, a.sharedSizeBytes, smem, a.localSizeBytes);
  return nb;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &tpa_factor_decode_launch); m.def("smem_bytes", &tpa_factor_smem_bytes); m.def("occupancy", &tpa_factor_occupancy); m.attr("TILE") = T; }
