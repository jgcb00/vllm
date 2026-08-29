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
constexpr int T = 12, NTOK = T + 1, NCOL = NTOK * R;   // 52 real columns (tokidx 0 = t0-1); 144 = 12 x 12
constexpr int NT8 = 7;                                   // n8 tiles covering 56 >= 52 columns (K side)
constexpr int KS = 4;                                    // k16 steps covering 64 >= 52 (V side)
constexpr int TOKBLK = 2 * R * 64;                       // per-token smem elems: [half][r][64], two swizzled 512B boxes
constexpr int ROW = 1216, OFF_BK = 0, OFF_BV = 512, OFF_C0 = 1024;
#ifndef TPA_NBUF
#define TPA_NBUF 2
#endif
#ifndef TPA_L2PROMO
#define TPA_L2PROMO CU_TENSOR_MAP_L2_PROMOTION_L2_128B
#endif
constexpr int THREADS = 128, NCONS = 96, NBUF = TPA_NBUF;   // 3 consumer warps (48 heads) + 1 producer warp
constexpr int CHUNKS_PER_TOK = 64 + 64 + 24;             // 16B chunks: Bk, Bv, 4x48 coefs

struct __align__(1024) Smem {
  __nv_bfloat16 rows[NBUF][NTOK * ROW];    // raw cache rows [tok][Bk 4x128 | Bv 4x128 | coefs], globally pre-swizzled
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
// Byte offset of 16B chunk `chunk` (0..15 over d) of factor row (tok, r) inside a raw-row buffer.
// Global rows are stored with chunk c at position (c & 8) | ((c & 7) ^ f), f = ((t & 1) << 2) | r, which makes the
// eight consecutive ldmatrix rows (two tokens x four r) hit eight distinct 16B bank groups.
__device__ __forceinline__ int fac_off(int tok, int r, int chunk) {
  const int par = (tok == 0) ? 1 : ((tok - 1) & 1);
  return tok * (ROW * 2) + r * 256 + ((chunk & 8) << 4) + ((((chunk & 7) ^ ((par << 2) | r))) << 4);
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

__device__ __forceinline__ void producer_issue(Smem& S, int buf, const __nv_bfloat16* __restrict__ cache,
                                               int BS, int seqlen, int t0, int lane) {
  char* rows = reinterpret_cast<char*>(S.rows[buf]);
  constexpr uint32_t RB = ROW * 2;
#ifdef TPA_NARROW_ONLY
  const bool wide = false;
#else
  const bool wide = (t0 >= 1) && (t0 + T <= seqlen);   // all 17 tokens valid, 16 main tokens contiguous in one block
#endif
  if (wide) {
    if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], NTOK * RB);
    __syncwarp();
    if (lane == 0) {
      const int tp = t0 - 1, gp = S.blocks[tp / BS - S.blk_first] * BS + (tp % BS);
      tma_bulk_g2s(rows, cache + (size_t)gp * ROW, RB, &S.full[buf]);
    } else if (lane == 1) {
      const int g0 = S.blocks[t0 / BS - S.blk_first] * BS + (t0 % BS);
      tma_bulk_g2s(rows + RB, cache + (size_t)g0 * ROW, T * RB, &S.full[buf]);
    }
    return;
  }
  const int tok = lane, t = t0 - 1 + tok;
  const bool mine = lane < NTOK, valid = mine && (t >= 0) && (t < seqlen);
  const unsigned nvalid = __popc(__ballot_sync(0xffffffffu, valid));
  if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], nvalid * RB);
  __syncwarp();
  if (!mine) return;
  char* dst = rows + tok * RB;
  if (valid) {
    const int grow = S.blocks[t / BS - S.blk_first] * BS + (t % BS);
    tma_bulk_g2s(dst, cache + (size_t)grow * ROW, RB, &S.full[buf]);
  } else {
    for (int c = 0; c < (int)(RB / 16); ++c) *reinterpret_cast<uint4*>(dst + c * 16) = make_uint4(0, 0, 0, 0);
    __threadfence_block();
  }
}

__device__ unsigned long long g_phase[5];
__global__ void __launch_bounds__(THREADS, 3)
tpa_decode_v9_kernel(                     const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache,
                     const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                     float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                     float sm_scale, int tok_per_split, int BS) {
  extern __shared__ __align__(1024) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qd = lane & 3, rl = lane >> 2;
  const int row0 = warp * 16 + rl, row1 = row0 + 8;             // this thread's two head rows
  const int g0 = min(row0, HQ - 1) / (HQ / HKV), g1 = min(row1, HQ - 1) / (HQ / HKV);
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);

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
#ifdef TPA_MICRO3
    // verbatim microbench loop: thread 0 only, linear addressing (harness blocks are contiguous), 2 copies per tile
    if (lane == 0) {
      const char* base = reinterpret_cast<const char*>(cache) + (size_t)(S.blocks[0] * BS + ((start > 0 ? start - 1 : 0) % BS)) * (ROW * 2) + (start > 0 ? ROW * 2 : 0);
      for (int it = 0; it < ntiles; ++it) {
        const int buf = it % NBUF;
        if (it >= NBUF) mbar_wait(&S.full[buf], ((it / NBUF) + 1) & 1);
        mbar_arrive_expect_tx(&S.full[buf], NTOK * ROW * 2);
        tma_bulk_g2s(S.rows[buf], base + (size_t)it * (T * ROW * 2) - ROW * 2, ROW * 2, &S.full[buf]);
        tma_bulk_g2s(reinterpret_cast<char*>(S.rows[buf]) + ROW * 2, base + (size_t)it * (T * ROW * 2), T * ROW * 2, &S.full[buf]);
      }
      for (int it = max(0, ntiles - NBUF); it < ntiles; ++it) mbar_wait(&S.full[it % NBUF], (it / NBUF) & 1);
    }
    return;
#endif
    // ===== producer warp: keep both buffers full, wait for consumers to release before refilling
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it % NBUF;
#if defined(TPA_MICRO)
      if (it >= NBUF) mbar_wait(&S.full[buf], ((it / NBUF) + 1) & 1);   // microbench structure: wait own previous fill
#elif !defined(TPA_FREERUN)
      if (it >= NBUF) mbar_wait(&S.empty[buf], ((it / NBUF) + 1) & 1);
#endif
      producer_issue(S, buf, cache, BS, seqlen, start + it * T, lane);
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
  constexpr int ROWB = ROW * 2;
  const int li_ = lane & 7, mi_ = lane >> 3;
  const int fK = ((((li_ >> 2) ^ 1) << 2) | (li_ & 3));
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
    const __nv_bfloat16* Bk = S.rows[buf]; const __nv_bfloat16* Bv = S.rows[buf] + OFF_BV; const __nv_bfloat16* Cf = S.rows[buf] + OFF_C0;

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
        const __nv_bfloat16* Ck = Cf + k * ROW;
        float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g0 * R + rp));
        float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g1 * R + rp));
        ac0 = c0.x * qB[nt][0] + c0.y * qB[nt][1];
        ac1 = c1.x * qB[nt][2] + c1.y * qB[nt][3];
      }
      if (k <= T - 1) {                 // D of token t=k (smem tok index k+1) applied to B of tokidx k
        const __nv_bfloat16* Dk = Cf + (k + 1) * ROW + 48;
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
      sc0[nt] = ok ? sc0[nt] * sm_scale : -CUDART_INF_F;
      sc1[nt] = ok ? sc1[nt] * sm_scale : -CUDART_INF_F;
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
              const __nv_bfloat16* Cv = Cf + tokidx * ROW + 2 * 48;
              float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g0 * R + rp));
              float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g1 * R + rp));
              v0.x += pa0 * c0.x; v0.y += pa0 * c0.y; v1.x += pa1 * c1.x; v1.y += pa1 * c1.y;
            }
            if (tokidx <= T - 1) {
              const __nv_bfloat16* Dv = Cf + (tokidx + 1) * ROW + 3 * 48;
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

void tpa_decode_v9_launch(torch::Tensor q, torch::Tensor cache, torch::Tensor block_table, torch::Tensor seq_lens,
                          torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                          double sm_scale, int64_t tok_per_split, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1);
  const __nv_bfloat16* cbase = reinterpret_cast<const __nv_bfloat16*>(cache.data_ptr());
  TORCH_CHECK(block_size % T == 0 && tok_per_split % T == 0, "block_size and tok_per_split must be multiples of 12");
  static bool attr_set = false;
  const int smem = (int)sizeof(Smem) + 1024;
  if (!attr_set) { cudaFuncSetAttribute(tpa_decode_v9_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(tpa_decode_v9_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); attr_set = true; }
  tpa_decode_v9_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), cbase, block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (int)tok_per_split, (int)block_size);
}
int64_t tpa_v9_smem_bytes() { return sizeof(Smem) + 1024; }
int64_t tpa_v9_occupancy() {
  const int smem = (int)sizeof(Smem) + 1024;
  cudaFuncSetAttribute(tpa_decode_v9_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(tpa_decode_v9_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int nb = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_decode_v9_kernel, THREADS, smem);
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, tpa_decode_v9_kernel);
  printf("occupancy: %d CTAs/SM (regs %d, static smem %zu, dyn %d, local %zu)\n", nb, a.numRegs, a.sharedSizeBytes, smem, a.localSizeBytes);
  return nb;
}

__device__ __forceinline__ void mb_init(void* m, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(m)), "r"(c)); }
__device__ __forceinline__ void mb_expect(void* m, uint32_t b) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(m)), "r"(b) : "memory"); }
__device__ __forceinline__ void mb_wait(void* m, int ph) { asm volatile("{\n .reg .pred P;\n W:\n mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n @!P bra W;\n}\n" :: "r"(smem_u32(m)), "r"(ph) : "memory"); }
__device__ __forceinline__ void mb_bulk(void* d, const void* s, uint32_t b, void* m) { asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(smem_u32(d)), "l"(s), "r"(b), "r"(smem_u32(m)) : "memory"); }
template <int PREV, int MAIN, int NSTAGE>
__global__ void tma_grid(const char* __restrict__ src, size_t stride_x, size_t stride_y, int n, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * stride_x + (size_t)blockIdx.y * stride_y + PREV;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mb_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f; unsigned uacc = 0u;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mb_wait(&full[st], ((it / NSTAGE) + 1) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      mb_expect(&full[st], STAGE);
      mb_bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      mb_bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mb_wait(&full[st], (it / NSTAGE) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
  atomicAdd(reinterpret_cast<unsigned*>(&sink[blockIdx.y * gridDim.x + blockIdx.x]), uacc); if (acc == 12345.f) sink[0] = acc;
}
template <int PREV, int MAIN, int NSTAGE>
__global__ void __launch_bounds__(160, 2) tma_grid_lb(const char* __restrict__ src, size_t stride_x, size_t stride_y, int n, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * stride_x + (size_t)blockIdx.y * stride_y + PREV;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mb_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f; unsigned uacc = 0u;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mb_wait(&full[st], ((it / NSTAGE) + 1) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      mb_expect(&full[st], STAGE);
      mb_bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      mb_bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mb_wait(&full[st], (it / NSTAGE) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
  atomicAdd(reinterpret_cast<unsigned*>(&sink[blockIdx.y * gridDim.x + blockIdx.x]), uacc); if (acc == 12345.f) sink[0] = acc;
}

void micro_grid(torch::Tensor src, int64_t gx, int64_t gy, int64_t stride_x, int64_t stride_y, int64_t n, int threads, torch::Tensor sink) {
  cudaFuncSetAttribute(tma_grid<2432, 38912, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 82688);
  tma_grid<2432, 38912, 2><<<dim3(gx, gy), threads, 82688, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), stride_x, stride_y, (int)n, sink.data_ptr<float>());
}
torch::Tensor phase_cycles() {
  unsigned long long h[5]; cudaMemcpyFromSymbol(h, g_phase, sizeof(h));
  unsigned long long z[5] = {0, 0, 0, 0, 0}; cudaMemcpyToSymbol(g_phase, z, sizeof(z));
  auto t = torch::empty({5}, torch::kFloat64); for (int i = 0; i < 5; ++i) t[i] = (double)h[i]; return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &tpa_decode_v9_launch); m.def("smem_bytes", &tpa_v9_smem_bytes); m.def("occupancy", &tpa_v9_occupancy); m.def("micro_grid", &micro_grid); m.def("phase_cycles", &phase_cycles); }
