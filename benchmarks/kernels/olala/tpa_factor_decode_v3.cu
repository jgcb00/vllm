// TPA-factorized decode attention, v2: register-resident mma.sync pipeline (sm_90).
// Cache row (bf16, 1216): Bk[4x128] | Bv[4x128] | Ck[12x4] | Dk[12x4] | Cv[12x4] | Dv[12x4]
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>

namespace {
constexpr int HQ = 48, HQP = 64, HKV = 12, R = 4, D = 128;
constexpr int T = 16, NTOK = T + 1, NCOL = NTOK * R;   // 68 real columns (tokidx 0 = t0-1)
constexpr int NT8 = 9;                                   // n8 tiles covering 72 >= 68 columns (K side)
constexpr int KS = 5;                                    // k16 steps covering 80 >= 68 (V side)
constexpr int KROWS = KS * 16;                           // 80 smem rows for Bk/Bv
constexpr int LD = D + 8;                                // padded row stride (bank-conflict free ldmatrix/lds)
constexpr int ROW = 1216, OFF_BK = 0, OFF_BV = 512, OFF_C0 = 1024;
constexpr int THREADS = 160, NCONS = 128, NBUF = 2;   // 4 consumer warps + 1 producer warp
constexpr int CHUNKS_PER_TOK = 64 + 64 + 24;             // 16B chunks: Bk, Bv, 4x48 coefs

struct __align__(128) Smem {
  __nv_bfloat16 Bk[NBUF][KROWS * LD];
  __nv_bfloat16 Bv[NBUF][KROWS * LD];
  __nv_bfloat16 coef[NBUF][NTOK * 4 * 48];
  float p[T * HQP];   // [t][row]
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
  // lanes 0..NTOK-1 own one token each; expected bytes = valid tokens * 2432
  const int tok = lane, t = t0 - 1 + tok;
  const bool mine = lane < NTOK, valid = mine && (t >= 0) && (t < seqlen);
  const unsigned nvalid = __popc(__ballot_sync(0xffffffffu, valid));
  if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], nvalid * (R * D + R * D + 4 * 48) * 2);
  __syncwarp();
  if (!mine) return;
  __nv_bfloat16* bk = S.Bk[buf] + tok * R * LD;
  __nv_bfloat16* bv = S.Bv[buf] + tok * R * LD;
  __nv_bfloat16* cf = S.coef[buf] + tok * 4 * 48;
  if (valid) {
    const int blk = S.blocks[t / BS - S.blk_first];
    const __nv_bfloat16* row = cache + ((size_t)blk * BS + (t % BS)) * ROW;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      tma_bulk_g2s(bk + r * LD, row + OFF_BK + r * D, D * 2, &S.full[buf]);
      tma_bulk_g2s(bv + r * LD, row + OFF_BV + r * D, D * 2, &S.full[buf]);
    }
    tma_bulk_g2s(cf, row + OFF_C0, 4 * 48 * 2, &S.full[buf]);
  } else {
    for (int r = 0; r < R; ++r) for (int c8 = 0; c8 < D / 8; ++c8) {
      *reinterpret_cast<uint4*>(bk + r * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(bv + r * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
    for (int c8 = 0; c8 < 4 * 48 / 8; ++c8) *reinterpret_cast<uint4*>(cf + c8 * 8) = make_uint4(0, 0, 0, 0);
    __threadfence_block();
  }
}

__global__ void __launch_bounds__(THREADS, 2)
tpa_decode_v3_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache,
                     const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                     float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                     float sm_scale, int tok_per_split, int BS) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qd = lane & 3, rl = lane >> 2;
  const int row0 = warp * 16 + rl, row1 = row0 + 8;             // this thread's two head rows
  const int g0 = min(row0, HQ - 1) / (HQ / HKV), g1 = min(row1, HQ - 1) / (HQ / HKV);
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);

  // Q fragments (A operand, 8 k16 steps), padded heads zero
  uint32_t qa[D / 16][4];
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
  // zero the pad rows (68..79) of both buffers once
  for (int i = tid; i < NBUF * (KROWS - NCOL) * D / 8; i += THREADS) {
    int buf = i / ((KROWS - NCOL) * D / 8), j = i % ((KROWS - NCOL) * D / 8);
    int row = NCOL + j / (D / 8), c8 = j % (D / 8);
    *reinterpret_cast<uint4*>(S.Bk[buf] + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(S.Bv[buf] + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
  }
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
    mbar_init(&S.full[0], 1); mbar_init(&S.full[1], 1); mbar_init(&S.empty[0], NCONS / 32); mbar_init(&S.empty[1], NCONS / 32);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntiles = (stop > start) ? (stop - start + T - 1) / T : 0;
  if (warp == NCONS / 32) {
    // ===== producer warp: keep both buffers full, wait for consumers to release before refilling
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it & 1;
      if (it >= 2) mbar_wait(&S.empty[buf], ((it >> 1) + 1) & 1);
      producer_issue(S, buf, cache, BS, seqlen, start + it * T, lane);
    }
    return;
  }

  for (int it = 0; it < ntiles; ++it) {
    const int buf = it & 1, t0 = start + it * T;
    mbar_wait(&S.full[buf], (it >> 1) & 1);
    consumer_sync();
    const __nv_bfloat16* Bk = S.Bk[buf]; const __nv_bfloat16* Bv = S.Bv[buf]; const __nv_bfloat16* Cf = S.coef[buf];

#ifdef TPA_NOCOMPUTE
    if (it < 0) {
#endif
    // ---- qB = Q * Bk^T : 9 n8 tiles (cols = tokidx*4 + r), fragments in registers
    float qB[NT8][4];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      qB[nt][0] = qB[nt][1] = qB[nt][2] = qB[nt][3] = 0.f;
      // ldmatrix x4: lanes 0-7 -> rows n0..7 at k0, 8-15 at k0+8, 16-23 at k0+16, 24-31 at k0+24 => b0,b1 for ks and ks+1
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int ks = 0; ks < D / 16; ks += 2) {
        uint32_t bfr[4];
        ldmatrix_x4(bfr, Bk + (nt * 8 + li) * LD + ks * 16 + mi * 8);
        mma16816(qB[nt], qa[ks], bfr);
        mma16816(qB[nt], qa[ks + 1], bfr + 2);
      }
    }
    // ---- scores. This thread holds cols {2qd, 2qd+1} of each tile: tokidx k = 2nt + (qd>>1), ranks 2(qd&1)+{0,1}
    float sc0[NT8], sc1[NT8], ad0[NT8], ad1[NT8];
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), rp = 2 * (qd & 1);
      float ac0 = 0.f, ac1 = 0.f, d0 = 0.f, d1 = 0.f;
      if (k >= 1 && k <= T) {           // own-token C of token t=k-1 (smem tok index k)
        const __nv_bfloat16* Ck = Cf + (k * 4 + 0) * 48;
        float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g0 * R + rp));
        float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ck + g1 * R + rp));
        ac0 = c0.x * qB[nt][0] + c0.y * qB[nt][1];
        ac1 = c1.x * qB[nt][2] + c1.y * qB[nt][3];
      }
      if (k <= T - 1) {                 // D of token t=k (smem tok index k+1) applied to B of tokidx k
        const __nv_bfloat16* Dk = Cf + ((k + 1) * 4 + 1) * 48;
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
    float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      const int k = 2 * nt + (qd >> 1), t = k - 1;
      float p0 = (sc0[nt] == -CUDART_INF_F) ? 0.f : __expf(sc0[nt] - mx0);
      float p1 = (sc1[nt] == -CUDART_INF_F) ? 0.f : __expf(sc1[nt] - mx1);
      ls0 += p0; ls1 += p1;
      if (k >= 1 && k <= T && (qd & 1) == 0) { S.p[t * HQP + row0] = p0; S.p[t * HQP + row1] = p1; }
    }
    // lanes qd and qd^1 hold the same token: reduce over the two distinct token lanes only
    ls0 += __shfl_xor_sync(0xffffffffu, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffffu, ls1, 2);
    l0 = l0 * al0 + ls0; l1 = l1 * al1 + ls1; m0 = mx0; m1 = mx1;
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) { acc[nt][0] *= al0; acc[nt][1] *= al0; acc[nt][2] *= al1; acc[nt][3] *= al1; }
    consumer_sync();   // p visible to all consumer warps

    // ---- values: acc += W * Bv,  W[row][n] = p[row][tokidx-1] Cv[tokidx-1] + p[row][tokidx] Dv[tokidx]
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t a[4];
      {
        // a0:(row0, n = ks*16+2qd+{0,1}) a1:(row1, same) a2:(row0, n+8) a3:(row1, n+8)
        // n -> tokidx = 4ks + (qd>>1) + 2*hi, r = 2(qd&1)+{0,1}
        const int rp = 2 * (qd & 1);
        float w[8];
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
          const int tokidx = 4 * ks + (qd >> 1) + 2 * hi;
          float2 v0 = make_float2(0.f, 0.f), v1 = make_float2(0.f, 0.f);
          if (tokidx < NTOK) {
            if (tokidx >= 1) {
              const float pa0 = S.p[(tokidx - 1) * HQP + row0], pa1 = S.p[(tokidx - 1) * HQP + row1];
              const __nv_bfloat16* Cv = Cf + (tokidx * 4 + 2) * 48;
              float2 c0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g0 * R + rp));
              float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Cv + g1 * R + rp));
              v0.x += pa0 * c0.x; v0.y += pa0 * c0.y; v1.x += pa1 * c1.x; v1.y += pa1 * c1.y;
            }
            if (tokidx <= T - 1) {
              const float pb0 = S.p[tokidx * HQP + row0], pb1 = S.p[tokidx * HQP + row1];
              const __nv_bfloat16* Dv = Cf + ((tokidx + 1) * 4 + 3) * 48;
              float2 d0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g0 * R + rp));
              float2 d1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Dv + g1 * R + rp));
              v0.x += pb0 * d0.x; v0.y += pb0 * d0.y; v1.x += pb1 * d1.x; v1.y += pb1 * d1.y;
            }
          }
          w[4 * hi + 0] = v0.x; w[4 * hi + 1] = v0.y; w[4 * hi + 2] = v1.x; w[4 * hi + 3] = v1.y;
        }
        a[0] = pack_bf16(w[0], w[1]); a[1] = pack_bf16(w[2], w[3]); a[2] = pack_bf16(w[4], w[5]); a[3] = pack_bf16(w[6], w[7]);
      }
#pragma unroll
      for (int nt = 0; nt < D / 8; nt += 2) {
        uint32_t bfr[4];
        // ldmatrix x4 trans: matrices (k0..7, n0), (k8..15, n0), (k0..7, n0+8), (k8..15, n0+8)
        const int mi = lane >> 3, li = lane & 7;
        const __nv_bfloat16* addr = Bv + (ks * 16 + (mi & 1) * 8 + li) * LD + nt * 8 + (mi >> 1) * 8;
        ldmatrix_x4_trans(bfr, addr);
        mma16816(acc[nt], a, bfr);
        mma16816(acc[nt + 1], a, bfr + 2);
      }
    }
#ifdef TPA_NOCOMPUTE
    }
#endif
    consumer_sync();   // all consumer reads of this buffer (and p) done
    if (lane == 0) mbar_arrive(&S.empty[buf]);   // release the buffer to the producer
  }
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

void tpa_decode_v3_launch(torch::Tensor q, torch::Tensor cache, torch::Tensor block_table, torch::Tensor seq_lens,
                          torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                          double sm_scale, int64_t tok_per_split, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1);
  static bool attr_set = false;
  if (!attr_set) { cudaFuncSetAttribute(tpa_decode_v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem)); attr_set = true; }
  tpa_decode_v3_kernel<<<dim3(B, SPLIT), THREADS, sizeof(Smem), c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cache.data_ptr()),
      block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (int)tok_per_split, (int)block_size);
}
int64_t tpa_v3_smem_bytes() { return sizeof(Smem); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &tpa_decode_v3_launch); m.def("smem_bytes", &tpa_v3_smem_bytes); }
