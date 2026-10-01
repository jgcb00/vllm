// Olala Differential-TPA decode attention over the TPA-factorized paged KV cache (sm_90a, wgmma edition).
// Per token the cache stores two 608-element bf16 rows: K plane [Bk (4x128) | Ck (12x4) | Dk (12x4)] and V plane
// [Bv | Cv | Dv] (plain layout; TMA applies the 128B swizzle). Built JIT by tpa_factor.py.
// One CTA per (request, KV split): a producer warp streams 13-token tiles with TMA tensor copies (128B swizzle,
// canonical wgmma layouts) into a 2-stage mbarrier pipeline; one consumer warpgroup (4 warps = 64 rows, 48 heads)
// computes qB = Q Bk^T with wgmma m64n56k16 (A = Q in registers, B K-major), reconstructs the scores through the
// rank-4 coefficients, runs the online softmax in registers and accumulates acc += W Bv with wgmma m64n64k16
// (A = W in registers, B MN-major via the transpose flag).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <unordered_map>

namespace {
constexpr int HQ = 48, HQP = 64, HKV = 12, R = 4, D = 128;
constexpr int T = 13, NTOK = T + 1, NCOL = NTOK * R;   // 56 columns (tokidx 0 = t0-1)
constexpr int NT8 = 7, KS = 4;
constexpr int ROWP = 608, OFF_C = 512, NCOEF = 96;       // plane row [B 4x128 | C 12x4 | D 12x4]; C+D = 96 per token
constexpr int THREADS = 160, NCONS = 128, NBUF = 2;      // 1 consumer warpgroup + 1 producer warp
constexpr int HALF_BYTES = 16 * R * 128;                 // 16 token slots x 4 rows x 128B (rows 56..63 stay zero)
constexpr unsigned TILE_TX = 2u * (NTOK * 2 * 512 + T * NCOEF * 2);   // bytes per full tile (both planes; no coefficients for token 0)

struct __align__(1024) Smem {
  char bk[NBUF][2][HALF_BYTES];   // [buf][k half] K-major SW128: row n = tok*4 + r holds k = half*64 .. +63
  char bv[NBUF][2][HALF_BYTES];   // [buf][n half] MN-major SW128: row k = tok*4 + r holds d = half*64 .. +63
  __nv_bfloat16 ck[NBUF][T * NCOEF + 96];   // tokens 1..13: [tok-1][Ck 48 | Dk 48] (TMA box; 2688 B = 21 x 128 keeps cv aligned)
  __nv_bfloat16 cv[NBUF][T * NCOEF + 96];
  unsigned long long full[NBUF];
  unsigned long long empty[NBUF];
  int blocks[64];
  int blk_first;
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(void* mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(void* mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(void* mbar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(void* mbar, unsigned parity) {
  unsigned done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(smem_u32(mbar)), "r"(parity) : "memory");
  }
}
__device__ __forceinline__ void tma4d(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, int c3, void* mbar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, void* mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s(void* dst, const void* src, unsigned bytes, void* mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(mbar)) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// ---- wgmma helpers
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
// Shared-memory matrix descriptor, 128B swizzle: start address, leading/stride byte offsets (>>4), layout type 1.
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ void wgmma_n56(float* d, const uint32_t* a, uint64_t bdesc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %32, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27}, {%28,%29,%30,%31}, %33, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(bdesc));
}

__device__ __forceinline__ void wgmma_n64(float* d, const uint32_t* a, uint64_t bdesc, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %36, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %37, p, 1, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(bdesc));
}


struct TmaMaps { CUtensorMap k13, k1, v13, v1, ck13, cv13; };

// token t -> (slot in block, block id) for this split
__device__ __forceinline__ void locate(const Smem& S, int t, int BS, int& slot, int& blk) { blk = S.blocks[t / BS - S.blk_first]; slot = t % BS; }

__device__ __forceinline__ void producer_issue(Smem& S, int buf, const TmaMaps& M, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl,
                                               long long bstride, int BS, int seqlen, int t0, int lane) {
  const bool wide = (t0 >= 1) && (t0 + T <= seqlen);
  if (wide) {
    if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], TILE_TX);
    __syncwarp();
    if (lane < 10) {
      int s0, b0, sp, bp; locate(S, t0, BS, s0, b0); locate(S, t0 - 1, BS, sp, bp);
      switch (lane) {
        case 0: tma4d(S.bk[buf][0] + 512, &M.k13, 0, 0, s0, b0, &S.full[buf]); break;
        case 1: tma4d(S.bk[buf][1] + 512, &M.k13, 64, 0, s0, b0, &S.full[buf]); break;
        case 2: tma4d(S.bv[buf][0] + 512, &M.v13, 0, 0, s0, b0, &S.full[buf]); break;
        case 3: tma4d(S.bv[buf][1] + 512, &M.v13, 64, 0, s0, b0, &S.full[buf]); break;
        case 4: tma3d(S.ck[buf], &M.ck13, 0, s0, b0, &S.full[buf]); break;
        case 5: tma3d(S.cv[buf], &M.cv13, 0, s0, b0, &S.full[buf]); break;
        case 6: tma4d(S.bk[buf][0], &M.k1, 0, 0, sp, bp, &S.full[buf]); break;
        case 7: tma4d(S.bk[buf][1], &M.k1, 64, 0, sp, bp, &S.full[buf]); break;
        case 8: tma4d(S.bv[buf][0], &M.v1, 0, 0, sp, bp, &S.full[buf]); break;
        case 9: tma4d(S.bv[buf][1], &M.v1, 64, 0, sp, bp, &S.full[buf]); break;
      }
    }
    return;
  }
  // narrow path (first / last tile): one lane per token, invalid tokens zero-filled before the release
  const int tok = lane, t = t0 - 1 + tok;
  const bool mine = lane < NTOK, valid = mine && (t >= 0) && (t < seqlen);
  const unsigned nvalid = __popc(__ballot_sync(0xffffffffu, valid));
  const unsigned nvalid_c = __popc(__ballot_sync(0xffffffffu, valid && tok >= 1));
  if (mine && !valid) {
    for (int h = 0; h < 2; ++h) for (int c = 0; c < 32; ++c) {
      *reinterpret_cast<uint4*>(S.bk[buf][h] + tok * 512 + c * 16) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(S.bv[buf][h] + tok * 512 + c * 16) = make_uint4(0, 0, 0, 0);
    }
    if (tok >= 1) {
      __nv_bfloat16* zk = S.ck[buf] + (tok - 1) * NCOEF; __nv_bfloat16* zv = S.cv[buf] + (tok - 1) * NCOEF;
      for (int c = 0; c < NCOEF / 8; ++c) { *reinterpret_cast<uint4*>(zk + c * 8) = make_uint4(0, 0, 0, 0); *reinterpret_cast<uint4*>(zv + c * 8) = make_uint4(0, 0, 0, 0); }
    }
    __threadfence_block();
  }
  __syncwarp();
  if (lane == 0) mbar_arrive_expect_tx(&S.full[buf], nvalid * 2u * 2 * 512 + nvalid_c * 2u * NCOEF * 2);
  __syncwarp();
  if (mine && valid) {
    int s, bl; locate(S, t, BS, s, bl);
    tma4d(S.bk[buf][0] + tok * 512, &M.k1, 0, 0, s, bl, &S.full[buf]);
    tma4d(S.bk[buf][1] + tok * 512, &M.k1, 64, 0, s, bl, &S.full[buf]);
    tma4d(S.bv[buf][0] + tok * 512, &M.v1, 0, 0, s, bl, &S.full[buf]);
    tma4d(S.bv[buf][1] + tok * 512, &M.v1, 64, 0, s, bl, &S.full[buf]);
    if (tok >= 1) {   // token 0's own coefficients are never used
      const size_t grow = (size_t)bl * bstride + (size_t)s * ROWP;   // plain bulk copies: 16B-aligned dst
      tma_bulk_g2s(S.ck[buf] + (tok - 1) * NCOEF, kpl + grow + OFF_C, NCOEF * 2, &S.full[buf]);
      tma_bulk_g2s(S.cv[buf] + (tok - 1) * NCOEF, vpl + grow + OFF_C, NCOEF * 2, &S.full[buf]);
    }
  }
}

__global__ void __launch_bounds__(THREADS, 2)
tpa_factor_decode_kernel(const __grid_constant__ TmaMaps maps, const __nv_bfloat16* __restrict__ kpl, const __nv_bfloat16* __restrict__ vpl, long long bstride,
              const __nv_bfloat16* __restrict__ q, const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
              float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
              float sm_scale, float softcap, const int* __restrict__ tok_per_split_ptr, int BS) {
  extern __shared__ __align__(1024) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int tok_per_split = *tok_per_split_ptr;
  const float inv_cap = softcap > 0.f ? 1.f / softcap : 0.f;
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qd = lane & 3, rl = lane >> 2;
  const int row0 = warp * 16 + rl, row1 = row0 + 8;
  const int g0 = min(row0, HQ - 1) / (HQ / HKV), g1 = min(row1, HQ - 1) / (HQ / HKV);
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);
  if (stop <= start) {
    if (tid < HQP) { const size_t base = ((size_t)b * nsplit + sp) * HQP; part_m[base + tid] = -CUDART_INF_F; part_l[base + tid] = 0.f; }
    return;
  }
  // Q fragments (wgmma A operand from registers: same per-warp layout as mma.sync), padded heads zero
  uint32_t qa[D / 16][4];
  if (warp < NCONS / 32) {
    const __nv_bfloat16* qb = q + (size_t)b * HQ * D;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      const int col = ks * 16 + 2 * qd;
      qa[ks][0] = row0 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row0 * D + col) : 0u;
      qa[ks][1] = row1 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row1 * D + col) : 0u;
      qa[ks][2] = row0 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row0 * D + col + 8) : 0u;
      qa[ks][3] = row1 < HQ ? *reinterpret_cast<const uint32_t*>(qb + row1 * D + col + 8) : 0u;
    }
  }
  // rows 56..63 of every B tile stay zero (never written by TMA)
  for (int i = tid; i < NBUF * 2 * 2 * (1024 / 16); i += THREADS) {
    const int buf = i / (2 * 2 * 64), rem = i % (2 * 2 * 64), pl = rem / 128, h = (rem / 64) % 2, c = rem % 64;
    char* base = pl == 0 ? S.bk[buf][h] : S.bv[buf][h];
    *reinterpret_cast<uint4*>(base + 7168 + c * 16) = make_uint4(0, 0, 0, 0);
  }
  float acc[D / 8 * 4];
#pragma unroll
  for (int i = 0; i < D / 8 * 4; ++i) acc[i] = 0.f;
  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;
  {
    const int bfirst = max(start - 1, 0) / BS, blast = (max(stop, 1) - 1) / BS;
    if (tid == 0) S.blk_first = bfirst;
    for (int i = tid; i <= blast - bfirst && i < 64; i += THREADS) S.blocks[i] = bt[b * bt_stride + bfirst + i];
  }
  if (tid == 0) {
    for (int i = 0; i < NBUF; ++i) { mbar_init(&S.full[i], 1); mbar_init(&S.empty[i], NCONS / 32); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntiles = (stop - start + T - 1) / T;

  if (warp == NCONS / 32) {
    for (int it = 0; it < ntiles; ++it) {
      const int buf = it % NBUF;
      if (it >= NBUF) mbar_wait(&S.empty[buf], ((it / NBUF) + 1) & 1);
      producer_issue(S, buf, maps, kpl, vpl, bstride, BS, seqlen, start + it * T, lane);
    }
    return;
  }

  for (int it = 0; it < ntiles; ++it) {
    const int buf = it % NBUF, t0 = start + it * T;
    mbar_wait(&S.full[buf], (it / NBUF) & 1);
    const __nv_bfloat16* CfK = S.ck[buf] - NCOEF;   // token tok -> CfK + tok*NCOEF for tok >= 1
    const __nv_bfloat16* CfV = S.cv[buf] - NCOEF;

    // ---- qB = Q Bk^T (64 x 56): 2 k-halves x 4 k16 steps of wgmma; B = K-major SW128 tile, k16 step = +32B
    float qB[NT8 * 4];
#pragma unroll
    for (int i = 0; i < NT8 * 4; ++i) qB[i] = 0.f;
    wg_fence();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t base = smem_u32(S.bk[buf][h]);
#pragma unroll
      for (int s = 0; s < 4; ++s) wgmma_n56(qB, qa[h * 4 + s], make_desc(base + s * 32, 16, 1024), (h | s) != 0);
    }
    wg_commit();
    wg_wait<0>();

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
      const bool ok = (k >= 1) && (k <= T) && (t0 + t < seqlen);
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

    // ---- W (A operand, 64 x 64): W[row][n] = p[row][tokidx-1] Cv[tokidx-1] + p[row][tokidx] Dv[tokidx], n = tokidx*4 + r
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
    // ---- acc += W Bv: per k16 step (16 rows = 2048B of the MN-major tile), two n64 halves
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
      for (int h = 0; h < 2; ++h) wgmma_n64(acc + h * 32, aW[ks], make_desc(smem_u32(S.bv[buf][h]) + ks * 2048, 8192, 1024), 1);
    }
    wg_commit();
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(&S.empty[buf]);
  }
  // ---- partials: acc[nt*4 + {0,1}] -> (row0, cols nt*8 + 2qd, +1); {2,3} -> row1
  const size_t base = ((size_t)b * nsplit + sp) * HQP;
#pragma unroll
  for (int nt = 0; nt < D / 8; ++nt) {
    float* o0 = part_o + (base + row0) * D + nt * 8 + 2 * qd; float* o1 = part_o + (base + row1) * D + nt * 8 + 2 * qd;
    o0[0] = acc[nt * 4 + 0]; o0[1] = acc[nt * 4 + 1]; o1[0] = acc[nt * 4 + 2]; o1[1] = acc[nt * 4 + 3];
  }
  if (qd == 0) { part_m[base + row0] = m0; part_m[base + row1] = m1; part_l[base + row0] = l0; part_l[base + row1] = l1; }
}
}  // namespace

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres; void* p = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}
// plane: (num_blocks, block_size, 608) with block stride `bstride` elements. Factor map dims (d 128, r 4, slot, block).
static CUtensorMap make_factor_map(const void* base, int64_t bs, int64_t nb, int64_t bstride, int box_tok) {
  CUtensorMap m;
  cuuint64_t gdim[4] = {(cuuint64_t)D, (cuuint64_t)R, (cuuint64_t)bs, (cuuint64_t)nb};
  cuuint64_t gstride[3] = {(cuuint64_t)D * 2, (cuuint64_t)ROWP * 2, (cuuint64_t)bstride * 2};
  cuuint32_t box[4] = {64, (cuuint32_t)R, (cuuint32_t)box_tok, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(base), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(factor) failed: ", (int)r);
  return m;
}
static CUtensorMap make_coef_map(const void* base, int64_t bs, int64_t nb, int64_t bstride, int box_tok) {
  CUtensorMap m;
  cuuint64_t gdim[3] = {(cuuint64_t)NCOEF, (cuuint64_t)bs, (cuuint64_t)nb};
  cuuint64_t gstride[2] = {(cuuint64_t)ROWP * 2, (cuuint64_t)bstride * 2};
  cuuint32_t box[3] = {(cuuint32_t)NCOEF, (cuuint32_t)box_tok, 1};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(coef) failed: ", (int)r);
  return m;
}

void tpa_factor_decode_launch(torch::Tensor q, torch::Tensor kplane, torch::Tensor vplane, torch::Tensor block_table, torch::Tensor seq_lens,
                   torch::Tensor tok_per_split, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                   double sm_scale, double softcap, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1);
  TORCH_CHECK(kplane.dim() == 3 && kplane.size(2) == ROWP && kplane.stride(2) == 1 && kplane.stride(1) == ROWP && q.is_contiguous());
  TORCH_CHECK(vplane.stride(0) == kplane.stride(0) && vplane.stride(1) == ROWP && kplane.size(1) == block_size && block_size % T == 0);
  TORCH_CHECK(block_table.dtype() == torch::kInt32 && seq_lens.dtype() == torch::kInt32 && tok_per_split.dtype() == torch::kInt32);
  static std::unordered_map<const void*, TmaMaps> maps;
  const void* kb = kplane.data_ptr(); const void* vb = vplane.data_ptr();
  auto it = maps.find(kb);
  if (it == maps.end()) {
    TmaMaps M;
    const int64_t nb = kplane.size(0), bs = block_size, bstr = kplane.stride(0);
    M.k13 = make_factor_map(kb, bs, nb, bstr, T); M.k1 = make_factor_map(kb, bs, nb, bstr, 1);
    M.v13 = make_factor_map(vb, bs, nb, bstr, T); M.v1 = make_factor_map(vb, bs, nb, bstr, 1);
    M.ck13 = make_coef_map(static_cast<const __nv_bfloat16*>(kb) + OFF_C, bs, nb, bstr, T);
    M.cv13 = make_coef_map(static_cast<const __nv_bfloat16*>(vb) + OFF_C, bs, nb, bstr, T);
    it = maps.emplace(kb, M).first;
  }
  static bool attr_set = false;
  const int smem = (int)sizeof(Smem) + 1024;
  if (!attr_set) { cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); attr_set = true; }
  tpa_factor_decode_kernel<<<dim3(B, SPLIT), THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      it->second, reinterpret_cast<const __nv_bfloat16*>(kb), reinterpret_cast<const __nv_bfloat16*>(vb), (long long)kplane.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (float)softcap, tok_per_split.data_ptr<int>(), (int)block_size);
}
int64_t tpa_factor_smem_bytes() { return sizeof(Smem) + 1024; }
int64_t tpa_factor_occupancy() {
  const int smem = (int)sizeof(Smem) + 1024;
  cudaFuncSetAttribute(tpa_factor_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int nb = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, tpa_factor_decode_kernel, THREADS, smem);
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, tpa_factor_decode_kernel);
  printf("occupancy: %d CTAs/SM (regs %d, dyn smem %d, local %zu)\n", nb, a.numRegs, smem, a.localSizeBytes);
  return nb;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &tpa_factor_decode_launch); m.def("smem_bytes", &tpa_factor_smem_bytes); m.def("occupancy", &tpa_factor_occupancy); m.attr("TILE") = T; }
