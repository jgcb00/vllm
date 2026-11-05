// Olala Mamba-3 MIMO decode step with ReplaySSM (Liou et al., "ReplaySSM: Cache SSM Inputs, Not State",
// arXiv 2610.13778), adapted to the trapezoidal, rotary, rank-R Mamba-3 update. Built JIT by mamba3_step_cuda.py.
//
// The state of a request is a checkpoint S_c (the "anchor", fp32 or bf16, plus an optional fp8 read copy) and a
// ring of the tokens decoded since. Token i contributes  X_i (x) B'_i  (X_i[p][r] = x_i[p] xproj[r][p], B'_i the
// biased + rotated B) with weight gamma_i at its own step and beta_{i+1} at the next one, so after the current
// token (index n-1; index 0 is the token before the window, kept in k_pool / v_pool):
//     S = exp(L_{n-1}) S_c + sum_k w_k X_k (x) B'_k,  w_k = gamma_k e^{L_{n-1}-L_k} + beta_{k+1} e^{L_{n-1}-L_{k+1}}
// with L the cumulative log-decay A*dt over the window. Most steps only read S_c (preferably the fp8 copy) and
// append the token to the ring; the output is o = e^{L} S_c C'^T + sum_k w_k X_k G_k with G_k = B'_k C'^T.
// When the ring is full the step "flushes": S is accumulated in fp32 from the anchor and the ring and written back
// to the anchor and, independently, to the read copy. The read copy is never fed back into the recurrence, so its
// rounding error does not propagate; the anchor is rounded at most once per window (stochastically when bf16).
//
// Ring layout per slot: x (H, L, D) bf16; B before bias/rotation (L, R, S) bf16, shared by the heads (one group);
// rotary phase (H, L, NA) fp32; (A*dt, gamma, beta, -) (H, L, 4) fp32; (count, flush target) (H, 2) int32.
//
// One CTA (4 warps) per (head, token); warp w owns state rows [16w, 16w + 16) in the mma fragment layout of the
// plain step (lane: g = lane / 4 -> rows 16w + g and 16w + g + 8, qd = lane % 4 -> columns 8j + 2qd, +1).
// Tensor cores: G = B' C'^T (bf16), the fp8 checkpoint read S_c C'^T (fp8 -> f16 is exact, f16 C' is exact), and the
// flush update sum_k w_k X_k (x) B'_k with the coefficients split in two bf16 terms (hi + lo: ~fp32 accurate).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {
constexpr int D = 64, S = 128, R = 4, NA = 32, THREADS = 128, HALF = S / 2;
constexpr int LD8 = S + 16;   // fp8 tile row (bytes): conflict-free fragment reads
constexpr int LDB = S + 8;    // bf16 / f16 operand rows: conflict-free ldmatrix
constexpr float PI = 3.14159265358979323846f, TWO_PI = 6.283185307179586f, FP8_MAX = 448.f;

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
  __half2 h = __floats2half2_rn(f.x, f.y); return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t hash32(uint32_t x) {   // lowbias32
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
// fp32 -> bf16 with stochastic rounding (add 16 random bits below the kept mantissa, truncate)
__device__ __forceinline__ __nv_bfloat16 bf16_sr(float v, uint32_t rnd) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7f800000u) != 0x7f800000u) u = (u + (rnd & 0xffffu)) & 0xffff0000u;
  return __ushort_as_bfloat16((unsigned short)(u >> 16));
}

struct Args {
  float* angle; void* anchor; __nv_fp8_e4m3* rd; float* rd_scale; __nv_bfloat16* kp; __nv_bfloat16* vp;
  __nv_bfloat16* xr; __nv_bfloat16* br; float* thr; float* scr; int* cnt;
  long long ang_sst, anc_sst, rd_sst, rds_sst, kp_sst, vp_sst, xr_sst, br_sst, thr_sst, scr_sst, cnt_sst;
  int P, H, L;
  const float* A; const __nv_bfloat16* Bm; const __nv_bfloat16* Cm; const float* Dp; const __nv_bfloat16* x;
  const float* dt; const float* trap; const float* xproj; const float* zproj; const float* outproj;
  const __nv_bfloat16* z; const float* bias_q; const float* bias_k; const __nv_bfloat16* angp; const int* slots;
  __nv_bfloat16* y; long long bc_bstride, angp_stride;
};

// dynamic smem carve-up for up to n = L + 1 tokens (B' rows padded to a whole number of 16-row mma tiles)
struct Layout {
  int q8, bk, cb, ch, xb, G, w, lc, gam, bet, thr, scr, bias, total, bk_rows;
  __host__ __device__ Layout(int L) {
    const int n = L + 1; int o = 0;
    bk_rows = ((n * R + 15) / 16) * 16;
    q8 = o;   o += D * LD8;                // fp8 checkpoint tile (read; staging of the fp8 store on a flush)
    bk = o;   o += bk_rows * LDB * 2;      // B rows (k, r), bf16: raw, then biased + rotated in place
    cb = o;   o += 8 * LDB * 2;            // C, same; rows 4..7 zero
    ch = o;   o += 8 * LDB * 2;            // C' f16, rows 4..7 zero
    xb = o;   o += n * D * 2;              // x_k, bf16
    thr = o;  o += L * NA * 4;             // ring phases
    scr = o;  o += L * 4 * 4;              // ring (A*dt, gamma, beta, -)
    bias = o; o += 2 * R * S * 4;          // bias_k, bias_q of this head
    G = o;    o += n * R * R * 4;          // G_k[r][r2]
    w = o;    o += n * 4;
    lc = o;   o += n * 4;                  // cumulative log-decay
    gam = o;  o += n * 4;
    bet = o;  o += (n + 1) * 4;
    total = o;
  }
};

#ifndef REPLAY_MINB
#define REPLAY_MINB 4
#endif
template <bool ANCHOR_F32, bool READ_FP8>
__global__ void __launch_bounds__(THREADS, REPLAY_MINB) replay_step_kernel(const Args a) {
  extern __shared__ __align__(16) char smem[];
  const Layout lo(a.L);
  uint8_t* sQ = reinterpret_cast<uint8_t*>(smem + lo.q8);
  __nv_bfloat16* sBk = reinterpret_cast<__nv_bfloat16*>(smem + lo.bk);
  __nv_bfloat16* sCb = reinterpret_cast<__nv_bfloat16*>(smem + lo.cb);
  __half* sCh = reinterpret_cast<__half*>(smem + lo.ch);
  __nv_bfloat16* sXb = reinterpret_cast<__nv_bfloat16*>(smem + lo.xb);
  float* sThr = reinterpret_cast<float*>(smem + lo.thr);
  float* sScr = reinterpret_cast<float*>(smem + lo.scr);
  float* sBias = reinterpret_cast<float*>(smem + lo.bias);
  float* sG = reinterpret_cast<float*>(smem + lo.G);
  float* sW = reinterpret_cast<float*>(smem + lo.w);
  float* sL = reinterpret_cast<float*>(smem + lo.lc);
  float* sGam = reinterpret_cast<float*>(smem + lo.gam);
  float* sBet = reinterpret_cast<float*>(smem + lo.bet);
  __shared__ float sTh[NA], sCosC[NA], sSinC[NA];
  __shared__ float sDecay;

  const int H = a.H, h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31, g = lane >> 2, qd = lane & 3;
  const int slot = a.slots[b];
  __nv_bfloat16* yrow = a.y + ((size_t)b * H + h) * D;
  // padding lanes: PAD_SLOT_ID (-1) and the null block (0), which CUDA-graph padding uses and no request owns (its
  // ring was never reset by a prefill: a garbage count there would index past the ring)
  if (slot <= 0 || slot >= a.P) { if (t < D) yrow[t] = __float2bfloat16(0.f); return; }

  // ---- fp8 checkpoint tile: async, issued before anything else (a flush step does not use it)
  if constexpr (READ_FP8) {
    const __nv_fp8_e4m3* src = a.rd + (size_t)slot * a.rd_sst + (size_t)h * D * S;
#pragma unroll
    for (int i = 0; i < D * S / 16 / THREADS; ++i) { const int c = t + i * THREADS, row = c >> 3, ch = c & 7; cp16(sQ + row * LD8 + ch * 16, src + c * 16); }
  }
  int* cntp = a.cnt + (size_t)slot * a.cnt_sst + h * 2;
  const int hc = min(max(cntp[0], 0), a.L - 1), lim = min(max(cntp[1], 1), a.L);   // defensive: stay inside the ring
  const bool flush = hc + 1 >= lim;
  const int n = hc + 2, cur = n - 1;   // tokens: 0 = before the window, 1..hc ring, cur = this one
  const int nrows = n * R, nks = (nrows + 15) / 16;
  const int p0 = warp * 16 + g, p1 = p0 + 8;
  // bf16 anchor: a lane's flush fragment (rows p0, p1 x 32 columns) is 32 bf16 pairs, read now so that its latency
  // hides behind the ring copies and the rotation instead of following them
  uint32_t apre[ANCHOR_F32 ? 1 : 32];
  if constexpr (!ANCHOR_F32) {
    if (flush) {
      const __nv_bfloat16* an = reinterpret_cast<const __nv_bfloat16*>(a.anchor) + (size_t)slot * a.anc_sst + (size_t)h * D * S;
#pragma unroll
      for (int jt = 0; jt < 16; ++jt) {
        apre[2 * jt] = *reinterpret_cast<const uint32_t*>(an + p0 * S + jt * 8 + 2 * qd);
        apre[2 * jt + 1] = *reinterpret_cast<const uint32_t*>(an + p1 * S + jt * 8 + 2 * qd);
      }
    }
  }

  // ---- everything else the step reads, in one async batch: B rows (token 0 = k_pool, already rotated; ring; current)
  // and C, x rows (v_pool, ring, current), ring phases and scalars, this head's biases
  {
    const __nv_bfloat16* br = a.br + (size_t)slot * a.br_sst;
    for (int i = t; i < (n + 1) * R * (S / 8); i += THREADS) {
      const int k = i / (R * (S / 8)), r = (i >> 4) & 3, c = (i & 15) * 8;
      const __nv_bfloat16* src = k == 0 ? a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S
                               : k < cur ? br + ((size_t)(k - 1) * R + r) * S
                               : k == cur ? a.Bm + (size_t)b * a.bc_bstride + r * S : a.Cm + (size_t)b * a.bc_bstride + r * S;
      __nv_bfloat16* dst = k <= cur ? sBk + (k * R + r) * LDB : sCb + r * LDB;
      cp16(dst + c, src + c);
    }
    const __nv_bfloat16* xr = a.xr + (size_t)slot * a.xr_sst + (size_t)h * a.L * D;
    for (int i = t; i < n * (D / 8); i += THREADS) {
      const int k = i >> 3, c = (i & 7) * 8;
      const __nv_bfloat16* src = k == 0 ? a.vp + (size_t)slot * a.vp_sst + h * D : k == cur ? a.x + ((size_t)b * H + h) * D : xr + (k - 1) * D;
      cp16(sXb + k * D + c, src + c);
    }
    const float* thr = a.thr + (size_t)slot * a.thr_sst + (size_t)h * a.L * NA;
    for (int i = t; i < hc * (NA / 4); i += THREADS) cp16(sThr + i * 4, thr + i * 4);
    const float* scr = a.scr + (size_t)slot * a.scr_sst + (size_t)h * a.L * 4;
    if (t < hc) cp16(sScr + t * 4, scr + t * 4);
    for (int i = t; i < 2 * R * S / 4; i += THREADS) {
      const int q = i / (R * S / 4), rem = i % (R * S / 4), r = rem / (S / 4), c = (rem % (S / 4)) * 4;
      cp16(sBias + (q * R + r) * S + c, (q ? a.bias_q : a.bias_k) + ((size_t)r * H + h) * S + c);
    }
    cp_commit();
  }

  const float dt = a.dt[b * H + h], Av = a.A[b * H + h], tr = a.trap[b * H + h];
  const float alpha = __expf(Av * dt), lam = Av * dt, gam = tr * dt, bet = (1.f - tr) * dt * alpha;
  // ---- current phase (as the plain step: accurate sincos, wrapped to [0, 2 pi)) while the copies land
  if (t < NA) {
    float th = a.angle[(size_t)slot * a.ang_sst + h * NA + t] + tanhf(bf2f(a.angp[b * a.angp_stride + t])) * dt * PI;
    th -= TWO_PI * floorf(th / TWO_PI);
    sTh[t] = th; sincosf(th, &sSinC[t], &sCosC[t]);
  }
  for (int i = t; i < (nks * 16 - nrows) * (S / 8); i += THREADS) {   // pad rows of the last mma tile: zero
    const int r = nrows + i / (S / 8), c = i % (S / 8);
    *reinterpret_cast<uint4*>(sBk + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
  }
  if (t < 4 * (S / 8)) {
    const int r = 4 + t / (S / 8), c = t % (S / 8);
    *reinterpret_cast<uint4*>(sCb + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(sCh + r * LDB + c * 8) = make_uint4(0, 0, 0, 0);
  }
  cp_wait_all();
  __syncthreads();

  if (t == 0) {   // window scalars; cumulative log-decay (sL[0] = 0 stands for the checkpoint)
    float acc = 0.f; sL[0] = 0.f; sGam[0] = 0.f; sBet[n] = 0.f;
    for (int k = 1; k < cur; ++k) { acc += sScr[(k - 1) * 4]; sL[k] = acc; sGam[k] = sScr[(k - 1) * 4 + 1]; sBet[k] = sScr[(k - 1) * 4 + 2]; }
    acc += lam; sL[cur] = acc; sGam[cur] = gam; sBet[cur] = bet;
    sDecay = __expf(acc);
  }
  // ---- B'_k = rot(B_k + bias_k) for the ring and the current token, C' = rot(C + bias_q), in place; bf16-rounded like
  // the plain step. Task = (token, r, 8 consecutive pair indices j): columns j..j+7 and j+64..j+71. Ring phases use
  // the fast sincos on [-pi, pi) (abs. error ~2^-21, far below the bf16 rounding of B' that follows).
  const int ntask = (n - 1) * R * (HALF / 8);
  for (int i = t; i < ntask + R * (HALF / 8); i += THREADS) {
    const bool isC = i >= ntask;
    const int ii = isC ? i - ntask : i;
    const int k = isC ? cur : 1 + ii / (R * (HALF / 8)), r = (ii >> 3) & 3, j0 = (ii & 7) * 8;
    __nv_bfloat16* row = isC ? sCb + r * LDB : sBk + (k * R + r) * LDB;
    const float* bias = sBias + ((isC ? R : 0) + r) * S;
    const uint4 ul = *reinterpret_cast<const uint4*>(row + j0), uh = *reinterpret_cast<const uint4*>(row + j0 + HALF);
    const __nv_bfloat16* el = reinterpret_cast<const __nv_bfloat16*>(&ul);
    const __nv_bfloat16* eh = reinterpret_cast<const __nv_bfloat16*>(&uh);
    float vl[8], vh[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) { vl[q] = bf2f(el[q]) + bias[j0 + q]; vh[q] = bf2f(eh[q]) + bias[j0 + HALF + q]; }
    if (j0 < NA) {
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        float c, sn;
        if (k == cur) { c = sCosC[j0 + q]; sn = sSinC[j0 + q]; }
        else { const float th = sThr[(k - 1) * NA + j0 + q]; __sincosf(th > PI ? th - TWO_PI : th, &sn, &c); }
        const float l2 = vl[q] * c - vh[q] * sn, h2 = vl[q] * sn + vh[q] * c; vl[q] = l2; vh[q] = h2;
      }
    }
    uint4 pl, ph;
    pl.x = pack_bf16(vl[0], vl[1]); pl.y = pack_bf16(vl[2], vl[3]); pl.z = pack_bf16(vl[4], vl[5]); pl.w = pack_bf16(vl[6], vl[7]);
    ph.x = pack_bf16(vh[0], vh[1]); ph.y = pack_bf16(vh[2], vh[3]); ph.z = pack_bf16(vh[4], vh[5]); ph.w = pack_bf16(vh[6], vh[7]);
    *reinterpret_cast<uint4*>(row + j0) = pl;
    *reinterpret_cast<uint4*>(row + j0 + HALF) = ph;
    if (isC) {   // f16 copy of the bf16-rounded C' (exact)
      pl.x = bf16x2_to_f16x2(pl.x); pl.y = bf16x2_to_f16x2(pl.y); pl.z = bf16x2_to_f16x2(pl.z); pl.w = bf16x2_to_f16x2(pl.w);
      ph.x = bf16x2_to_f16x2(ph.x); ph.y = bf16x2_to_f16x2(ph.y); ph.z = bf16x2_to_f16x2(ph.z); ph.w = bf16x2_to_f16x2(ph.w);
      *reinterpret_cast<uint4*>(sCh + r * LDB + j0) = pl;
      *reinterpret_cast<uint4*>(sCh + r * LDB + j0 + HALF) = ph;
    }
  }
  __syncthreads();
  if (t < n) {
    const float Lt = sL[cur];
    float wk = sGam[t] * __expf(Lt - sL[t]);
    if (t + 1 < n) wk += sBet[t + 1] * __expf(Lt - sL[t + 1]);
    sW[t] = wk;
  }

  float xp[2][R];
#pragma unroll
  for (int r = 0; r < R; ++r) { xp[0][r] = a.xproj[((size_t)r * H + h) * D + p0]; xp[1][r] = a.xproj[((size_t)r * H + h) * D + p1]; }
  const int mi = lane >> 3, li = lane & 7;
  float o[2];   // o[e] = output for row p_e and r2 = qd

  if (!flush) {
    // ---- G = B' C'^T: 16-row tiles of the (k, r) rows spread over the warps
    for (int mt = warp; mt < nks; mt += 4) {
      float gc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < S; kk += 16) {
        uint32_t af[4], bfr[2];
        ldsm_x4(af, sBk + (mt * 16 + (mi & 1) * 8 + li) * LDB + kk + (mi >> 1) * 8);
        ldsm_x2(bfr, sCb + (lane & 7) * LDB + kk + ((lane >> 3) & 1) * 8);
        mma_bf16(gc, af, bfr);
      }
      if (qd < 2) {
        const int m0 = mt * 16 + g, m1 = m0 + 8;
        if (m0 < nrows) { sG[m0 * R + 2 * qd] = gc[0]; sG[m0 * R + 2 * qd + 1] = gc[1]; }
        if (m1 < nrows) { sG[m1 * R + 2 * qd] = gc[2]; sG[m1 * R + 2 * qd + 1] = gc[3]; }
      }
    }
    // ---- checkpoint read: oc[p][r2] = sum_s S_c[p][s] C'[r2][s]
    float oc[2][R];
    if constexpr (READ_FP8) {
      float c4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < S; kk += 16) {
        uint32_t af[4], bfr[2];
        af[0] = fp8x2_to_f16x2(*reinterpret_cast<const uint16_t*>(sQ + p0 * LD8 + kk + 2 * qd));
        af[1] = fp8x2_to_f16x2(*reinterpret_cast<const uint16_t*>(sQ + p1 * LD8 + kk + 2 * qd));
        af[2] = fp8x2_to_f16x2(*reinterpret_cast<const uint16_t*>(sQ + p0 * LD8 + kk + 8 + 2 * qd));
        af[3] = fp8x2_to_f16x2(*reinterpret_cast<const uint16_t*>(sQ + p1 * LD8 + kk + 8 + 2 * qd));
        ldsm_x2(bfr, sCh + (lane & 7) * LDB + kk + ((lane >> 3) & 1) * 8);
        mma_f16(c4, af, bfr);
      }
      // c4: (p0, r2 = 2qd, 2qd + 1), (p1, ...) on lanes qd < 2 -> gather (p_e, r2) for all r2 on every lane
      const float s0 = a.rd_scale[(size_t)slot * a.rds_sst + h * D + p0], s1 = a.rd_scale[(size_t)slot * a.rds_sst + h * D + p1];
#pragma unroll
      for (int r2 = 0; r2 < R; ++r2) {
        const int src = (g << 2) | (r2 >> 1);
        const float v0 = __shfl_sync(0xffffffffu, (r2 & 1) ? c4[1] : c4[0], src);
        const float v1 = __shfl_sync(0xffffffffu, (r2 & 1) ? c4[3] : c4[2], src);
        oc[0][r2] = v0 * s0; oc[1][r2] = v1 * s1;
      }
    } else {
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int r2 = 0; r2 < R; ++r2) oc[e][r2] = 0.f;
#pragma unroll 4
      for (int jt = 0; jt < 16; ++jt) {
        const int nn = jt * 8 + 2 * qd;
        float2 cr[R];
#pragma unroll
        for (int r2 = 0; r2 < R; ++r2) cr[r2] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sCb + r2 * LDB + nn));
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + (e ? p1 : p0)) * S + nn;
          float2 v;
          if constexpr (ANCHOR_F32) v = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(a.anchor) + off);
          else v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(a.anchor) + off));
#pragma unroll
          for (int r2 = 0; r2 < R; ++r2) oc[e][r2] = fmaf(v.x, cr[r2].x, fmaf(v.y, cr[r2].y, oc[e][r2]));
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int r2 = 0; r2 < R; ++r2) { oc[e][r2] += __shfl_xor_sync(0xffffffffu, oc[e][r2], 1); oc[e][r2] += __shfl_xor_sync(0xffffffffu, oc[e][r2], 2); }
    }
    __syncthreads();   // sG, sW
    // ---- window term for r2 = qd: o = e^L oc + sum_k w_k x_k[p] sum_r xproj[r][p] G_k[r][qd]
    const float decay = sDecay;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int p = e ? p1 : p0;
      float acc = decay * (qd == 0 ? oc[e][0] : qd == 1 ? oc[e][1] : qd == 2 ? oc[e][2] : oc[e][3]);
      for (int k = 0; k < n; ++k) {
        const float wx = sW[k] * bf2f(sXb[k * D + p]);
        const float* gk = sG + k * R * R + qd;
        acc = fmaf(wx, xp[e][0] * gk[0] + xp[e][1] * gk[R] + xp[e][2] * gk[2 * R] + xp[e][3] * gk[3 * R], acc);
      }
      o[e] = acc;
    }
  } else {
    __syncthreads();   // sW
    // ---- flush: S = e^L S_anchor + U B' with U[p][(k, r)] = w_k x_k[p] xproj[r][p] (mma, U split hi + lo)
    const float decay = sDecay;
    float acc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int nn = jt * 8 + 2 * qd;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float2 v;
        if constexpr (ANCHOR_F32) {
          const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + (e ? p1 : p0)) * S + nn;
          v = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(a.anchor) + off);
        } else {
          v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&apre[2 * jt + e]));
        }
        acc[jt][2 * e] = decay * v.x; acc[jt][2 * e + 1] = decay * v.y;
      }
    }
    for (int ks = 0; ks < nks; ++ks) {
      uint32_t ahi[4], alo[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {   // a0: (p0, kk, kk+1), a1: (p1, ...), a2: (p0, kk+8, +9), a3: (p1, kk+8, +9)
        const int e = q & 1, kk = ks * 16 + 2 * qd + (q >> 1) * 8, p = e ? p1 : p0;
        float u[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int m = kk + j, k = m >> 2;
          const int rr = (2 * qd + j) & 3;   // = m & 3, without indexing xp dynamically
          const float xpr = rr == 0 ? xp[e][0] : rr == 1 ? xp[e][1] : rr == 2 ? xp[e][2] : xp[e][3];
          u[j] = m < nrows ? sW[k] * bf2f(sXb[k * D + p]) * xpr : 0.f;
        }
        const float h0 = rbf(u[0]), h1 = rbf(u[1]);
        ahi[q] = pack_bf16(h0, h1); alo[q] = pack_bf16(u[0] - h0, u[1] - h1);
      }
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        uint32_t bfr[4];
        ldsm_x4_t(bfr, sBk + (ks * 16 + (mi & 1) * 8 + li) * LDB + jp * 16 + (mi >> 1) * 8);
        mma_bf16(acc[2 * jp], ahi, bfr); mma_bf16(acc[2 * jp], alo, bfr);
        mma_bf16(acc[2 * jp + 1], ahi, bfr + 2); mma_bf16(acc[2 * jp + 1], alo, bfr + 2);
      }
    }
    // output from the full state (fp32), as the plain fp32 step
    float oo[2][R];
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int r2 = 0; r2 < R; ++r2) oo[e][r2] = 0.f;
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int nn = jt * 8 + 2 * qd;
#pragma unroll
      for (int r2 = 0; r2 < R; ++r2) {
        const float2 cr = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(sCb + r2 * LDB + nn));
        oo[0][r2] = fmaf(acc[jt][0], cr.x, fmaf(acc[jt][1], cr.y, oo[0][r2]));
        oo[1][r2] = fmaf(acc[jt][2], cr.x, fmaf(acc[jt][3], cr.y, oo[1][r2]));
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
      for (int r2 = 0; r2 < R; ++r2) { oo[e][r2] += __shfl_xor_sync(0xffffffffu, oo[e][r2], 1); oo[e][r2] += __shfl_xor_sync(0xffffffffu, oo[e][r2], 2); }
      o[e] = qd == 0 ? oo[e][0] : qd == 1 ? oo[e][1] : qd == 2 ? oo[e][2] : oo[e][3];
    }
    // ---- new checkpoint: anchor (fp32, or bf16 with stochastic rounding) and the fp8 read copy, both from acc
    const uint32_t seed = hash32(__float_as_uint(sL[cur]) ^ (__float_as_uint(sTh[0]) * 0x9e3779b9u) ^ ((uint32_t)slot * 0x85ebca6bu));
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int nn = jt * 8 + 2 * qd;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = e ? p1 : p0;
        const size_t off = (size_t)slot * a.anc_sst + ((size_t)h * D + p) * S + nn;
        if constexpr (ANCHOR_F32) *reinterpret_cast<float2*>(reinterpret_cast<float*>(a.anchor) + off) = make_float2(acc[jt][2 * e], acc[jt][2 * e + 1]);
        else {
          const uint32_t rnd = hash32(((uint32_t)(h * D + p) * S + nn) ^ seed);
          __nv_bfloat162 v; v.x = bf16_sr(acc[jt][2 * e], rnd); v.y = bf16_sr(acc[jt][2 * e + 1], rnd >> 16 | rnd << 16);
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(a.anchor) + off) = v;
        }
      }
    }
    if constexpr (READ_FP8) {
      float sc[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float m = 0.f;
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) m = fmaxf(m, fmaxf(fabsf(acc[jt][2 * e]), fabsf(acc[jt][2 * e + 1])));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1)); m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        sc[e] = m > 0.f ? m / FP8_MAX : 1.f;
        const float inv = 1.f / sc[e];
        const int p = e ? p1 : p0;
#pragma unroll
        for (int jt = 0; jt < 16; ++jt) {
          const __nv_fp8x2_storage_t q = __nv_cvt_float2_to_fp8x2(make_float2(acc[jt][2 * e] * inv, acc[jt][2 * e + 1] * inv), __NV_SATFINITE, __NV_E4M3);
          *reinterpret_cast<__nv_fp8x2_storage_t*>(sQ + p * LD8 + jt * 8 + 2 * qd) = q;
        }
        if (qd == 0) a.rd_scale[(size_t)slot * a.rds_sst + h * D + p] = sc[e];
      }
      __syncthreads();
      __nv_fp8_e4m3* dst = a.rd + (size_t)slot * a.rd_sst + (size_t)h * D * S;
#pragma unroll
      for (int i = 0; i < D * S / 16 / THREADS; ++i) { const int c = t + i * THREADS, row = c >> 3, ch = c & 7;
        *reinterpret_cast<uint4*>(dst + c * 16) = *reinterpret_cast<const uint4*>(sQ + row * LD8 + ch * 16); }
    }
  }

  // ---- gated MIMO output (as the plain step): lane qd holds r = qd for rows p0, p1
  {
    const float Dh = a.Dp[h];
    float ys[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int p = e ? p1 : p0;
      const float xt = bf2f(sXb[cur * D + p]), zt = bf2f(a.z[((size_t)b * H + h) * D + p]);
      float orr = o[e] + Dh * xt * (qd == 0 ? xp[e][0] : qd == 1 ? xp[e][1] : qd == 2 ? xp[e][2] : xp[e][3]);
      const float zv = zt * a.zproj[((size_t)qd * H + h) * D + p];
      orr *= __fdividef(zv, 1.f + __expf(-zv));
      ys[e] = orr * a.outproj[((size_t)qd * H + h) * D + p];
      ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 1); ys[e] += __shfl_xor_sync(0xffffffffu, ys[e], 2);
    }
    if (qd == 0) { yrow[p0] = __float2bfloat16(ys[0]); yrow[p1] = __float2bfloat16(ys[1]); }
  }
  // ---- state bookkeeping
  if (t < NA) a.angle[(size_t)slot * a.ang_sst + h * NA + t] = sTh[t];
  if (flush) {
    if (t < R * S / 8) {   // B'_cur and x_cur become the token before the next window
      const int r = t / (S / 8), c = t % (S / 8);
      *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) =
          *reinterpret_cast<const uint4*>(sBk + (cur * R + r) * LDB + c * 8);
    }
    if (t < D / 8) *reinterpret_cast<uint4*>(a.vp + (size_t)slot * a.vp_sst + h * D + t * 8) = *reinterpret_cast<const uint4*>(a.x + ((size_t)b * H + h) * D + t * 8);
    if (t == 0) { cntp[0] = 0; cntp[1] = a.L; }
  } else {
    if (t < D / 8) *reinterpret_cast<uint4*>(a.xr + (size_t)slot * a.xr_sst + ((size_t)h * a.L + hc) * D + t * 8) = *reinterpret_cast<const uint4*>(a.x + ((size_t)b * H + h) * D + t * 8);
    if (t >= 32 && t < 32 + NA) a.thr[(size_t)slot * a.thr_sst + ((size_t)h * a.L + hc) * NA + t - 32] = sTh[t - 32];
    if (t == 0) {
      *reinterpret_cast<float4*>(a.scr + (size_t)slot * a.scr_sst + ((size_t)h * a.L + hc) * 4) = make_float4(lam, gam, bet, 0.f);
      cntp[0] = hc + 1;
    }
    if (h == 0 && t >= 64 && t < 64 + R * S / 8) {   // B is shared by the heads: head 0 records it
      const int i = t - 64;
      *reinterpret_cast<uint4*>(a.br + (size_t)slot * a.br_sst + (size_t)hc * R * S + i * 8) =
          *reinterpret_cast<const uint4*>(a.Bm + (size_t)b * a.bc_bstride + i * 8);
    }
  }
}
}  // namespace

void mamba3_replay_step(torch::Tensor angle, torch::Tensor anchor, c10::optional<torch::Tensor> rd, c10::optional<torch::Tensor> rd_scale,
                        torch::Tensor kp, torch::Tensor vp, torch::Tensor xr, torch::Tensor br, torch::Tensor thr, torch::Tensor scr, torch::Tensor cnt,
                        torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dp, torch::Tensor x, torch::Tensor dt, torch::Tensor trap,
                        torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj, torch::Tensor z, torch::Tensor bias_q, torch::Tensor bias_k,
                        torch::Tensor angp, torch::Tensor slots, torch::Tensor y) {
  Args a{};
  const int H = x.size(1), L = xr.size(2), N = x.size(0);
  TORCH_CHECK(x.size(2) == D && anchor.size(2) == D && anchor.size(3) == S && kp.size(1) == R && angle.size(2) == NA, "Olala replay step: unsupported shapes");
  TORCH_CHECK(L >= 2 && br.size(1) == L && thr.size(2) == L && scr.size(2) == L && scr.size(3) == 4 && cnt.size(2) == 2);
  TORCH_CHECK(anchor.stride(3) == 1 && anchor.stride(2) == S && anchor.stride(1) == D * S, "anchor inner dims must be contiguous");
  TORCH_CHECK(kp.stride(1) == H * S && kp.stride(2) == S && vp.stride(1) == D && xr.stride(1) == L * D && br.stride(1) == R * S
              && thr.stride(1) == L * NA && scr.stride(1) == L * 4 && cnt.stride(1) == 2 && angle.stride(1) == NA);
  TORCH_CHECK(Bm.stride(1) == S && Bm.stride(2) == 1 && Cm.stride(0) == Bm.stride(0) && Cm.stride(1) == S && angp.stride(1) == 1);
  TORCH_CHECK((Bm.stride(0) * 2) % 16 == 0 && x.is_contiguous() && z.is_contiguous(), "B/C batch stride must be 16B aligned; x, z contiguous");
  const bool f32 = anchor.dtype() == torch::kFloat32;
  TORCH_CHECK(f32 || anchor.dtype() == torch::kBFloat16);
  TORCH_CHECK((anchor.stride(0) * anchor.element_size()) % 16 == 0 && (kp.stride(0) * 2) % 16 == 0 && (vp.stride(0) * 2) % 16 == 0
              && (xr.stride(0) * 2) % 16 == 0 && (br.stride(0) * 2) % 16 == 0 && (scr.stride(0) * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  if (rd) TORCH_CHECK(rd->stride(0) % 16 == 0 && rd->stride(1) == D * S && rd_scale.has_value());
  a.angle = angle.data_ptr<float>(); a.anchor = anchor.data_ptr();
  a.rd = rd ? reinterpret_cast<__nv_fp8_e4m3*>(rd->data_ptr()) : nullptr; a.rd_scale = rd_scale ? rd_scale->data_ptr<float>() : nullptr;
  a.kp = reinterpret_cast<__nv_bfloat16*>(kp.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(vp.data_ptr());
  a.xr = reinterpret_cast<__nv_bfloat16*>(xr.data_ptr()); a.br = reinterpret_cast<__nv_bfloat16*>(br.data_ptr());
  a.thr = thr.data_ptr<float>(); a.scr = scr.data_ptr<float>(); a.cnt = cnt.data_ptr<int>();
  a.ang_sst = angle.stride(0); a.anc_sst = anchor.stride(0); a.rd_sst = rd ? rd->stride(0) : 0; a.rds_sst = rd_scale ? rd_scale->stride(0) : 0;
  a.kp_sst = kp.stride(0); a.vp_sst = vp.stride(0); a.xr_sst = xr.stride(0); a.br_sst = br.stride(0); a.thr_sst = thr.stride(0);
  a.scr_sst = scr.stride(0); a.cnt_sst = cnt.stride(0);
  a.P = anchor.size(0); a.H = H; a.L = L;
  a.A = A.data_ptr<float>(); a.Bm = reinterpret_cast<const __nv_bfloat16*>(Bm.data_ptr()); a.Cm = reinterpret_cast<const __nv_bfloat16*>(Cm.data_ptr());
  a.Dp = Dp.data_ptr<float>(); a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); a.dt = dt.data_ptr<float>(); a.trap = trap.data_ptr<float>();
  a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>();
  a.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()); a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>();
  a.angp = reinterpret_cast<const __nv_bfloat16*>(angp.data_ptr()); a.slots = slots.data_ptr<int>(); a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.bc_bstride = Bm.stride(0); a.angp_stride = angp.stride(0);
  if (N == 0) return;
  const int smem = Layout(L).total;
  const dim3 grid(H, N);
  auto stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, THREADS, smem, stream>>>(a);
  };
  const bool fp8 = rd.has_value();
  if (f32) { if (fp8) launch(replay_step_kernel<true, true>); else launch(replay_step_kernel<true, false>); }
  else { if (fp8) launch(replay_step_kernel<false, true>); else launch(replay_step_kernel<false, false>); }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &mamba3_replay_step); }
