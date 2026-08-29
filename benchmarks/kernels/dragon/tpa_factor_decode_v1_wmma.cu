// Decode attention over a TPA-factorized KV cache (Dragon DiffTPA), sm_90.
// Cache row (bf16, 1216): Bk[4x128] | Bv[4x128] | Ck[12x4] | Dk[12x4] | Cv[12x4] | Dv[12x4]
// k_t[h] = sum_r Ck[t,g(h),r] Bk_t[r] + Dk[t,g(h),r] Bk_{t-1}[r]   (same for v)
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math_constants.h>
using namespace nvcuda;

namespace {
constexpr int HQ = 48, HQP = 64, HKV = 12, R = 4, D = 128, T = 8, NTOK = T + 1, NCOL = NTOK * R, KC = 48;
constexpr int ROW = 1216, OFF_BK = 0, OFF_BV = 512, OFF_C0 = 1024;  // coef arrays at 1024 + a*48
constexpr int THREADS = 128;

struct __align__(128) Smem {
  __nv_bfloat16 Q[HQP * D];
  __nv_bfloat16 Bk[KC * D];
  __nv_bfloat16 Bv[KC * D];
  __nv_bfloat16 coef[NTOK * 4 * 48];
  __nv_bfloat16 W[HQP * KC];
  float qB[HQP * KC];
  float acc[HQP * D];
  float s[HQP * T];
  float p[HQP * T];
  float alpha[HQP], m[HQP], l[HQP];
};

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

__global__ void __launch_bounds__(THREADS)
tpa_decode_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache,
                  const int* __restrict__ bt, int bt_stride, const int* __restrict__ seqlens,
                  float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
                  float sm_scale, int tok_per_split, int BS) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  const int b = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int seqlen = seqlens[b];
  const int start = sp * tok_per_split, stop = min(start + tok_per_split, seqlen);

  // Q (padded heads zero), pad rows of Bk/Bv, W, acc, m, l
  for (int i = tid; i < HQP * D / 8; i += THREADS) {
    int row = i / (D / 8), c8 = i % (D / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < HQ) v = *reinterpret_cast<const uint4*>(q + ((size_t)b * HQ + row) * D + c8 * 8);
    *reinterpret_cast<uint4*>(S.Q + row * D + c8 * 8) = v;
  }
  for (int i = tid; i < (KC - NCOL) * D / 8; i += THREADS) {
    int row = NCOL + i / (D / 8), c8 = i % (D / 8);
    *reinterpret_cast<uint4*>(S.Bk + row * D + c8 * 8) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(S.Bv + row * D + c8 * 8) = make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < HQP * D; i += THREADS) S.acc[i] = 0.f;
  for (int i = tid; i < HQP * KC; i += THREADS) S.W[i] = __float2bfloat16(0.f);
  if (tid < HQP) { S.m[tid] = -CUDART_INF_F; S.l[tid] = 0.f; }
  __syncthreads();

  for (int t0 = start; t0 < stop; t0 += T) {
    // ---- gather 9 tokens (t0-1 .. t0+T-1): 152 x 16B chunks each
    for (int i = tid; i < NTOK * 152; i += THREADS) {
      int tok = i / 152, c = i % 152, t = t0 - 1 + tok;
      bool valid = (t >= 0) && (t < seqlen);
      uint4 v = make_uint4(0, 0, 0, 0);
      const __nv_bfloat16* src = nullptr;
      __nv_bfloat16* dst;
      if (c < 64) { dst = S.Bk + tok * (R * D) + c * 8; if (valid) src = cache + OFF_BK + c * 8; }
      else if (c < 128) { dst = S.Bv + tok * (R * D) + (c - 64) * 8; if (valid) src = cache + OFF_BV + (c - 64) * 8; }
      else { int a = (c - 128) / 6, k = (c - 128) % 6; dst = S.coef + (tok * 4 + a) * 48 + k * 8; if (valid) src = cache + OFF_C0 + a * 48 + k * 8; }
      if (valid) {
        int blk = bt[b * bt_stride + t / BS];
        v = *reinterpret_cast<const uint4*>(src + ((size_t)blk * BS + (t % BS)) * ROW);
      }
      *reinterpret_cast<uint4*>(dst) = v;
    }
    __syncthreads();
    // ---- qB[64 x 48] = Q[64 x 128] * Bk^T   (B operand: col-major view of Bk rows)
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bb;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      for (int nt = 0; nt < KC / 16; ++nt) {
        wmma::fill_fragment(c, 0.f);
        for (int k = 0; k < D / 16; ++k) {
          wmma::load_matrix_sync(a, S.Q + warp * 16 * D + k * 16, D);
          wmma::load_matrix_sync(bb, S.Bk + nt * 16 * D + k * 16, D);
          wmma::mma_sync(c, a, bb, c);
        }
        wmma::store_matrix_sync(S.qB + warp * 16 * KC + nt * 16, c, KC, wmma::mem_row_major);
      }
    }
    __syncthreads();
    // ---- scores: thread -> (head, half of the tile)
    {
      int h = tid >> 1, half = tid & 1, g = min(h, HQ - 1) / (HQ / HKV);
      for (int tt = 0; tt < T / 2; ++tt) {
        int tl = half * (T / 2) + tt;                   // token t0+tl is smem token index tl+1
        const __nv_bfloat16* Ck = S.coef + ((tl + 1) * 4 + 0) * 48 + g * R;
        const __nv_bfloat16* Dk = S.coef + ((tl + 1) * 4 + 1) * 48 + g * R;
        const float* qcur = S.qB + h * KC + (tl + 1) * R;
        const float* qprv = S.qB + h * KC + tl * R;
        float sc = 0.f;
#pragma unroll
        for (int r = 0; r < R; ++r) sc += bf(Ck[r]) * qcur[r] + bf(Dk[r]) * qprv[r];
        sc *= sm_scale;
        if (t0 + tl >= seqlen) sc = -CUDART_INF_F;
        if (h >= HQ) sc = 0.f;
        S.s[h * T + tl] = sc;
      }
    }
    __syncthreads();
    // ---- online softmax per head
    if (tid < HQP) {
      float mo = S.m[tid], mx = mo;
      for (int t = 0; t < T; ++t) mx = fmaxf(mx, S.s[tid * T + t]);
      float alpha = (mo == -CUDART_INF_F) ? 0.f : __expf(mo - mx);
      float sum = 0.f;
      for (int t = 0; t < T; ++t) { float pt = __expf(S.s[tid * T + t] - mx); S.p[tid * T + t] = pt; sum += pt; }
      S.l[tid] = S.l[tid] * alpha + sum; S.m[tid] = mx; S.alpha[tid] = alpha;
    }
    __syncthreads();
    // ---- W operand: column n=(tokidx, r): own-token Cv (tl = tokidx-1) + next-token Dv (tl = tokidx)
    for (int i = tid; i < HQP * NCOL; i += THREADS) {
      int h = i / NCOL, n = i % NCOL, tokidx = n / R, r = n % R, g = min(h, HQ - 1) / (HQ / HKV);
      float w = 0.f;
      if (tokidx >= 1) { int tl = tokidx - 1; w += S.p[h * T + tl] * bf(S.coef[((tl + 1) * 4 + 2) * 48 + g * R + r]); }
      if (tokidx < T) { int tl = tokidx; w += S.p[h * T + tl] * bf(S.coef[((tl + 1) * 4 + 3) * 48 + g * R + r]); }
      S.W[h * KC + n] = __float2bfloat16(w);
    }
    for (int i = tid; i < HQP * D; i += THREADS) S.acc[i] *= S.alpha[i / D];
    __syncthreads();
    // ---- acc[64 x 128] += W[64 x 48] * Bv[48 x 128]
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bb;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      for (int nt = 0; nt < D / 16; ++nt) {
        wmma::load_matrix_sync(c, S.acc + warp * 16 * D + nt * 16, D, wmma::mem_row_major);
        for (int k = 0; k < KC / 16; ++k) {
          wmma::load_matrix_sync(a, S.W + warp * 16 * KC + k * 16, KC);
          wmma::load_matrix_sync(bb, S.Bv + k * 16 * D + nt * 16, D);
          wmma::mma_sync(c, a, bb, c);
        }
        wmma::store_matrix_sync(S.acc + warp * 16 * D + nt * 16, c, D, wmma::mem_row_major);
      }
    }
    __syncthreads();
  }
  const size_t base = ((size_t)b * nsplit + sp) * HQP;
  for (int i = tid; i < HQP * D; i += THREADS) part_o[base * D + i] = S.acc[i];
  if (tid < HQP) { part_m[base + tid] = S.m[tid]; part_l[base + tid] = S.l[tid]; }
}
}  // namespace

void tpa_decode_launch(torch::Tensor q, torch::Tensor cache, torch::Tensor block_table, torch::Tensor seq_lens,
                       torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                       double sm_scale, int64_t tok_per_split, int64_t block_size) {
  const int B = q.size(0), SPLIT = part_o.size(1);
  static bool attr_set = false;
  if (!attr_set) { cudaFuncSetAttribute(tpa_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem)); attr_set = true; }
  dim3 grid(B, SPLIT);
  tpa_decode_kernel<<<grid, THREADS, sizeof(Smem), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cache.data_ptr()),
      block_table.data_ptr<int>(), (int)block_table.stride(0), seq_lens.data_ptr<int>(),
      part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(), (float)sm_scale, (int)tok_per_split, (int)block_size);
}
int64_t tpa_smem_bytes() { return sizeof(Smem); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tpa_decode_launch", &tpa_decode_launch);
  m.def("tpa_smem_bytes", &tpa_smem_bytes);
}
