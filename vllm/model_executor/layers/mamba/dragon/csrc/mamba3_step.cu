// Dragon Mamba-3 MIMO decode step (one token per request), persistent CUDA kernel; replaces the CuteDSL step on the
// fused decode path (vllm/model_executor/layers/mamba/dragon/mamba3.py). Built JIT by mamba3_step_cuda.py.
// Mamba-3 MIMO decode step, persistent edition: each CTA (128 threads) walks (token, head) items with a 2-stage
// cp.async pipeline (state 16 KB + previous B 4 KB + token rows per item) so the next item's loads overlap the
// current item's preamble, update and store. Warp w owns state rows p in [16w, 16w+16); lane l owns columns
// s in [4l, 4l+4) -> per row a warp touches 256 contiguous bytes of the smem tile (conflict-free), the rank-4
// coefficients B'[r][s], Bstate[r][s], C'[r][s] of the thread's 4 columns live in registers.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int H = 48, D = 64, S = 128, R = 4, NA = 32;
#ifndef STEP_NST
#define STEP_NST 2
#endif
constexpr int THREADS = 128, NST = STEP_NST;
constexpr int LDS = S + 8;   // padded smem row (bf16 elems) -> conflict-free ldmatrix on the state tile
constexpr int LDB = S + 8;   // padded rows for the B / C operand tiles
struct __align__(128) Stage {
  __nv_bfloat16 state[D * LDS];   // 17 KB (padded rows)
  __nv_bfloat16 bst[R * S];       // 1 KB previous B (k_pool)
  __nv_bfloat16 bm[R * S];        // 1 KB token B rows
  __nv_bfloat16 cm[R * S];        // 1 KB token C rows
  __nv_bfloat16 x[D], z[D], xst[D];
  __nv_bfloat16 angp[NA];
  float angst[NA];
};
struct __align__(128) Smem {
  Stage st[NST];
  float bias_k[R * S], bias_q[R * S];                  // per-head (fixed per CTA)
  float xproj[R * D], zproj[R * D], outproj[R * D];
  float sCos[NA], sSin[NA];
  __nv_bfloat16 sBk[16][LDB];       // mma B operand of the update: rows 0..3 B'[r], 4..7 Bstate[r], 8..15 zero
  __nv_bfloat16 sCb[8][LDB];        // rotated C' bf16, rows 4..7 zero: mma B operand for o = h C'^T
  float sTheta[NA];
};
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }

struct Args {
  __nv_bfloat16* ssm; __nv_bfloat16* kp; __nv_bfloat16* vp; float* angle; int P;
  const float* A; const __nv_bfloat16* Bm; const __nv_bfloat16* Cm; const float* Dp; const __nv_bfloat16* x; const float* dt; const float* trap;
  const float* xproj; const float* zproj; const float* outproj; const __nv_bfloat16* z; const float* bias_q; const float* bias_k;
  const __nv_bfloat16* angp; const int* slots; __nv_bfloat16* y; int B;
  long long bc_bstride, angp_stride;   // batch stride of Bm/Cm (elements), row stride of angle_proj
  long long ssm_sst, kp_sst, vp_sst, ang_sst;   // slot (dim 0) strides of the pools in elements (pages may be padded)
};

__device__ __forceinline__ void issue_item(const Args& a, Stage& st, int item, int t) {
  const int h = item % H, b = item / H;
  const int slot = a.slots[b]; const int srow = (slot >= 0 && slot < a.P) ? slot : 0;
  // state 16 KB = 1024 x 16B -> 8 per thread
  const __nv_bfloat16* src = a.ssm + (size_t)srow * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int c = t + i * THREADS, row = c >> 4, ch = c & 15; cp16(st.state + row * LDS + ch * 8, src + c * 8); }
  // previous B (k_pool), token B / C rows: 64 x 16B chunks each
  if (t < 64) {
    const int r = t >> 4, c = t & 15;
    cp16(st.bst + r * S + c * 8, a.kp + (size_t)srow * a.kp_sst + ((size_t)r * H + h) * S + c * 8);
    cp16(st.bm + t * 8, a.Bm + (size_t)b * a.bc_bstride + t * 8);
    cp16(st.cm + t * 8, a.Cm + (size_t)b * a.bc_bstride + t * 8);
  } else {
    const int q = t - 64;   // 64 threads: x, z, xst (8 chunks each), angp (4), angst (8 fp32 chunks)
    if (q < 8) cp16(st.x + q * 8, a.x + ((size_t)b * H + h) * D + q * 8);
    else if (q < 16) cp16(st.z + (q - 8) * 8, a.z + ((size_t)b * H + h) * D + (q - 8) * 8);
    else if (q < 24) cp16(st.xst + (q - 16) * 8, a.vp + (size_t)srow * a.vp_sst + (size_t)h * D + (q - 16) * 8);
    else if (q < 28) cp16(st.angp + (q - 24) * 8, a.angp + (size_t)b * a.angp_stride + (q - 24) * 8);
    else if (q < 36) cp16(st.angst + (q - 28) * 4, a.angle + (size_t)srow * a.ang_sst + (size_t)h * NA + (q - 28) * 4);
  }
}

#ifndef STEP_MINB
#define STEP_MINB 4
#endif
__global__ void __launch_bounds__(THREADS, STEP_MINB) mamba3_step_cuda_kernel(const Args a) {
  extern __shared__ __align__(128) char smem_raw[];
  Smem& S_ = *reinterpret_cast<Smem*>(smem_raw);
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int h = blockIdx.x % H, nb_stride = gridDim.x / H;   // grid is a multiple of H
  const int nitems = a.B * H;
  for (int i = t; i < 8 * LDB; i += THREADS) { S_.sBk[8 + i / LDB][i % LDB] = __float2bfloat16(0.f); if (i < 4 * LDB) S_.sCb[4 + i / LDB][i % LDB] = __float2bfloat16(0.f); }
  // per-head constants once per CTA
  { const int r = t >> 5, c = t & 31;
    cp16(S_.bias_k + r * S + c * 4, a.bias_k + ((size_t)r * H + h) * S + c * 4);
    cp16(S_.bias_q + r * S + c * 4, a.bias_q + ((size_t)r * H + h) * S + c * 4); }
  if (t < 64) { const int r = t >> 4, c = t & 15; cp16(S_.xproj + r * D + c * 4, a.xproj + ((size_t)r * H + h) * D + c * 4); cp16(S_.outproj + r * D + c * 4, a.outproj + ((size_t)r * H + h) * D + c * 4); }
  else { const int q = t - 64, r = q >> 4, c = q & 15; cp16(S_.zproj + r * D + c * 4, a.zproj + ((size_t)r * H + h) * D + c * 4); }
  int b0 = blockIdx.x / H;
  auto item_of = [&](int k) { return (b0 + k * nb_stride) * H + h; };   // token index b = b0 + k*nb_stride
  const int nk = (a.B - b0 + nb_stride - 1) / nb_stride;                 // items of this CTA
  if (b0 >= a.B) return;
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) { if (s < nk) issue_item(a, S_.st[s], item_of(s), t); cp_commit(); }
  for (int k = 0; k < nk; ++k) {
    const int item = item_of(k);
    { const int pf = k + NST - 1; if (pf < nk) issue_item(a, S_.st[pf % NST], item_of(pf), t); cp_commit(); }
    const int b = item / H;
    const int slot = a.slots[b]; const bool valid = slot >= 0 && slot < a.P;
    const float dt = a.dt[b * H + h], A = a.A[b * H + h], trap = a.trap[b * H + h];
    const float Dh = a.Dp[h];
    cp_wait<NST - 1>();
    __syncthreads();
    Stage& st = S_.st[k % NST];
    const float alpha = __expf(A * dt), gamma = trap * dt, beta = (1.f - trap) * dt * alpha;
    // ---- preamble: angles, bias + rotary on B/C (bf16-rounded)
    if (t < NA) { const float th = st.angst[t] + tanhf(bf2f(st.angp[t])) * dt * 3.14159265358979323846f; S_.sTheta[t] = th; __sincosf(fmodf(th, 6.283185307179586f), &S_.sSin[t], &S_.sCos[t]); }
    __syncthreads();
#ifndef V_NOPRE
    for (int i = t; i < R * (S / 2); i += THREADS) {
      const int r = i / (S / 2), j = i % (S / 2);
      float blo = bf2f(st.bm[r * S + j]) + S_.bias_k[r * S + j], bhi = bf2f(st.bm[r * S + j + 64]) + S_.bias_k[r * S + j + 64];
      float clo = bf2f(st.cm[r * S + j]) + S_.bias_q[r * S + j], chi = bf2f(st.cm[r * S + j + 64]) + S_.bias_q[r * S + j + 64];
      if (j < NA) {
        const float c = S_.sCos[j], s_ = S_.sSin[j];
        const float b0 = blo * c - bhi * s_, b1 = blo * s_ + bhi * c, c0 = clo * c - chi * s_, c1 = clo * s_ + chi * c;
        blo = b0; bhi = b1; clo = c0; chi = c1;
      }
      const __nv_bfloat16 z0 = __float2bfloat16(0.f);
      S_.sBk[r][j] = __float2bfloat16(blo); S_.sBk[r][j + 64] = __float2bfloat16(bhi);
      S_.sBk[4 + r][j] = st.bst[r * S + j]; S_.sBk[4 + r][j + 64] = st.bst[r * S + j + 64];
      S_.sCb[r][j] = __float2bfloat16(clo); S_.sCb[r][j + 64] = __float2bfloat16(chi);

    }
#endif
    __syncthreads();
    // ---- state update on tensor cores: h_new = alpha h + A Bk, A[p][k] = (k<4: gamma x[p] xproj[k][p]; 4<=k<8: beta xstate[p] xproj[k-4][p])
    const int qd = lane & 3, p0 = warp * 16 + (lane >> 2);
    uint32_t af[4];
    {
      float av[2][2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int p = p0 + e * 8;
        const float xp = bf2f(st.x[p]), xs = bf2f(st.xst[p]);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int k = 2 * qd + jj;
          const int r = k & 3;
          const float xpj = S_.xproj[r * D + p];
          av[e][jj] = (k < 4) ? gamma * xp * xpj : beta * xs * xpj;
        }
      }
      af[0] = pack_bf16(av[0][0], av[0][1]); af[1] = pack_bf16(av[1][0], av[1][1]); af[2] = 0u; af[3] = 0u;
    }
    float acc[16][4];
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      float2 h0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.state + p0 * LDS + n));
      float2 h1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(st.state + (p0 + 8) * LDS + n));
      acc[jt][0] = alpha * h0.x; acc[jt][1] = alpha * h0.y; acc[jt][2] = alpha * h1.x; acc[jt][3] = alpha * h1.y;
    }
    {
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {   // two n8 tiles per ldmatrix.x4.trans: (k0-7,n8), (k8-15,n8), (k0-7,n8+1), (k8-15,n8+1)
        uint32_t bfr[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(bfr[0]), "=r"(bfr[1]), "=r"(bfr[2]), "=r"(bfr[3]) : "r"(smem_u32(&S_.sBk[(mi & 1) * 8 + li][jp * 16 + (mi >> 1) * 8])));
        mma16816(acc[2 * jp], af, bfr);
        mma16816(acc[2 * jp + 1], af, bfr + 2);
      }
    }
    // write h_new (bf16) back into the tile: it is the pool value and the A operand of the C dot
#pragma unroll
    for (int jt = 0; jt < 16; ++jt) {
      const int n = jt * 8 + 2 * qd;
      *reinterpret_cast<__nv_bfloat162*>(st.state + p0 * LDS + n) = __floats2bfloat162_rn(acc[jt][0], acc[jt][1]);
      *reinterpret_cast<__nv_bfloat162*>(st.state + (p0 + 8) * LDS + n) = __floats2bfloat162_rn(acc[jt][2], acc[jt][3]);
    }
    __syncwarp();
    // ---- o[p][r] = sum_s h_new[p][s] C'[r][s] on tensor cores: A = this warp's 16 state rows (bf16, smem), B = sCb rows
    {
      float oc[4] = {0.f, 0.f, 0.f, 0.f};
      const int mi = lane >> 3, li = lane & 7;
#pragma unroll
      for (int kk = 0; kk < S; kk += 16) {
        uint32_t af[4], bfr[4];
        ldmatrix_x4(af, st.state + (warp * 16 + (mi & 1) * 8 + li) * LDS + kk + (mi >> 1) * 8);
        // B operand (n = r rows 0..7, k = s): x4 covers k halves kk..kk+7 and kk+8..kk+15 for the 8 n rows (+ next k16 unused half)
        ldmatrix_x4(bfr, &S_.sCb[(mi & 1) * 0 + li][kk + (mi & 1) * 8 + (mi >> 1) * 0]);
        mma16816(oc, af, bfr);
      }
      // oc: (row p0 = warp*16 + lane/4, n = 2(lane%4) + {0,1}) and (p0 + 8, same n); n = r for lane%4 < 2, zero otherwise
      float ys[2] = {0.f, 0.f};
      if (qd < 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {          // e: row p0 (c0,c1) or p0+8 (c2,c3)
          const int p = p0 + e * 8;
          const float xp = bf2f(st.x[p]), zp = bf2f(st.z[p]);
#pragma unroll
          for (int j = 0; j < 2; ++j) {        // n = 2qd + j = r
            const int r = 2 * qd + j;
            float orr = oc[e * 2 + j] + Dh * xp * S_.xproj[r * D + p];
            const float zv = zp * S_.zproj[r * D + p];
            orr *= __fdividef(zv, 1.f + __expf(-zv));
            ys[e] += orr * S_.outproj[r * D + p];
          }
        }
      }
      ys[0] += __shfl_xor_sync(0xffffffffu, ys[0], 1); ys[1] += __shfl_xor_sync(0xffffffffu, ys[1], 1);
      if (qd == 0) { a.y[((size_t)b * H + h) * D + p0] = __float2bfloat16(valid ? ys[0] : 0.f); a.y[((size_t)b * H + h) * D + p0 + 8] = __float2bfloat16(valid ? ys[1] : 0.f); }
    }
    __syncthreads();   // state tile fully updated in smem
#ifdef V_NOSTORE
    if (false) {
#else
    if (valid) {
#endif
      // coalesced store of the state tile: 1024 x 16B
      __nv_bfloat16* dst = a.ssm + (size_t)slot * a.ssm_sst + ((size_t)h * D) * S;
#pragma unroll
      for (int i = 0; i < 8; ++i) { const int c = t + i * THREADS, row = c >> 4, ch = c & 15; *reinterpret_cast<uint4*>(dst + c * 8) = *reinterpret_cast<const uint4*>(st.state + row * LDS + ch * 8); }
      if (t < 64) { const int r = t >> 4, c = t & 15; *reinterpret_cast<uint4*>(a.kp + (size_t)slot * a.kp_sst + ((size_t)r * H + h) * S + c * 8) = *reinterpret_cast<const uint4*>(&S_.sBk[r][c * 8]); }
      if (t < D) a.vp[(size_t)slot * a.vp_sst + (size_t)h * D + t] = st.x[t];
      if (t < NA) a.angle[(size_t)slot * a.ang_sst + (size_t)h * NA + t] = S_.sTheta[t];
    }
    __syncthreads();   // stage reusable by the prefetch of item + 2*grid
  }
}
}  // namespace

void mamba3_step_cuda(torch::Tensor ssm_pool, torch::Tensor k_pool, torch::Tensor v_pool, torch::Tensor angle_pool,
                    torch::Tensor A, torch::Tensor Bm, torch::Tensor Cm, torch::Tensor Dp, torch::Tensor x, torch::Tensor dt, torch::Tensor trap,
                    torch::Tensor xproj, torch::Tensor zproj, torch::Tensor outproj, torch::Tensor z, torch::Tensor bias_q, torch::Tensor bias_k,
                    torch::Tensor angle_proj, torch::Tensor slots, torch::Tensor y, int64_t ctas_per_sm) {
  Args a;
  a.ssm = reinterpret_cast<__nv_bfloat16*>(ssm_pool.data_ptr()); a.kp = reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr()); a.vp = reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr());
  a.angle = angle_pool.data_ptr<float>(); a.P = ssm_pool.size(0); a.A = A.data_ptr<float>(); a.Bm = reinterpret_cast<const __nv_bfloat16*>(Bm.data_ptr()); a.Cm = reinterpret_cast<const __nv_bfloat16*>(Cm.data_ptr());
  a.Dp = Dp.data_ptr<float>(); a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); a.dt = dt.data_ptr<float>(); a.trap = trap.data_ptr<float>();
  a.xproj = xproj.data_ptr<float>(); a.zproj = zproj.data_ptr<float>(); a.outproj = outproj.data_ptr<float>(); a.z = reinterpret_cast<const __nv_bfloat16*>(z.data_ptr());
  a.bias_q = bias_q.data_ptr<float>(); a.bias_k = bias_k.data_ptr<float>(); a.angp = reinterpret_cast<const __nv_bfloat16*>(angle_proj.data_ptr()); a.slots = slots.data_ptr<int>();
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); a.B = x.size(0);
  a.bc_bstride = Bm.stride(0); a.angp_stride = angle_proj.stride(0);
  a.ssm_sst = ssm_pool.stride(0); a.kp_sst = k_pool.stride(0); a.vp_sst = v_pool.stride(0); a.ang_sst = angle_pool.stride(0);
  TORCH_CHECK(ssm_pool.stride(1) == D * S && ssm_pool.stride(2) == S && ssm_pool.stride(3) == 1, "ssm_pool inner dims must be contiguous");
  TORCH_CHECK(k_pool.stride(1) == H * S && k_pool.stride(2) == S && k_pool.stride(3) == 1, "k_pool inner dims must be contiguous");
  TORCH_CHECK(v_pool.stride(1) == D && v_pool.stride(2) == 1 && angle_pool.stride(1) == NA && angle_pool.stride(2) == 1);
  TORCH_CHECK((a.ssm_sst * 2) % 16 == 0 && (a.kp_sst * 2) % 16 == 0 && (a.vp_sst * 2) % 16 == 0 && (a.ang_sst * 4) % 16 == 0, "pool slot strides must be 16B aligned");
  TORCH_CHECK(Bm.stride(1) == S && Bm.stride(2) == 1 && Cm.stride(0) == Bm.stride(0) && Cm.stride(1) == S && angle_proj.stride(1) == 1);
  TORCH_CHECK((Bm.stride(0) * 2) % 16 == 0 && (angle_proj.stride(0) * 2) % 16 == 0, "B/C batch stride and angle_proj row stride must be 16B aligned");
  TORCH_CHECK(ssm_pool.dtype() == torch::kBFloat16 && k_pool.dtype() == torch::kBFloat16 && v_pool.dtype() == torch::kBFloat16 && angle_pool.dtype() == torch::kFloat32);
  static bool attr = false; const int smem = sizeof(Smem);
  if (!attr) { cudaFuncSetAttribute(mamba3_step_cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr = true; }
  int nsm = 132; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int items = a.B * H; int grid = std::min<int>(items, nsm * (int)ctas_per_sm); grid = std::max(H, (grid / H) * H);
  mamba3_step_cuda_kernel<<<grid, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("step", &mamba3_step_cuda); }
