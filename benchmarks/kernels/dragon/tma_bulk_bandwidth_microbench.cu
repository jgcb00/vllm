// TMA bulk-copy streaming bandwidth microbenchmark: each CTA streams a contiguous slice with
// 1D cp.async.bulk copies of CHUNK bytes, NSTAGE-deep mbarrier pipeline, one producer thread.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(void* m, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(m)), "r"(c)); }
__device__ __forceinline__ void mbar_expect(void* m, uint32_t b) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(m)), "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(void* m, int ph) {
  asm volatile("{\n .reg .pred P;\n W:\n mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n @!P bra W;\n}\n" :: "r"(smem_u32(m)), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(void* d, const void* s, uint32_t b, void* m) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(smem_u32(d)), "l"(s), "r"(b), "r"(smem_u32(m)) : "memory");
}
template <int CHUNK, int NSTAGE, int NCOPY>
__global__ void tma_stream(const char* __restrict__ src, size_t per_cta, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  const char* base = src + (size_t)blockIdx.x * per_cta;
  const int n = per_cta / CHUNK;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mbar_wait(&full[st], ((it / NSTAGE) + 1) & 1); acc += ((const float*)(smem + st * CHUNK))[threadIdx.x]; }
    __syncthreads();   // stage free (all threads read it)
    if (threadIdx.x == 0) {
      mbar_expect(&full[st], CHUNK);
      for (int c = 0; c < NCOPY; ++c) bulk(smem + st * CHUNK + c * (CHUNK / NCOPY), base + (size_t)it * CHUNK + c * (CHUNK / NCOPY), CHUNK / NCOPY, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mbar_wait(&full[st], (it / NSTAGE) & 1); acc += ((const float*)(smem + st * CHUNK))[threadIdx.x]; }
  if (acc == 12345.f) sink[0] = acc;
}
template <int CHUNK, int NSTAGE, int NCOPY>
void run(torch::Tensor src, int ctas, int threads, torch::Tensor sink) {
  const size_t per_cta = src.numel() / ctas;
  cudaFuncSetAttribute(tma_stream<CHUNK, NSTAGE, NCOPY>, cudaFuncAttributeMaxDynamicSharedMemorySize, CHUNK * NSTAGE);
  tma_stream<CHUNK, NSTAGE, NCOPY><<<ctas, threads, CHUNK * NSTAGE, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), per_cta, sink.data_ptr<float>());
}
// our exact tile pattern: per tile, copy PREV bytes ending at the tile start (re-read) + MAIN bytes; stride MAIN
template <int PREV, int MAIN, int NSTAGE, bool REREAD>
__global__ void tma_rows(const char* __restrict__ src, size_t per_cta, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * per_cta + PREV;
  const int n = (per_cta - PREV) / MAIN;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mbar_wait(&full[st], ((it / NSTAGE) + 1) & 1); acc += ((const float*)(smem + st * STAGE))[threadIdx.x]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      mbar_expect(&full[st], REREAD ? STAGE : MAIN);
      if (REREAD) bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mbar_wait(&full[st], (it / NSTAGE) & 1); acc += ((const float*)(smem + st * STAGE))[threadIdx.x]; }
  if (acc == 12345.f) sink[0] = acc;
}
template <int PREV, int MAIN, int NSTAGE, bool REREAD>
void run_rows(torch::Tensor src, int ctas, int threads, torch::Tensor sink) {
  const size_t per_cta = src.numel() / ctas;
  cudaFuncSetAttribute(tma_rows<PREV, MAIN, NSTAGE, REREAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (PREV + MAIN) * NSTAGE);
  tma_rows<PREV, MAIN, NSTAGE, REREAD><<<ctas, threads, (PREV + MAIN) * NSTAGE, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), per_cta, sink.data_ptr<float>());
}
// arbitrary 2D grid addressing: CTA (x, y) streams `n` tiles starting at x*stride_x + y*stride_y
template <int PREV, int MAIN, int NSTAGE>
__global__ void tma_grid(const char* __restrict__ src, size_t stride_x, size_t stride_y, int n, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * stride_x + (size_t)blockIdx.y * stride_y + PREV;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f; unsigned uacc = 0u;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mbar_wait(&full[st], ((it / NSTAGE) + 1) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      mbar_expect(&full[st], STAGE);
      bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mbar_wait(&full[st], (it / NSTAGE) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
  atomicAdd(reinterpret_cast<unsigned*>(&sink[blockIdx.y * gridDim.x + blockIdx.x]), uacc); if (acc == 12345.f) sink[0] = acc;
}
template <int PREV, int MAIN, int NSTAGE>
__global__ void tma_grid_hog(const char* __restrict__ src, size_t stride_x, size_t stride_y, int n, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * stride_x + (size_t)blockIdx.y * stride_y + PREV;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f; unsigned uacc = 0u;
  float hog[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) hog[i] = (float)(threadIdx.x * 131 + i) * stride_x;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mbar_wait(&full[st], ((it / NSTAGE) + 1) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 128; ++i) hog[i] = hog[i] * 0.999f + (float)it;
    if (threadIdx.x == 0) {
      mbar_expect(&full[st], STAGE);
      bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mbar_wait(&full[st], (it / NSTAGE) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
  atomicAdd(reinterpret_cast<unsigned*>(&sink[blockIdx.y * gridDim.x + blockIdx.x]), uacc);
#pragma unroll
  for (int i = 0; i < 128; ++i) acc += hog[i];
  if (acc == 12345.f) sink[0] = acc;
}
template <int PREV, int MAIN, int NSTAGE>
__global__ void __launch_bounds__(160, 2) tma_grid_lb(const char* __restrict__ src, size_t stride_x, size_t stride_y, int n, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) unsigned long long full[NSTAGE];
  constexpr int STAGE = PREV + MAIN;
  const char* base = src + (size_t)blockIdx.x * stride_x + (size_t)blockIdx.y * stride_y + PREV;
  if (threadIdx.x == 0) { for (int i = 0; i < NSTAGE; ++i) mbar_init(&full[i], 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
  __syncthreads();
  float acc = 0.f; unsigned uacc = 0u;
  for (int it = 0; it < n; ++it) {
    const int st = it % NSTAGE;
    if (it >= NSTAGE) { mbar_wait(&full[st], ((it / NSTAGE) + 1) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
    __syncthreads();
    if (threadIdx.x == 0) {
      mbar_expect(&full[st], STAGE);
      bulk(smem + st * STAGE, base + (size_t)it * MAIN - PREV, PREV, &full[st]);
      bulk(smem + st * STAGE + PREV, base + (size_t)it * MAIN, MAIN, &full[st]);
    }
  }
  for (int it = max(0, n - NSTAGE); it < n; ++it) { const int st = it % NSTAGE; mbar_wait(&full[st], (it / NSTAGE) & 1); for (int w = threadIdx.x; w < MAIN / 4; w += blockDim.x) uacc += ((const unsigned*)(smem + st * STAGE + PREV))[w]; }
  atomicAdd(reinterpret_cast<unsigned*>(&sink[blockIdx.y * gridDim.x + blockIdx.x]), uacc); if (acc == 12345.f) sink[0] = acc;
}
void run_grid(torch::Tensor src, int64_t gx, int64_t gy, int64_t stride_x, int64_t stride_y, int64_t n, int threads, torch::Tensor sink, int64_t smem, int64_t carveout, bool lb) {
  if (lb) {
    cudaFuncSetAttribute(tma_grid_lb<2432, 38912, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (carveout >= 0) cudaFuncSetAttribute(tma_grid_lb<2432, 38912, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)carveout);
    tma_grid_lb<2432, 38912, 2><<<dim3(gx, gy), threads, smem, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), stride_x, stride_y, (int)n, sink.data_ptr<float>());
  } else {
    cudaFuncSetAttribute(tma_grid<2432, 38912, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (carveout >= 0) cudaFuncSetAttribute(tma_grid<2432, 38912, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)carveout);
    tma_grid<2432, 38912, 2><<<dim3(gx, gy), threads, smem, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), stride_x, stride_y, (int)n, sink.data_ptr<float>());
  }
}
void run_grid_hog(torch::Tensor src, int64_t gx, int64_t gy, int64_t stride_x, int64_t stride_y, int64_t n, int threads, torch::Tensor sink) {
  cudaFuncSetAttribute(tma_grid_hog<2432, 38912, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 82688);
  tma_grid_hog<2432, 38912, 2><<<dim3(gx, gy), threads, 82688, c10::cuda::getCurrentCUDAStream()>>>((const char*)src.data_ptr(), stride_x, stride_y, (int)n, sink.data_ptr<float>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid", &run_grid); m.def("grid_hog", &run_grid_hog);
  m.def("rows_reread", &run_rows<2432, 38912, 2, true>); m.def("rows_noreread", &run_rows<2432, 38912, 2, false>);
  m.def("rows_reread_s3", &run_rows<2432, 38912, 3, true>);
  m.def("c512_s4", &run<512, 4, 1>); m.def("c2k_s4", &run<2048, 4, 1>); m.def("c8k_s4", &run<8192, 4, 1>);
  m.def("c16k_s4", &run<16384, 4, 1>); m.def("c40k_s2", &run<40960, 2, 1>); m.def("c40k_s4", &run<40960, 4, 1>);
  m.def("c8k_s8", &run<8192, 8, 1>); m.def("c2k_s16", &run<2048, 16, 1>);
  m.def("c41k_s2_x2", &run<41344, 2, 2>);
  m.def("c41k_s2_x1", &run<41344, 2, 1>);
  m.def("c41k_s3_x1", &run<41344, 3, 1>);
}
