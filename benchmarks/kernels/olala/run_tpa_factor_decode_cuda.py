import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
from torch.utils.cpp_extension import load
ext = load(name="tpa_decode_v9_ext", sources=["tpa_factor_decode_v9.cu"],
           extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-gencode=arch=compute_90,code=sm_90", "-Xptxas", "-v"] + os.environ.get("TPA_DEFS", "").split(), extra_ldflags=["-lcuda"], verbose=False,
           build_directory=os.environ.get("TPA_BUILD", "build_v9"))
from tpa_factor_decode_prototype import make_cache, ref_attention, _tpa_decode_combine_kernel, HQ, HQP, D, ROW
def swizzle_cache(cache):
    """Apply the global chunk swizzle expected by the v7 kernel: within each Bk/Bv factor row (4 r x 16 chunks of 8),
    chunk c moves to (c & 8) | ((c & 7) ^ f) with f = ((t & 1) << 2) | r; t parity == slot parity (block size even)."""
    NB, bs, _ = cache.shape
    fac = cache[:, :, :1024].view(NB, bs, 2, 4, 16, 8).clone()
    c = torch.arange(16, device=cache.device).view(1, 1, 1, 1, 16, 1)
    r = torch.arange(4, device=cache.device).view(1, 1, 1, 4, 1, 1)
    par = (torch.arange(bs, device=cache.device) & 1).view(1, bs, 1, 1, 1, 1)
    src = (c & 8) | ((c & 7) ^ ((par << 2) | r))            # involution: out[..., c] = in[..., c ^ f]
    idx = src.expand(NB, bs, 2, 4, 16, 8)
    cache[:, :, :1024] = torch.gather(fac, 4, idx).reshape(NB, bs, 1024)
    return cache

def tpa_decode_cuda(q, cache, bt, sl, sm_scale, bs, tok_per_split=1020):
    B = q.shape[0]; max_len = int(sl.max().item()); SPLIT = max(1, -(-max_len // tok_per_split))
    part_o = torch.empty(B, SPLIT, HQP, D, dtype=torch.float32, device=q.device)
    part_m = torch.empty(B, SPLIT, HQP, dtype=torch.float32, device=q.device); part_l = torch.empty_like(part_m)
    ext.launch(q, cache, bt, sl, part_o, part_m, part_l, sm_scale, tok_per_split, bs)
    o = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=q.device)
    _tpa_decode_combine_kernel[(B,)](part_o, part_m, part_l, o, SPLIT=SPLIT, HQ=HQ, HQP=HQP, D=D, num_warps=4)
    return o
if __name__ == "__main__":
    print("smem bytes:", ext.smem_bytes()); ext.occupancy()
    torch.manual_seed(0); dev = "cuda"
    for (B, L) in [(2, 300), (4, 1000), (3, 1500), (2, 2100)]:
        cache, bt, sl, kd, vd = make_cache(B, L, 144, dev); swizzle_cache(cache); q = (torch.randn(B, HQ, D, device=dev) * 0.5).bfloat16()
        out = tpa_decode_cuda(q, cache, bt, sl, 1.0 / D ** 0.5, 144); torch.cuda.synchronize()
        ref = ref_attention(q, kd, vd, 1.0 / D ** 0.5)
        err = (out.float() - ref).abs().max().item(); scale = ref.abs().max().item()
        print(f"B={B} L={L}: max|diff|={err:.3e} (ref max {scale:.3f}, rel {err/scale:.2e})")
    if "--bench" in sys.argv:
        for (B, L) in [(64, 8192), (32, 24576)]:
            cache, bt, sl, kd, vd = make_cache(B, L, 144, dev); swizzle_cache(cache); q = (torch.randn(B, HQ, D, device=dev) * 0.5).bfloat16()
            for sp in (1020, 2040):
                for _ in range(3): tpa_decode_cuda(q, cache, bt, sl, 1.0 / D ** 0.5, 144, sp)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(10): tpa_decode_cuda(q, cache, bt, sl, 1.0 / D ** 0.5, 144, sp)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10 * 1e3
                gb = B * L * ROW * 2 / 1e9
                print(f"CUDA-v9 decode B={B} L={L} split={sp}: {dt:.3f} ms  ({gb:.2f} GB -> {gb/dt:.2f} TB/s)")
            del cache; torch.cuda.empty_cache()
