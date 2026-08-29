# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPA-factorized paged KV cache for Dragon's Differential-TPA layers.

Dragon's ``V`` layers build K/V as rank-``R`` tensor products, mix in the
previous token (token shift) and RMS-normalize K per head:

    k_t = norm_h(a_t k_{t-1} + (1 - a_t) A_t B_t / R)

Everything that depends on the head is a scalar per ``(t, h, r)``, so the
whole thing stays rank ``R`` per token::

    k_t[h] = sum_r Ck[t,h,r] Bk'[t,r] + sum_r Dk[t,h,r] Bk'[t-1,r]

with ``Bk' = (1 + w_norm) * B``, ``Ck = (1 - a) * inv_rms * A / R`` and
``Dk = a * inv_rms * A_prev / R``; V is the same without the norm. Instead of
the dense ``2 * 12 * 128`` bf16 per token, the cache keeps two 608-element
rows per token (``[B (4x128) | C (12x4) | D (12x4)]`` for K and for V):
2.5x less KV memory and 2.5x less decode-time KV traffic. Decode attention
runs a dedicated sm_90 kernel over the factors; prefill reconstructs the
dense K/V of its context and runs FlashAttention varlen.

Rows are stored in plain ``[r][d]`` order; the decode kernel streams them
with TMA tensor copies (128-byte swizzle) into the canonical wgmma layouts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

HQ, HKV, RANK, HEAD_DIM = 48, 12, 4, 128
HQP = 64  # query heads padded to the kernel's accumulator rows
ROWP = RANK * HEAD_DIM + 2 * HKV * RANK  # 608 bf16 per plane row
OFF_C, OFF_D = RANK * HEAD_DIM, RANK * HEAD_DIM + HKV * RANK
TILE = 13  # tokens per kernel tile; block sizes are multiples of it
TARGET_CTAS = 400  # ~3 CTAs per SM on GH200
MIN_SPLIT_TOKENS = 2 * TILE  # at least two tiles per active split; extra splits stay empty (cheap)


def factor_cache_enabled() -> bool:
    return os.environ.get("DRAGON_TPA_FACTOR", "1") != "0"


# --------------------------------------------------------------------------
# KV cache spec / backend / metadata
# --------------------------------------------------------------------------


def _factor_block_size(vllm_config: VllmConfig) -> int:
    """Smallest multiple of TILE whose page is at least the Mamba page.

    The hybrid allocator needs one page size for every group; the Mamba page
    (fixed by the SSM state, possibly already padded by the platform's
    block-size alignment) is padded up to ours, so the smallest factor page
    that covers it wastes nothing. Evaluated when the spec is requested, i.e.
    after ``update_block_size_for_backend`` has run.
    """
    import math

    cache = vllm_config.cache_config
    elem = torch.tensor([], dtype=vllm_config.model_config.dtype).element_size()
    row_bytes = 2 * ROWP * elem
    mamba_page = cache.mamba_page_size_padded or 0
    try:
        from vllm.model_executor.models.dragon import DragonForCausalLM

        shapes = DragonForCausalLM.get_mamba_state_shape_from_config(vllm_config)
        dtypes = DragonForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
        raw = sum(math.prod(sh) * torch.tensor([], dtype=dt).element_size() for sh, dt in zip(shapes, dtypes))
        mamba_page = max(mamba_page, raw)
    except Exception:  # noqa: BLE001 - fall back to the cache config
        pass
    if mamba_page == 0:
        return -(-cache.block_size // TILE) * TILE
    rows_needed = -(-mamba_page // row_bytes)
    return -(-rows_needed // TILE) * TILE


@dataclass
class DragonTPAFactorMetadata:
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    num_actual_tokens: int
    block_table: torch.Tensor  # (num_reqs, max_blocks) int32
    seq_lens: torch.Tensor  # (num_reqs,) int32
    slot_mapping: torch.Tensor  # (num_tokens,) int64, -1 for padding
    max_seq_len: int
    max_query_len: int
    query_start_loc_p_cpu: list[int] | None = None  # prefill rows, rebased to 0
    seq_lens_p_cpu: list[int] | None = None
    has_initial_state_p_cpu: list[bool] | None = None
    positions_p_cpu: list[int] | None = None  # first position of each prefill request
    tok_per_split: dict[int, torch.Tensor] | None = None  # per nsplit, device int32 (1,)

    def split_size(self, nsplit: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Device (tok_per_split, num_active_splits) for a static split count."""
        if self.tok_per_split is None:
            self.tok_per_split = {}
        t = self.tok_per_split.get(nsplit)
        if t is None:
            mx = self.seq_lens.max()
            tps = ((mx + nsplit - 1) // nsplit + TILE - 1) // TILE * TILE
            tps = torch.clamp_min(tps, MIN_SPLIT_TOKENS)
            nact = (mx + tps - 1) // tps
            t = (tps.to(torch.int32).reshape(1), nact.to(torch.int32).reshape(1))
            self.tok_per_split[nsplit] = t
        return t


class DragonTPAFactorBackend(AttentionBackend):
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "DRAGON_TPA_FACTOR"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("the factor cache layer never goes through AttentionImpl")

    @staticmethod
    def get_builder_cls() -> type["DragonTPAFactorMetadataBuilder"]:
        return DragonTPAFactorMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(TILE)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # K plane and V plane of factor rows.
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == ROWP


class DragonTPAFactorMetadataBuilder(AttentionMetadataBuilder[DragonTPAFactorMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "The TPA-factorized KV cache does not support speculative decoding; "
                "set DRAGON_TPA_FACTOR=0 to use the dense paged KV cache."
            )
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DragonTPAFactorMetadata:
        m = common_attn_metadata
        nd, np_, ndt, npt = split_decodes_and_prefills(m, decode_threshold=1)
        qsl_p = seq_p = init_p = pos_p = None
        if np_ > 0:
            qsl_cpu = m.query_start_loc_cpu[nd:]
            qsl_p = (qsl_cpu - qsl_cpu[0]).tolist()
            seq_p = m.seq_lens[nd:].tolist()
            computed = m.compute_num_computed_tokens()[nd:].tolist()
            init_p = [c > 0 for c in computed]
            pos_p = computed
        return DragonTPAFactorMetadata(
            num_decodes=nd,
            num_prefills=np_,
            num_decode_tokens=ndt,
            num_prefill_tokens=npt,
            num_actual_tokens=m.num_actual_tokens,
            block_table=m.block_table_tensor,
            seq_lens=m.seq_lens,
            slot_mapping=m.slot_mapping,
            max_seq_len=m.max_seq_len,
            max_query_len=m.max_query_len,
            query_start_loc_p_cpu=qsl_p,
            seq_lens_p_cpu=seq_p,
            has_initial_state_p_cpu=init_p,
            positions_p_cpu=pos_p,
        )


class DragonTPAFactorCache(nn.Module, AttentionLayerBase):
    """Owner of one V layer's factorized paged KV cache.

    Never runs ``forward``: ``DragonDiffTPAAttention`` writes the factor rows
    and runs the decode kernel on ``self.kv_cache`` directly.
    """

    kv_cache: torch.Tensor

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.dtype = vllm_config.model_config.dtype
        self._block_size: int | None = None
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DragonTPAFactorBackend

    @property
    def block_size(self) -> int:
        assert self._block_size is not None, "kv cache spec not requested yet"
        return self._block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self._block_size is None:
            self._block_size = _factor_block_size(vllm_config)
        return FullAttentionSpec(
            block_size=self._block_size,
            num_kv_heads=1,
            head_size=ROWP,
            dtype=self.dtype,
        )


# --------------------------------------------------------------------------
# Factor row construction
# --------------------------------------------------------------------------


def swizzle_b(b: torch.Tensor) -> torch.Tensor:
    """Factor rows are stored in plain [r][d] order; the decode kernel's TMA copies apply the 128B swizzle."""
    return b


def factor_rows(
    A: torch.Tensor,  # (N, H, R)
    A_prev: torch.Tensor,  # (N, H, R)
    B: torch.Tensor,  # (N, R, D)
    a: torch.Tensor,  # (N, H) shift gate (already zero at document starts)
    inv: torch.Tensor | None,  # (N, H) 1/rms of the shifted K, or None for V
    w: torch.Tensor | None,  # (D,) norm gain folded into B, or None
) -> torch.Tensor:
    """Build (N, ROWP) factor rows ``[B' | C | D]`` for one plane."""
    n = A.shape[0]
    scale = (1.0 - a) / RANK
    dscale = a / RANK
    if inv is not None:
        scale = scale * inv
        dscale = dscale * inv
    C = (A.float() * scale.unsqueeze(-1)).to(B.dtype)
    Dc = (A_prev.float() * dscale.unsqueeze(-1)).to(B.dtype)
    Bp = B if w is None else (B.float() * w.float()).to(B.dtype)
    return torch.cat([swizzle_b(Bp).reshape(n, RANK * HEAD_DIM), C.reshape(n, -1), Dc.reshape(n, -1)], dim=1)


def write_rows(plane: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Scatter rows into a (num_blocks, block_size, ROWP) plane (any block stride); slots < 0 are skipped."""
    valid = slots >= 0
    if not bool(valid.all()):
        slots = slots[valid]
        rows = rows[valid]
    bs = plane.shape[1]
    slots = slots.long()
    plane[slots // bs, slots % bs] = rows.to(plane.dtype)


def reconstruct_dense(plane: torch.Tensor, blocks: torch.Tensor, length: int) -> torch.Tensor:
    """Dense (length, H, D) K or V of one request from its factor rows."""
    return reconstruct_dense_batched(plane, blocks.unsqueeze(0), [length])


def reconstruct_dense_batched(plane: torch.Tensor, block_tables: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Dense (sum(lengths), H, D) K or V of several requests, concatenated.

    ``block_tables[r]`` is request ``r``'s block table row; one gather over
    the plane, one unswizzle and two einsums for the whole batch.
    """
    bs = plane.shape[1]
    dev = plane.device
    total = sum(lengths)
    len_t = torch.tensor(lengths, dtype=torch.int64, device=dev)
    starts = torch.cumsum(len_t, 0) - len_t
    r_idx = torch.repeat_interleave(torch.arange(len(lengths), device=dev), len_t, output_size=total)
    t_idx = torch.arange(total, device=dev) - starts[r_idx]
    blk = block_tables[r_idx, t_idx // bs].long()
    rows = plane[blk, t_idx % bs]  # (total, ROWP)
    B = swizzle_b(rows[:, :OFF_C].reshape(total, RANK, HEAD_DIM)).float()
    C = rows[:, OFF_C:OFF_D].reshape(total, HKV, RANK).float()
    Dc = rows[:, OFF_D:].reshape(total, HKV, RANK).float()
    Bprev = torch.cat([torch.zeros_like(B[:1]), B[:-1]], dim=0)
    Bprev = Bprev.masked_fill((t_idx == 0).view(total, 1, 1), 0.0)  # no previous token at a request start
    out = torch.einsum("lhr,lrd->lhd", C, B) + torch.einsum("lhr,lrd->lhd", Dc, Bprev)
    return out.to(plane.dtype)


# --------------------------------------------------------------------------
# Fused decode row writer (one program per (token, kv head))
# --------------------------------------------------------------------------


@triton.jit
def _factor_decode_write_kernel(
    q_ptr, sq_n,                        # q_all (N, Hq*D)
    ak_ptr, av_ptr, sa_n,               # A_k / A_v (N, Hkv*R)
    bk_ptr, bv_ptr, sb_n,               # B_k / B_v (N, R*D)
    alk_ptr, alv_ptr, sal_n,            # shift logits (N, Hkv)
    pos_ptr, pslots_ptr, cslots_ptr,    # positions, pool slots, cache slots (int64, -1 = padding)
    kpool_ptr, vpool_ptr, sp_n, sp_h, P,
    akpool_ptr, avpool_ptr, sap_n, sap_h,
    qw_ptr, kw_ptr, eps,
    scaler_ptr, wsize,
    kplane_ptr, vplane_ptr, BS, bstride,   # planes (num_blocks, block_size, ROWP): rows contiguous, block stride `bstride`
    qo_ptr,                             # (N, Hq*D)
    HKV: tl.constexpr, GQ: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    ROWP_: tl.constexpr, OFF_C_: tl.constexpr, OFF_D_: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)
    offr = tl.arange(0, R)

    # ---- rank-R K/V reconstruction (dense rows needed for the norm + pool)
    k = tl.zeros([D], dtype=tl.float32)
    v = tl.zeros([D], dtype=tl.float32)
    for r in range(R):
        ak_r = tl.load(ak_ptr + n * sa_n + h * R + r).to(tl.float32)
        av_r = tl.load(av_ptr + n * sa_n + h * R + r).to(tl.float32)
        k += ak_r * tl.load(bk_ptr + n * sb_n + r * D + offs).to(tl.float32)
        v += av_r * tl.load(bv_ptr + n * sb_n + r * D + offs).to(tl.float32)
    k = (k.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)
    v = (v.to(tl.bfloat16).to(tl.float32) / R).to(tl.bfloat16).to(tl.float32)

    # ---- token shift against the one-token pool
    slot = tl.load(pslots_ptr + n)
    srow = tl.minimum(tl.maximum(slot, 0), P - 1).to(tl.int64)
    kp = tl.load(kpool_ptr + srow * sp_n + h * sp_h + offs).to(tl.float32)
    a_k = tl.sigmoid(tl.load(alk_ptr + n * sal_n + h).to(tl.float32))
    a_v = tl.sigmoid(tl.load(alv_ptr + n * sal_n + h).to(tl.float32))
    pos = tl.load(pos_ptr + n)
    a_k = tl.where(pos == 0, 0.0, a_k)
    a_v = tl.where(pos == 0, 0.0, a_v)
    ks = (a_k * kp + (1.0 - a_k) * k).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(ks * ks, axis=0) / D + eps)
    ak_prev = tl.load(akpool_ptr + srow * sap_n + h * sap_h + offr).to(tl.float32)
    av_prev = tl.load(avpool_ptr + srow * sap_n + h * sap_h + offr).to(tl.float32)
    ak_cur = tl.load(ak_ptr + n * sa_n + h * R + offr).to(tl.float32)
    av_cur = tl.load(av_ptr + n * sa_n + h * R + offr).to(tl.float32)
    ak_prev = tl.where(pos == 0, 0.0, ak_prev)
    av_prev = tl.where(pos == 0, 0.0, av_prev)

    # pool update: raw K/V and A of this token
    tl.store(kpool_ptr + srow * sp_n + h * sp_h + offs, k.to(kpool_ptr.dtype.element_ty))
    tl.store(vpool_ptr + srow * sp_n + h * sp_h + offs, v.to(vpool_ptr.dtype.element_ty))
    tl.store(akpool_ptr + srow * sap_n + h * sap_h + offr, ak_cur.to(akpool_ptr.dtype.element_ty))
    tl.store(avpool_ptr + srow * sap_n + h * sap_h + offr, av_cur.to(avpool_ptr.dtype.element_ty))

    # ---- factor coefficients of this head
    cslot = tl.load(cslots_ptr + n)
    valid = cslot >= 0
    cs = tl.maximum(cslot, 0).to(tl.int64)
    crow = (cs // BS) * bstride + (cs % BS) * ROWP_
    ck = ak_cur * ((1.0 - a_k) * inv / R)
    dk = ak_prev * (a_k * inv / R)
    cv = av_cur * ((1.0 - a_v) / R)
    dv = av_prev * (a_v / R)
    m_r = valid & (offr >= 0)
    tl.store(kplane_ptr + crow + OFF_C_ + h * R + offr, ck.to(kplane_ptr.dtype.element_ty), mask=m_r)
    tl.store(kplane_ptr + crow + OFF_D_ + h * R + offr, dk.to(kplane_ptr.dtype.element_ty), mask=m_r)
    tl.store(vplane_ptr + crow + OFF_C_ + h * R + offr, cv.to(vplane_ptr.dtype.element_ty), mask=m_r)
    tl.store(vplane_ptr + crow + OFF_D_ + h * R + offr, dv.to(vplane_ptr.dtype.element_ty), mask=m_r)

    # ---- factor rows: heads 0..R-1 write Bk' row h (norm gain folded), heads R..2R-1 write Bv row h-R
    kw = (1.0 + tl.load(kw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if h < R:
        b = tl.load(bk_ptr + n * sb_n + h * D + offs).to(tl.float32)
        tl.store(kplane_ptr + crow + h * D + offs, (b * kw).to(kplane_ptr.dtype.element_ty), mask=valid & (offs >= 0))
    elif h < 2 * R:
        hr = h - R
        b = tl.load(bv_ptr + n * sb_n + hr * D + offs)
        tl.store(vplane_ptr + crow + hr * D + offs, b.to(vplane_ptr.dtype.element_ty), mask=valid & (offs >= 0))

    # ---- q heads of this kv group: norm + scalable-softmax scale (eager rounding points)
    qw = (1.0 + tl.load(qw_ptr + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    p1 = tl.maximum(pos.to(tl.float32) + 1.0, 1.0)
    if wsize > 0:
        p1 = tl.minimum(p1, wsize)
    log_pos = tl.log(p1).to(tl.bfloat16).to(tl.float32)
    for j in range(GQ):
        hq = h * GQ + j
        q = tl.load(q_ptr + n * sq_n + hq * D + offs).to(tl.float32)
        inv_q = 1.0 / tl.sqrt(tl.sum(q * q, axis=0) / D + eps)
        qn = (q * inv_q).to(tl.bfloat16).to(tl.float32)
        qn = (qn * qw).to(tl.bfloat16).to(tl.float32)
        sc = tl.load(scaler_ptr + hq).to(tl.bfloat16).to(tl.float32)
        scale = (sc * log_pos).to(tl.bfloat16).to(tl.float32)
        tl.store(qo_ptr + n * (HKV * GQ * D) + hq * D + offs, (qn * scale).to(tl.bfloat16))


def factor_decode_write(
    q_all, A_k, A_v, B_k, B_v, alpha_k, alpha_v, positions, pool_slots, cache_slots,
    k_pool, v_pool, ak_pool, av_pool, q_w, k_w, eps, scaler, wsize, kplane, vplane, gq,
) -> torch.Tensor:
    n = q_all.shape[0]
    q_out = torch.empty(n, HQ * HEAD_DIM, dtype=q_all.dtype, device=q_all.device)
    _factor_decode_write_kernel[(n, HKV)](
        q_all, q_all.stride(0),
        A_k, A_v, A_k.stride(0),
        B_k, B_v, B_k.stride(0),
        alpha_k, alpha_v, alpha_k.stride(0),
        positions, pool_slots, cache_slots,
        k_pool, v_pool, k_pool.stride(0), k_pool.stride(1), k_pool.shape[0],
        ak_pool, av_pool, ak_pool.stride(0), ak_pool.stride(1),
        q_w, k_w, eps, scaler, float(wsize),
        kplane, vplane, kplane.shape[1], kplane.stride(0), q_out,
        HKV=HKV, GQ=gq, R=RANK, D=HEAD_DIM, ROWP_=ROWP, OFF_C_=OFF_C, OFF_D_=OFF_D,
        num_warps=2,
    )
    return q_out


# --------------------------------------------------------------------------
# Decode attention: JIT CUDA kernel + split combine
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_kernel():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "tpa_factor_decode.cu")
    build_dir = os.environ.get(
        "DRAGON_TPA_BUILD_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "dragon_tpa_factor"),
    )
    os.makedirs(build_dir, exist_ok=True)
    ext = load(
        name="dragon_tpa_factor_decode",
        sources=[src],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a"],
        extra_ldflags=["-lcuda"],
        build_directory=build_dir,
        verbose=False,
    )
    assert ext.TILE == TILE
    return ext


@triton.jit
def _factor_combine_kernel(part_o_ptr, part_m_ptr, part_l_ptr, nactive_ptr, o_ptr,
                           SPLIT: tl.constexpr, HQ_: tl.constexpr, HQP_: tl.constexpr, HB: tl.constexpr, D: tl.constexpr,
                           CHUNK: tl.constexpr):
    """Merge the per-split (o, m, l) of one request; program = (request, block of HB heads).

    Only the first ``*nactive_ptr`` splits hold data (static grid, dynamic split
    length); they are visited in unrolled chunks of CHUNK so the loads of a chunk
    are in flight together (the merge is latency-bound)."""
    b = tl.program_id(0)
    offs_h = tl.program_id(1) * HB + tl.arange(0, HB)
    offs_d = tl.arange(0, D)
    hmask = offs_h < HQ_
    nactive = tl.minimum(tl.load(nactive_ptr), SPLIT)
    m = tl.full([HB], float("-inf"), tl.float32)
    for s0 in range(0, nactive, CHUNK):
        for j in tl.static_range(CHUNK):
            s = s0 + j
            m = tl.maximum(m, tl.load(part_m_ptr + (b * SPLIT + s) * HQP_ + offs_h, mask=hmask & (s < nactive), other=float("-inf")))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    l = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, D], tl.float32)
    for s0 in range(0, nactive, CHUNK):
        for j in tl.static_range(CHUNK):
            s = s0 + j
            ms = tl.load(part_m_ptr + (b * SPLIT + s) * HQP_ + offs_h, mask=hmask & (s < nactive), other=float("-inf"))
            w = tl.where(ms == float("-inf"), 0.0, tl.exp(ms - m_safe))
            l += w * tl.load(part_l_ptr + (b * SPLIT + s) * HQP_ + offs_h, mask=hmask & (s < nactive), other=0.0)
            po = tl.load(part_o_ptr + ((b * SPLIT + s) * HQP_ + offs_h[:, None]) * D + offs_d[None, :],
                         mask=hmask[:, None] & (w[:, None] > 0), other=0.0)
            acc += w[:, None] * po
    l_safe = tl.where(l == 0.0, 1.0, l)
    tl.store(o_ptr + b * HQ_ * D + offs_h[:, None] * D + offs_d[None, :], (acc / l_safe[:, None]).to(tl.bfloat16),
             mask=hmask[:, None])


def num_splits_for_batch(batch: int) -> int:
    """KV splits per request: ~TARGET_CTAS CTAs in total, rounded up to a power
    of two so the combine kernel has at most 8 specializations (all compiled
    during cudagraph capture)."""
    want = max(1, min(128, -(-TARGET_CTAS // max(batch, 1))))
    return 1 << (want - 1).bit_length()


def factor_decode_attention(
    q: torch.Tensor,  # (B, HQ*D) bf16, normalized and scaled
    kplane: torch.Tensor,
    vplane: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    split: tuple[torch.Tensor, torch.Tensor],  # (tok_per_split, num_active_splits) device int32
    nsplit: int,
    sm_scale: float,
    softcap: float,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    ext = _load_kernel()
    B = q.shape[0]
    part_o = torch.empty(B, nsplit, HQP, HEAD_DIM, dtype=torch.float32, device=q.device)
    part_m = torch.empty(B, nsplit, HQP, dtype=torch.float32, device=q.device)
    part_l = torch.empty_like(part_m)
    bt = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    sl = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    tok_per_split, nactive = split
    ext.launch(q.view(B, HQ, HEAD_DIM), kplane, vplane, bt, sl, tok_per_split, part_o, part_m, part_l,
               float(sm_scale), float(softcap), int(block_size))
    if out is None:
        out = torch.empty(B, HQ * HEAD_DIM, dtype=q.dtype, device=q.device)
    assert out.is_contiguous() and out.shape == (B, HQ * HEAD_DIM)
    # 8 heads per program and up to 16 splits in flight: the merge of a single long request (128 splits) is latency-bound
    _factor_combine_kernel[(B, HQP // 8)](part_o, part_m, part_l, nactive, out, SPLIT=nsplit, HQ_=HQ, HQP_=HQP, HB=8, D=HEAD_DIM,
                                          CHUNK=min(16, nsplit), num_warps=4)
    return out
