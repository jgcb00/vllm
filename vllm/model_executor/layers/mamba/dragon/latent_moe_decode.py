# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch path for Dragon's latent MoE (decode).

The generic modular MoE path (permute / align / two grouped gemms / unpermute
/ reduce) launches ~15 kernels per layer, which at batch <= 16 costs more in
launch floor than in bytes moved: per token only ``top_k`` experts of
384x768 are touched. This path computes the routed experts with two
gather-GEMV kernels (up + relu^2, down + routed weight + scale) and keeps
the shared expert and latent projections as plain GEMVs. Same math as the
runner: ``shared(x) + fc2(scale * sum_k w_k * E_k(fc1(x)))``.

The whole MoE call is the custom op ``vllm::dragon_latent_moe`` (layer
looked up by name, like the mixers): the small/generic dispatch on the
token count then happens inside the op, where torch.compile cannot
specialize it away.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.dragon.decode_gemv import decode_gemv
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

SMALL_BATCH_MAX_TOKENS = 16


@triton.jit
def _latent_moe_up_kernel(
    x_ptr,       # [M, D] bf16 latent input
    logits_ptr,  # [M, E] fp32 router logits
    bias_ptr,    # [E] fp32 e_score_correction_bias
    ids_ptr,     # [M*K] int32 out (written by the nb == 0 programs)
    tw_ptr,      # [M*K] fp32 out routing weights (normalized)
    w1_ptr,      # [E, N, D] bf16
    h_ptr,       # [M*K, N] bf16 out
    E: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    K: tl.constexpr,
):
    """Expert up-projection + relu^2 for one (token, slot k, N-block).

    The router top-k (sigmoid scores, ranked by score + bias, weights
    normalized over the k winners — topk_sigmoid semantics) is recomputed by
    every program from the 256 logits: cheaper than a separate launch. The
    nb == 0 program of each (token, k) publishes the id and weight for the
    down kernel."""
    pid = tl.program_id(0)  # m * K + k
    nb = tl.program_id(1)
    m = pid // K
    k_slot = pid % K
    offs_e = tl.arange(0, E)
    sc = 1.0 / (1.0 + tl.exp(-tl.load(logits_ptr + m * E + offs_e)))
    sb = sc + tl.load(bias_ptr + offs_e)
    tot = 0.0
    e = 0
    wsel = 0.0
    for kk in range(K):
        mx = tl.max(sb, axis=0)
        idx = tl.min(tl.where(sb == mx, offs_e, E), axis=0)
        w = tl.sum(tl.where(offs_e == idx, sc, 0.0), axis=0)
        tot += w
        if kk == k_slot:
            e = idx
            wsel = w
        sb = tl.where(offs_e == idx, float("-inf"), sb)
    if nb == 0:
        tl.store(ids_ptr + pid, e.to(tl.int32))
        tl.store(tw_ptr + pid, wsel / tot)
    e64 = e.to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for d0 in range(0, D, BD):
        offs_d = d0 + tl.arange(0, BD)
        x = tl.load(x_ptr + m * D + offs_d).to(tl.float32)
        w = tl.load(w1_ptr + e64 * N * D + offs_n[:, None] * D + offs_d[None, :])
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    h = acc.to(tl.bfloat16).to(tl.float32)
    h = tl.maximum(h, 0.0)
    h = h * h
    tl.store(h_ptr + pid * N + offs_n, h.to(tl.bfloat16))


@triton.jit
def _latent_moe_down_partial_kernel(
    h_ptr,      # [M*K, N1] bf16
    ids_ptr,    # [M*K] int32
    tw_ptr,     # [M*K] fp32 routing weights
    w2_ptr,     # [E, D2, N1] bf16
    part_ptr,   # [M*K, D2] fp32 out (routing weight applied)
    N1: tl.constexpr,
    D2: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)  # m * K + k
    nb = tl.program_id(1)
    e = tl.load(ids_ptr + row).to(tl.int64)
    wk = tl.load(tw_ptr + row)
    offs_n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for d0 in range(0, N1, BD):
        offs_d = d0 + tl.arange(0, BD)
        h = tl.load(h_ptr + row * N1 + offs_d).to(tl.float32)
        w = tl.load(w2_ptr + e * D2 * N1 + offs_n[:, None] * N1 + offs_d[None, :])
        acc += tl.sum(w.to(tl.float32) * h[None, :], axis=1)
    tl.store(part_ptr + row * D2 + offs_n, acc * wk)


@triton.jit
def _latent_moe_epilogue_kernel(
    part_ptr,   # [M*K, D2] fp32 weighted expert partials
    s_ptr,      # [M, S] bf16 shared-expert pre-activation (from the fused GEMV)
    cat_ptr,    # [M, D2 + S] bf16 out: [scale * sum_k part | relu2(s)]
    scale,
    S_LD,       # leading dim of s rows
    D2: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
):
    m = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BS + tl.arange(0, BS)
    if nb * BS < D2:
        acc = tl.zeros([BS], dtype=tl.float32)
        for k in range(K):
            acc += tl.load(part_ptr + (m * K + k) * D2 + offs)
        tl.store(cat_ptr + m * (D2 + S) + offs, (acc * scale).to(tl.bfloat16))
    else:
        so = offs - D2
        v = tl.load(s_ptr + m * S_LD + so).to(tl.float32)
        v = tl.maximum(v, 0.0)
        tl.store(cat_ptr + m * (D2 + S) + offs, (v * v).to(tl.bfloat16))


def dragon_moe_small(
    hidden: torch.Tensor,         # [M, hidden] bf16
    router_logits: torch.Tensor,  # [M, E] fp32
    e_score_bias: torch.Tensor,   # [E] fp32
    w_in: torch.Tensor,           # [latent + shared_inter, hidden]  (fc1 ‖ shared_up)
    w_out: torch.Tensor,          # [hidden, latent + shared_inter]  (fc2 ‖ shared_down)
    w13: torch.Tensor,            # [E, inter, latent]
    w2: torch.Tensor,             # [E, latent, inter]
    top_k: int,
    routed_scale: float,
) -> torch.Tensor:
    """Small-batch latent MoE. ``w_in``/``w_out`` are the cached concatenations
    of the latent and shared-expert projections (see DragonLatentMoE)."""
    M = hidden.shape[0]
    dev = hidden.device
    E, N1, D1 = w13.shape
    D2 = w2.shape[1]
    S = w_in.shape[0] - D1

    topk_w = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    topk_ids = torch.empty(M, top_k, dtype=torch.int32, device=dev)

    x_in = decode_gemv(hidden, w_in)                 # [M, D1 + S]: latent | shared pre-act
    latent = x_in[:, :D1].contiguous()
    h = torch.empty(M * top_k, N1, dtype=hidden.dtype, device=dev)
    _latent_moe_up_kernel[(M * top_k, N1 // 16)](
        latent, router_logits, e_score_bias, topk_ids, topk_w, w13, h,
        E=E, D=D1, N=N1, BN=16, BD=64, K=top_k, num_warps=4,
    )
    part = torch.empty(M * top_k, D2, dtype=torch.float32, device=dev)
    _latent_moe_down_partial_kernel[(M * top_k, D2 // 8)](
        h, topk_ids, topk_w, w2, part, N1=N1, D2=D2, BN=8, BD=256, num_warps=4,
    )
    cat = torch.empty(M, D2 + S, dtype=hidden.dtype, device=dev)
    _latent_moe_epilogue_kernel[(M, (D2 + S) // 128)](
        part, x_in[:, D1:], cat, routed_scale, x_in.stride(0),
        D2=D2, S=S, BS=128, K=top_k, num_warps=1,
    )
    return decode_gemv(cat, w_out)                    # shared + fc2(routed), one GEMV


def _dragon_latent_moe(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return layer.forward_dispatch(hidden, router_logits)


def _dragon_latent_moe_fake(
    hidden: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(hidden)


direct_register_custom_op(
    op_name="dragon_latent_moe",
    op_func=_dragon_latent_moe,
    mutates_args=[],
    fake_impl=_dragon_latent_moe_fake,
)
