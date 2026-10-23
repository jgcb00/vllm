# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala small-batch latent MoE with int8 expert weights (experts_int8) must
match the same path over the dequantized bf16 weights."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala.latent_moe_decode import olala_moe_small


def _quant_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.float().abs().amax(dim=-1) / 127.0
    q = (w.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return q, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m", [1, 3, 8])
def test_int8_small_path_matches_dequantized(m: int):
    torch.manual_seed(m)
    E, K, H, D1, N1, S = 256, 6, 1536, 384, 768, 1536
    dev = "cuda"
    hidden = torch.randn(m, H, device=dev).bfloat16()
    logits = torch.randn(m, E, device=dev)
    bias = torch.randn(E, device=dev) * 0.1
    w_in = (torch.randn(D1 + S, H, device=dev) * H**-0.5).bfloat16()
    w_out = (torch.randn(H, D1 + S, device=dev) * (D1 + S) ** -0.5).bfloat16()
    w13 = (torch.randn(E, N1, D1, device=dev) * D1**-0.5).bfloat16()
    w2 = (torch.randn(E, D1, N1, device=dev) * N1**-0.5).bfloat16()
    q13, s13 = _quant_rows(w13)
    q2, s2 = _quant_rows(w2)
    dq13 = (q13.float() * s13.unsqueeze(-1)).bfloat16()
    dq2 = (q2.float() * s2.unsqueeze(-1)).bfloat16()

    out_q = olala_moe_small(hidden, logits, bias, w_in, w_out, q13, q2, K, 2.45, w13_scale=s13, w2_scale=s2)
    out_ref = olala_moe_small(hidden, logits, bias, w_in, w_out, dq13, dq2, K, 2.45)
    torch.testing.assert_close(out_q.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
