# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olala TPA-factor decode attention (the JIT CUDA kernel of the running GPU)
against dense attention over the K/V reconstructed from the factor rows."""

import pytest
import torch

from vllm.model_executor.layers.mamba.olala import tpa_factor as tf

requires_factor = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs an sm_80+ GPU",
)

HQ, HKV, R, D, ROWP = tf.HQ, tf.HKV, tf.RANK, tf.HEAD_DIM, tf.ROWP
GQ = HQ // HKV


def _make_cache(seq_lens, block_size, gen):
    """Random factor planes in a shuffled paged layout + their dense K/V."""
    nblk = [-(-s // block_size) for s in seq_lens]
    total = sum(nblk) + 3
    kplane = torch.zeros(total, block_size, ROWP, dtype=torch.bfloat16, device="cuda")
    vplane = torch.zeros_like(kplane)
    perm = torch.randperm(total, generator=gen, device="cpu").tolist()
    bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32)
    dense = []
    i = 0
    for b, s in enumerate(seq_lens):
        ids = perm[i : i + nblk[b]]
        i += nblk[b]
        bt[b, : nblk[b]] = torch.tensor(ids, dtype=torch.int32)
        planes = []
        for plane in (kplane, vplane):
            rows = torch.randn(s, ROWP, generator=gen, device="cpu") * 0.5
            rows[:, tf.OFF_C :] *= 0.5
            rows = rows.bfloat16().cuda()
            for t in range(0, s, block_size):
                plane[ids[t // block_size], : min(block_size, s - t)] = rows[t : t + block_size]
            Bf = rows[:, : R * D].float().view(s, R, D)
            C = rows[:, tf.OFF_C : tf.OFF_D].float().view(s, HKV, R)
            Dc = rows[:, tf.OFF_D :].float().view(s, HKV, R)
            Bprev = torch.cat([torch.zeros_like(Bf[:1]), Bf[:-1]])
            planes.append(torch.einsum("thr,trd->thd", C, Bf) + torch.einsum("thr,trd->thd", Dc, Bprev))
        dense.append(planes)
    return kplane, vplane, bt.cuda(), dense


def _ref(q, dense, scale, softcap):
    out = []
    for b, (k, v) in enumerate(dense):
        qb = q[b].float().view(HKV, GQ, D)
        s = torch.einsum("hgd,thd->hgt", qb, k) * scale
        if softcap > 0:
            s = softcap * torch.tanh(s / softcap)
        out.append(torch.einsum("hgt,thd->hgd", s.softmax(-1), v).reshape(HQ * D))
    return torch.stack(out)


def _split(seq_lens, nsplit):
    mx = max(seq_lens)
    tps = max((-(-mx // nsplit) + tf.TILE - 1) // tf.TILE * tf.TILE, tf.MIN_SPLIT_TOKENS)
    nact = -(-mx // tps)
    dev = "cuda"
    return (torch.tensor([tps], dtype=torch.int32, device=dev), torch.tensor([nact], dtype=torch.int32, device=dev))


@requires_factor
@pytest.mark.parametrize(
    "seq_lens,block_size,nsplit",
    [([1], 26, 1), ([5, 13, 14], 26, 2), ([300, 7, 1000], 52, 4), ([2100], 650, 16), ([64, 999], 13, 8)],
)
@pytest.mark.parametrize("softcap", [0.0, 150.0])
def test_factor_decode_matches_dense(seq_lens, block_size, nsplit, softcap):
    if not tf.factor_cache_enabled():
        pytest.skip("factor kernel unavailable (JIT build failed)")
    gen = torch.Generator().manual_seed(sum(seq_lens) + block_size)
    kplane, vplane, bt, dense = _make_cache(seq_lens, block_size, gen)
    q = (torch.randn(len(seq_lens), HQ * D, generator=gen) * 0.5).bfloat16().cuda()
    scale = D**-0.5 * 4
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = tf.factor_decode_attention(q, kplane, vplane, bt, sl, _split(seq_lens, nsplit), nsplit, scale, softcap, block_size)
    ref = _ref(q, dense, scale, softcap)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
