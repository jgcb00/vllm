# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/jgcb00/mamba/blob/9a3daf5c488d9bf01d71441988de0293fd60ef0b/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py
# (Mamba-3, Dao AI Lab / Goombalab, Apache-2.0); forward/inference parts only.

"""Mamba-3 MIMO varlen prefill (inference only).

``mamba3_mimo`` is the forward of mamba_ssm's autograd wrapper restricted to
the packed varlen layout vLLM uses; ``mamba3_mimo_varlen_grouped`` is the
group-parallel exact prefill for long sequences (same contract).
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from .angle_dt import angle_dt_fwd
from .mimo_fwd_varlen import mamba_mimo_forward_varlen
from .segsum import compute_dacs_segsum_triton_varlen


@torch.no_grad()
def mamba3_mimo(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    MIMO_V: Tensor,
    MIMO_Z: Tensor,
    MIMO_Out: Tensor,
    Angles: Tensor,
    D: Tensor,
    Z: Tensor,
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    fuse_pregate_headwise_rms_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
):
    """Packed varlen Mamba-3 MIMO forward.

    Shapes (B = 1 packed batch): Q/K (B, S, R, 1, N), V/Z (B, S, H, P),
    ADT/DT/Trap (B, H, S), Angles (B, S, H, A), biases and MIMO projections
    (H, R, N|P), D (H,). ``Input_States`` = (angle (NS, H, A), ssm
    (NS, H, P, N), k (NS, R, H, N), v (NS, H, P)).

    Returns ``Out`` or, with ``return_state``, ``(Out, Final_Angle,
    Final_SSM (NS, H, P, N), Final_K, Final_V)``.
    """
    assert cu_seqlens is not None, "vLLM uses the packed varlen layout only"
    assert chunk_size >= 8
    assert rotary_dim_divisor in (2, 4)
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out,
     outproj_norm_weight, Angles, D, Z) = (
        t.contiguous() if t is not None else None
        for t in (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
                  MIMO_Out, outproj_norm_weight, Angles, D, Z))
    if Input_States is not None:
        In_Angle, In_SSM, In_K, In_V = (t.contiguous() for t in Input_States)
    else:
        In_Angle = In_SSM = In_K = In_V = None

    Angles_Cumsum, angle_out_state = angle_dt_fwd(
        Angles, DT, init_state=In_Angle, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=cu_seqlens,
    )
    DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
        ADT, chunk_size, cu_seqlens=cu_seqlens)
    Out, Final_SSM, Final_K = mamba_mimo_forward_varlen(
        Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
        Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
        cu_seqlens=cu_seqlens,
        initial_states=(In_SSM, In_K, In_V) if In_SSM is not None else None,
        return_state=return_state,
        chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
        dtype=dtype,
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
    )
    if not return_state:
        return Out
    Final_Angle = torch.remainder(angle_out_state, 2 * torch.pi).contiguous()
    ends = cu_seqlens[1:].to(torch.long) - 1
    return (Out, Final_Angle, Final_SSM.permute(0, 1, 3, 2).contiguous(),
            Final_K.contiguous(), V[0, ends].contiguous())


@torch.no_grad()
def mamba3_mimo_varlen_grouped(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    MIMO_V: Tensor,
    MIMO_Z: Tensor,
    MIMO_Out: Tensor,
    Angles: Tensor,
    D: Tensor,
    Z: Tensor,
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    cu_seqlens: Tensor,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    group_tokens: int = 1536,
    min_split_tokens: int = 4096,
    cu_seqlens_cpu: Optional[Tensor] = None,
    fuse_pregate_headwise_rms_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
    threads: int = 128,
    num_stages: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Group-parallel exact prefill: same math as ``mamba3_mimo`` (varlen,
    ``return_state=True``), restructured for GPU occupancy on long prompts.

    The chunked-scan kernel launches one thread-block per (head, sequence)
    and walks the sequence's chunks serially, so a single long prefill uses
    only ``H`` blocks.  This wrapper splits every sequence longer than
    ``min_split_tokens`` into virtual sequences of ``group_tokens`` tokens
    and runs the kernel twice over the virtual layout:

    1. groups run with zero initial state, returning each group's local
       final state and last rotated K;
    2. group input states are chained with the closed form
       ``final = local_final + exp(sum ADT) * folded_input`` (the state
       recurrence is linear, with scalar-per-head decay), reusing the same
       boundary K/V fold the kernel applies for ``Input_States``;
    3. a second kernel pass consumes the chained states and produces exact
       outputs and final states.

    Cost is 2x kernel FLOPs for ~(S / group_tokens)x more parallelism —
    a large net win whenever ``H * num_groups`` was below GPU saturation.

    Group boundaries reuse the battle-tested chunked-prefill ``Input_States``
    machinery, so numerics match the single-pass kernel to bf16 rounding.

    Returns ``(Out, Final_Angle, Final_SSM, Final_K, Final_V)`` — the exact
    contract of ``mamba3_mimo(..., return_state=True)``.
    """
    assert cu_seqlens is not None, "grouped prefill is varlen-only"
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    tensors = (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
               MIMO_Out, Angles, D, Z)
    (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
     MIMO_Out, Angles, D, Z) = tuple(
        t.contiguous() if t is not None else None for t in tensors)

    if Input_States is not None:
        In_Angle, In_SSM, In_K, In_V = (
            t.contiguous() if t is not None else None for t in Input_States)
    else:
        In_Angle = In_SSM = In_K = In_V = None

    # ---- virtual group layout ------------------------------------------------
    # cu_seqlens_cpu (when given) avoids a device sync for the host-side split.
    cu_list = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
    ns_true = len(cu_list) - 1
    v_bounds = [0]           # virtual cu_seqlens
    seq_of_group: list[int] = []
    last_group_rows = [0] * ns_true
    for s in range(ns_true):
        start, end = cu_list[s], cu_list[s + 1]
        length = end - start
        if length >= min_split_tokens and length > group_tokens:
            starts = list(range(start, end, group_tokens))
        else:
            starts = [start]
        for g in starts:
            v_bounds.append(min(g + group_tokens, end) if len(starts) > 1 else end)
            seq_of_group.append(s)
            last_group_rows[s] = len(seq_of_group) - 1

    if len(seq_of_group) == ns_true:  # nothing split: single-pass fast path
        Angles_Cumsum, angle_out_state = angle_dt_fwd(
            Angles, DT, init_state=In_Angle, chunk_size=chunk_size,
            return_output_state=True, cu_seqlens=cu_seqlens,
        )
        Final_Angle = torch.remainder(angle_out_state, 2 * torch.pi).contiguous()
        DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
            ADT, chunk_size, cu_seqlens=cu_seqlens)
        o, h, k_fin = mamba_mimo_forward_varlen(
            Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
            Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
            chunk_size, rotary_dim_divisor, dtype, cu_seqlens=cu_seqlens,
            return_state=True,
            initial_states=(In_SSM, In_K, In_V) if In_SSM is not None else None,
            fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
            outproj_norm_weight=outproj_norm_weight,
            outproj_norm_eps=outproj_norm_eps,
            threads=threads, num_stages=num_stages,
        )
        ends = cu_seqlens[1:].to(torch.long) - 1
        return (o, Final_Angle, h.permute(0, 1, 3, 2).contiguous(),
                k_fin.contiguous(), V[0, ends].contiguous())

    v_cu = torch.tensor(v_bounds, dtype=torch.int32, device=cu_seqlens.device)
    nsv = len(seq_of_group)

    # ---- rotary phases: same two-pass trick over the virtual layout ---------
    # angle_dt walks chunks serially per (head, sequence), so it has the same
    # occupancy cliff as the scan kernel. Pass 1 gets per-group phase totals
    # (mod 2*pi is kept bounded inside the kernel), a tiny chain turns them
    # into per-group phase offsets, and pass 2 emits the continuous cumsum.
    TWO_PI = 2 * torch.pi
    _, ang_local = angle_dt_fwd(
        Angles, DT, init_state=None, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=v_cu,
    )
    # Phase offsets have no decay, so the chain is an exclusive cumsum per
    # true sequence (group phases are bounded by the kernel's mod 2*pi, and
    # there are few groups, so fp32 cumsum is exact enough).
    first_rows = [0] + [last_group_rows[s - 1] + 1 for s in range(1, ns_true)]
    ang_in = torch.empty_like(ang_local)
    for s in range(ns_true):
        sl = slice(first_rows[s], last_group_rows[s] + 1)
        base = In_Angle[s].float() if In_Angle is not None \
            else torch.zeros_like(ang_local[0])
        excl = torch.cat([torch.zeros_like(ang_local[:1]),
                          torch.cumsum(ang_local[sl][:-1], dim=0)])
        ang_in[sl] = torch.remainder(base + excl, TWO_PI)
    Angles_Cumsum, ang_state2 = angle_dt_fwd(
        Angles, DT, init_state=ang_in, chunk_size=chunk_size,
        return_output_state=True, cu_seqlens=v_cu,
    )

    DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton_varlen(
        ADT, chunk_size, cu_seqlens=v_cu)

    # ---- pass 1: local group states only (zero input, no output compute) ----
    _, h1, k1 = mamba_mimo_forward_varlen(
        Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
        Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
        chunk_size, rotary_dim_divisor, dtype, cu_seqlens=v_cu,
        return_state=True, initial_states=None, compute_outputs=False,
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
        threads=threads, num_stages=num_stages,
    )

    # ---- chain group states (linear recurrence, closed form) ----------------
    H_, P_ = V.shape[-2], V.shape[-1]
    N_ = Q.shape[-1]
    R_ = Q.shape[2]
    dev = V.device

    # Per-group total decay exp(sum ADT) [nsv, H] and boundary trap scale
    # DT[first] * sigmoid(-Trap[first]) [nsv, H] — the same quantities the
    # kernel derives per chunk / at Input_States fold time.
    adt_cs = ADT[0].float().cumsum(-1)                      # [H, S]
    starts_t = torch.tensor(v_bounds[:-1], device=dev, dtype=torch.long)
    ends_t = torch.tensor(v_bounds[1:], device=dev, dtype=torch.long)
    tot = adt_cs[:, ends_t - 1] - torch.where(
        starts_t > 0, adt_cs[:, (starts_t - 1).clamp(min=0)],
        torch.zeros_like(adt_cs[:, :1]))
    decay = tot.exp().t().contiguous()                      # [nsv, H]
    bscale = (DT[0, :, starts_t].float()
              * torch.sigmoid(-Trap[0, :, starts_t].float())).t()  # [nsv, H]

    mimo_v_f = MIMO_V.float()                               # [H, R, P]
    ssm_in = torch.zeros(nsv, H_, N_, P_, device=dev, dtype=torch.float32)
    k_in = torch.zeros(nsv, R_, H_, N_, device=dev, dtype=k1.dtype)
    v_in = torch.zeros(nsv, H_, P_, device=dev, dtype=V.dtype)

    # k/v inputs are state-independent, so they are known for every group up
    # front: pass-1 finals for continuation groups, Input_States for firsts.
    first_set = set(first_rows)
    cont = [g for g in range(nsv) if g not in first_set]
    if cont:
        cont_t = torch.tensor(cont, device=dev, dtype=torch.long)
        k_in[cont_t] = k1[cont_t - 1]
        v_in[cont_t] = V[0, torch.tensor([v_bounds[g] - 1 for g in cont],
                                         device=dev, dtype=torch.long)]
    if In_SSM is not None:
        fr = torch.tensor(first_rows, device=dev, dtype=torch.long)
        ssm_in[fr] = In_SSM.float().transpose(-1, -2)  # [NS,H,P,N]->[NS,H,N,P]
        k_in[fr] = In_K
        v_in[fr] = In_V

    # pre[g] = local_final + decay * folded_kv; only the scalar-decay affine
    # accumulation stays sequential (one addcmul per group).
    fold = torch.einsum("grhn,ghp,hrp,gh->ghnp",
                        k_in.float(), v_in.float(), mimo_v_f, bscale)
    pre = h1 + decay.view(nsv, H_, 1, 1) * fold
    for g in cont:
        ssm_in[g] = torch.addcmul(pre[g - 1],
                                  decay[g - 1].view(H_, 1, 1), ssm_in[g - 1])

    # ---- pass 2: exact outputs with chained input states --------------------
    o2, h2, k2 = mamba_mimo_forward_varlen(
        Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out, Z, D, MIMO_Z,
        Angles_Cumsum, DA_CS, DA_CS_REV, DT, Trap, Segsum,
        chunk_size, rotary_dim_divisor, dtype, cu_seqlens=v_cu,
        return_state=True,
        initial_states=(ssm_in.transpose(-1, -2).contiguous(), k_in, v_in),
        fuse_pregate_headwise_rms_norm=fuse_pregate_headwise_rms_norm,
        outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
        threads=threads, num_stages=num_stages,
    )

    rows = torch.tensor(last_group_rows, device=dev, dtype=torch.long)
    ends = cu_seqlens[1:].to(torch.long) - 1
    Final_Angle = torch.remainder(ang_state2[rows], TWO_PI).contiguous()
    return (o2, Final_Angle,
            h2[rows].permute(0, 1, 3, 2).contiguous(),
            k2[rows].contiguous(),
            V[0, ends].contiguous())
