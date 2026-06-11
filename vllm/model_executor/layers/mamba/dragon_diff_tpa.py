# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dragon Differential-TPA attention mixer for vLLM.

Native port of ``DragonDifferentialTensorProductAttentionV2``.

Architecture:
    - TPA low-rank K/V reconstruction: ``key = (A_k @ B_k) / rank``
      (and similarly for V). ``A_k`` / ``A_v`` are per-head low-rank
      coefficients; ``B_k`` / ``B_v`` are shared head bases.
    - Optional token-shift EMA mixing of the *previous* token's raw K/V
      into the current token. The 1-token (k_last, v_last) buffer is
      managed by a side-channel ``DRAGON_DIFF_TPA`` mamba backend so that
      vLLM's paged state manager owns per-request slots.
    - Main softmax attention runs through ``vllm.attention.Attention``
      over the flat ``(num_tokens, H*D)`` tensors, using vLLM's paged KV.
    - Differential recombination: output heads are split into ``snr``
      signal heads and ``1`` noise head per noise-head group, then
      combined as ``sig - sigmoid(lambda) * noise``.

First-pass limitations:
    - No ``token_conv1d_attn`` (would need conv1d state).
    - No value embeddings (``use_ve``), no ``xsa``, no
      ``intra_doc_masking``, no per-layer sliding window.
    - Uses vLLM's default ``get_rope`` in place of Dragon's custom rope.
    - TP sharding: ``num_attention_heads``, ``num_noise_heads`` and
      ``num_kv_heads`` must all be divisible by ``tp_size``. ``cosnet``
      is not supported with TP > 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.dragon_diff_tpa_attn import (
    DragonDiffTPAMetadata,
)

from .dragon_mamba3 import _DragonRMSNorm


class _DragonTokenShiftState(nn.Module, MambaBase):
    """Side-channel owner of the per-request 1-token (k_last, v_last) buffer.

    Treated as a MambaBase layer so that vLLM's paged state manager
    allocates and shards its two state tensors alongside the real mamba
    states. The mixer never calls ``forward`` on this module — it reads
    and writes ``self.kv_cache`` directly.
    """

    kv_cache: tuple[torch.Tensor, ...]

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        vllm_config: VllmConfig,
        prefix: str,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.prefix = prefix
        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> str:
        return "dragon_diff_tpa"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.attn_last_kv_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return MambaStateShapeCalculator.attn_last_kv_state_shape(
            tp_world_size=self.tp_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )


class DragonDiffTPAAttention(nn.Module):
    """Dragon DifferentialTPA attention mixer (vLLM-native, tp=1)."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.prefix = prefix
        tp = get_tensor_model_parallel_world_size()
        self.tp_size = tp
        if tp > 1 and getattr(config, "cosnet", False):
            raise NotImplementedError(
                "DragonDiffTPAAttention: cosnet=True is not supported with "
                "tensor_parallel_size > 1 (would require a custom weight "
                "loader for the CosNet sidecar)."
            )

        for flag in ("token_conv1d_attn", "xsa", "intra_doc_masking"):
            if getattr(config, flag, False):
                raise NotImplementedError(
                    f"DragonDiffTPAAttention: config.{flag} not supported yet"
                )

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rank = config.tpa_rank
        self.qk_norm = config.qk_norm
        self.softcap = float(config.softcap_attn) if config.softcap_attn else 0.0
        self.num_attention_heads = config.num_attention_heads
        self.num_signal_heads = (
            config.num_signal_heads_diff
            if config.num_signal_heads_diff
            else self.num_attention_heads // 2
        )
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads
        assert self.num_signal_heads % self.num_noise_heads == 0
        self.snr = self.num_signal_heads // self.num_noise_heads
        self.num_kv_heads = self.num_noise_heads
        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if tp > 1:
            assert self.num_attention_heads % tp == 0, (
                f"num_attention_heads={self.num_attention_heads} not "
                f"divisible by tp={tp}"
            )
            assert self.num_noise_heads % tp == 0, (
                f"num_noise_heads={self.num_noise_heads} not divisible by "
                f"tp={tp}"
            )
            assert self.num_kv_heads % tp == 0, (
                f"num_kv_heads={self.num_kv_heads} not divisible by tp={tp}"
            )
        self.num_attention_heads_local = divide(self.num_attention_heads, tp)
        self.num_signal_heads_local = divide(self.num_signal_heads, tp)
        self.num_noise_heads_local = divide(self.num_noise_heads, tp)
        self.num_kv_heads_local = divide(self.num_kv_heads, tp)
        self.q_size_local = self.num_attention_heads_local * self.head_dim
        self.kv_size_local = self.num_kv_heads_local * self.head_dim

        self.token_shift = bool(getattr(config, "token_shift_attn", False))
        self.scalable_softmax = bool(getattr(config, "scalable_softmax", False))
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(
                torch.ones(
                    self.num_attention_heads_local, dtype=torch.float32
                )
            )
            set_weight_attrs(
                self.softmax_scaler,
                {"weight_loader": sharded_weight_loader(0)},
            )

        # Projections — all row-split along the head-dim of the output so
        # each rank owns its own slice of heads. W_B_k / W_B_v are the
        # shared TPA bases and are replicated across ranks.
        cp = lambda o, name: ColumnParallelLinear(  # noqa: E731
            input_size=self.hidden_size,
            output_size=o,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.{name}",
        )
        rep = lambda o, name: ReplicatedLinear(  # noqa: E731
            input_size=self.hidden_size,
            output_size=o,
            bias=False,
            prefix=f"{prefix}.{name}",
        )
        self.c_q = cp(self.q_size, "c_q")
        self.W_A_k = cp(self.num_kv_heads * self.rank, "W_A_k")
        self.W_A_v = cp(self.num_kv_heads * self.rank, "W_A_v")
        self.W_B_k = rep(self.rank * self.head_dim, "W_B_k")
        self.W_B_v = rep(self.rank * self.head_dim, "W_B_v")
        self.lambda_proj = cp(self.num_noise_heads, "lambda_proj")

        if self.token_shift:
            self.shift_proj_k = cp(self.num_kv_heads, "shift_proj_k")
            self.shift_proj_v = cp(self.num_kv_heads, "shift_proj_v")
            self.shift_state = _DragonTokenShiftState(
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                vllm_config=vllm_config,
                prefix=f"{prefix}.shift_state",
            )
        else:
            self.shift_state = None

        eps = config.norm_epsilon
        zc = getattr(config, "zero_centered_gamma", False)
        if self.qk_norm:
            self.q_norm = _DragonRMSNorm(self.head_dim, eps=eps, zero_centered=zc)
            self.k_norm = _DragonRMSNorm(self.head_dim, eps=eps, zero_centered=zc)

        # RoPE — use vLLM's default rope (Dragon's custom ``p-rope`` is a
        # fidelity TODO). Fall through rope_theta if present.
        rope_theta = float(getattr(config, "rope_theta", 0.0) or 0.0)
        if rope_theta > 0.0:
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                max_position=getattr(config, "max_position_embeddings", 131072),
                rope_parameters={
                    "rope_type": "default",
                    "base": rope_theta,
                    "rotary_dim": self.head_dim,
                },
            )
        else:
            self.rotary_emb = None

        # Attention scale: 1/sqrt(head_dim) unless use_completed_p is set,
        # in which case Dragon uses 1/head_dim.
        if getattr(config, "use_completed_p", False):
            scale = 1.0 / self.head_dim
        else:
            scale = self.head_dim ** -0.5

        cache_config = vllm_config.cache_config
        # ``Attention`` expects per-rank (post-TP) head counts.
        self.attn = Attention(
            num_heads=self.num_attention_heads_local,
            head_size=self.head_dim,
            scale=scale,
            num_kv_heads=self.num_kv_heads_local,
            cache_config=cache_config,
            logits_soft_cap=self.softcap if self.softcap > 0.0 else None,
            prefix=f"{prefix}.attn",
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        positions: torch.Tensor,        # (N,)
        hidden_states: torch.Tensor,    # (N, D)
    ) -> torch.Tensor:
        """Return differential attention output ``(N, num_signal_heads*D)``."""
        n = hidden_states.shape[0]

        q_all, _ = self.c_q(hidden_states)
        A_k_all, _ = self.W_A_k(hidden_states)
        A_v_all, _ = self.W_A_v(hidden_states)
        B_k_all, _ = self.W_B_k(hidden_states)
        B_v_all, _ = self.W_B_v(hidden_states)

        q = q_all.view(n, self.num_attention_heads_local, self.head_dim)
        A_k = A_k_all.view(n, self.num_kv_heads_local, self.rank)
        A_v = A_v_all.view(n, self.num_kv_heads_local, self.rank)
        B_k = B_k_all.view(n, self.rank, self.head_dim)
        B_v = B_v_all.view(n, self.rank, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank)  # (N, Hkv_local, D)
        v = torch.bmm(A_v, B_v).div_(self.rank)

        if self.token_shift:
            k, v = self._apply_token_shift(hidden_states, positions, k, v)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Flatten for rope / attention.
        q_flat = q.reshape(n, self.q_size_local)
        k_flat = k.reshape(n, self.kv_size_local)
        v_flat = v.reshape(n, self.kv_size_local)

        if self.rotary_emb is not None:
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)

        if self.scalable_softmax:
            # scalable-softmax: q <- s * log(max(pos+1, wsize)) * q per-head.
            # sliding_window_size == 0 disables the clamp.
            pos = (positions.float() + 1.0).clamp_min(1.0)
            wsize = float(getattr(self.config, "sliding_window_size", 0) or 0)
            log_pos = pos.clamp_max(wsize).log() if wsize > 0 else pos.log()
            scale = self.softmax_scaler.to(q_flat.dtype) * log_pos.to(
                q_flat.dtype
            ).unsqueeze(-1)                      # (N, H_local)
            q_flat = q_flat.view(
                n, self.num_attention_heads_local, self.head_dim
            )
            q_flat = q_flat * scale.unsqueeze(-1)
            q_flat = q_flat.reshape(n, self.q_size_local)

        attn_out = self.attn(q_flat, k_flat, v_flat)  # (N, H_local*D)

        # Differential recombination on the local head-shard.
        attn_out = attn_out.view(
            n, self.num_noise_heads_local, self.snr + 1, self.head_dim
        )
        sig = attn_out[:, :, :self.snr, :]          # (N, Hnoise_l, snr, D)
        noi = attn_out[:, :, self.snr:self.snr + 1, :]

        lam_all, _ = self.lambda_proj(hidden_states)
        lam = lam_all.view(n, self.num_noise_heads_local, 1, 1)
        out = sig - torch.sigmoid(lam) * noi        # (N, Hnoise_l, snr, D)
        return out.reshape(
            n, self.num_signal_heads_local * self.head_dim
        )

    # ------------------------------------------------------------------
    def _apply_token_shift(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the previous token's raw K/V into the current token.

        Uses the DRAGON_DIFF_TPA side-channel metadata to find per-request
        state slots and prefill chunk boundaries.
        """
        alpha_k_all, _ = self.shift_proj_k(hidden_states)
        alpha_v_all, _ = self.shift_proj_v(hidden_states)
        alpha_k = torch.sigmoid(alpha_k_all.float())
        alpha_v = torch.sigmoid(alpha_v_all.float())
        alpha_k = alpha_k.unsqueeze(-1).to(k.dtype)  # (N, Hkv_local, 1)
        alpha_v = alpha_v.unsqueeze(-1).to(v.dtype)

        fwd_ctx = get_forward_context()
        attn_md = fwd_ctx.attn_metadata
        if attn_md is None:
            # profile / dry run — treat as pure new-sequence shift.
            k_prev = torch.zeros_like(k)
            v_prev = torch.zeros_like(v)
            k_shift = alpha_k * k_prev + (1 - alpha_k) * k
            v_shift = alpha_v * v_prev + (1 - alpha_v) * v
            return k_shift, v_shift

        md: DragonDiffTPAMetadata = attn_md[self.shift_state.prefix]
        k_pool, v_pool = self.shift_state.kv_cache

        slots = md.state_indices_tensor
        nd, np_ = md.num_decodes, md.num_prefills
        ndt, npt = md.num_decode_tokens, md.num_prefill_tokens

        k_prev = torch.zeros_like(k)
        v_prev = torch.zeros_like(v)

        # -- Decode tokens: gather cached last, scatter current back. ----
        if nd > 0:
            dslots = slots[:nd]
            k_prev[:ndt] = k_pool[dslots].to(k.dtype)
            v_prev[:ndt] = v_pool[dslots].to(v.dtype)
            k_pool[dslots] = k[:ndt].to(k_pool.dtype)
            v_pool[dslots] = v[:ndt].to(v_pool.dtype)

        # -- Prefill tokens: shift-within-chunk with cached seed. --------
        if np_ > 0:
            pslots = slots[nd:nd + np_]
            # Prefer the builder's CPU mirrors (no per-layer GPU sync).
            qsl_cpu = getattr(md, "query_start_loc_p_cpu", None)
            qsl = (qsl_cpu if qsl_cpu is not None
                   else md.query_start_loc_p).tolist()
            has_init_cpu = getattr(md, "has_initial_state_cpu", None)
            if has_init_cpu is None:
                has_init_cpu = md.has_initial_state
            has_init = (
                has_init_cpu.tolist()
                if has_init_cpu is not None
                else [False] * np_
            )
            pslots_list = pslots.tolist()  # one sync for all requests
            k_pref = k[ndt:ndt + npt]
            v_pref = v[ndt:ndt + npt]
            kp = torch.empty_like(k_pref)
            vp = torch.empty_like(v_pref)
            for r in range(np_):
                s, e = qsl[r], qsl[r + 1]
                if e == s:
                    continue
                slot = int(pslots_list[r])
                if has_init[r]:
                    kp[s:s + 1] = k_pool[slot].to(k.dtype).unsqueeze(0)
                    vp[s:s + 1] = v_pool[slot].to(v.dtype).unsqueeze(0)
                else:
                    kp[s:s + 1] = 0
                    vp[s:s + 1] = 0
                if e - s > 1:
                    kp[s + 1:e] = k_pref[s:e - 1]
                    vp[s + 1:e] = v_pref[s:e - 1]
                # Cache this chunk's last raw K/V for the next chunk / decode.
                k_pool[slot] = k_pref[e - 1].to(k_pool.dtype)
                v_pool[slot] = v_pref[e - 1].to(v_pool.dtype)
            k_prev[ndt:ndt + npt] = kp
            v_prev[ndt:ndt + npt] = vp

        # Doc-boundary mask: positions == 0 zeros out both previous and alpha.
        doc_start = (positions == 0).view(-1, 1, 1)
        k_prev = k_prev.masked_fill(doc_start, 0)
        v_prev = v_prev.masked_fill(doc_start, 0)
        alpha_k = alpha_k.masked_fill(doc_start, 0)
        alpha_v = alpha_v.masked_fill(doc_start, 0)

        k_shift = alpha_k * k_prev + (1 - alpha_k) * k
        v_shift = alpha_v * v_prev + (1 - alpha_v) * v
        return k_shift, v_shift
