# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dragon's RMS norm, kept bit-compatible with the reference checkpoint.

Dragon stores every norm as ``<name>.norm.weight``: a ``DragonNorm`` wrapper
around a ``DragonRMSNorm``. vLLM's own ``RMSNorm`` cannot be substituted
because it would change the parameter path and does not implement the
zero-centered (``weight + 1``) variant Dragon trains with.
"""

import torch
from torch import nn


class DragonRMSNorm(nn.Module):
    """RMS norm with an optionally zero-centered gain."""

    def __init__(self, hidden_size: int, eps: float, zero_centered: bool):
        super().__init__()
        self.rms = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        init = torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)
        self.zero_centered = zero_centered

    @property
    def eps(self) -> float:
        return self.rms.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rms(x)
        if self.zero_centered:
            return y * (1.0 + self.weight)
        return y * self.weight


class DragonNorm(nn.Module):
    """Checkpoint-shaped wrapper: exposes the gain at ``<name>.norm.weight``."""

    def __init__(self, hidden_size: int, *, eps: float, zero_centered: bool):
        super().__init__()
        self.norm = DragonRMSNorm(hidden_size, eps=eps, zero_centered=zero_centered)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
