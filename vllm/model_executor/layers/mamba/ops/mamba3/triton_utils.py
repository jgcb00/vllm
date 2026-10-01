# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/jgcb00/mamba/blob/9a3daf5c488d9bf01d71441988de0293fd60ef0b/mamba_ssm/ops/triton/mamba3/utils.py
# (Mamba-3, Dao AI Lab / Goombalab, Apache-2.0); forward/inference parts only.

from vllm.triton_utils import tl, triton


@triton.jit
def tanh_approx(x):
    """
    (Fast) hyperbolic tangent approximation using PTX inline assembly.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate tanh values in float32
    """
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
