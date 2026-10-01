# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loud, once-per-cause reporting when Olala leaves a fast path.

The fallbacks stay correct but cost real throughput, so they must never pass
unnoticed in a server log.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

_reported: set[str] = set()


def warn_slow_path(path: str, reason: str, impact: str, fix: str) -> None:
    if path in _reported:
        return
    _reported.add(path)
    bar = "!" * 78
    logger.warning(
        "\n%s\n!! OLALA SLOW PATH: %s\n!!   reason : %s\n!!   impact : %s\n"
        "!!   fix    : %s\n%s",
        bar, path, reason, impact, fix, bar,
    )
