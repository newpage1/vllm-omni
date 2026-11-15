# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token-row chunked FFN execution for the MiniMax-H3 DiT (design item B1/D1).

The fused ``fc1``/``act``/``fc2`` triple can run over token-row slices so the
gate/up intermediate and the ``silu(gate) * up`` product stay chunk-sized
instead of full-sequence sized. Chunking is per token row, which keeps every
element-wise and GEMM contract unchanged: only the number of GEMM calls and
the size of the intermediates change.

Environment contract:

``H3_FFN_CHUNKS``
    ``1`` (default), ``2``, ``4`` or ``6``. ``1`` takes the original code path
    unchanged. Any other value is rejected at use time.

The helper never builds a Python list of chunk results and never concatenates:
the final output is allocated once and each chunk is copied into its row slice.
"""

from __future__ import annotations

import os
from typing import Any

import torch

ENV_FFN_CHUNKS = "H3_FFN_CHUNKS"
SUPPORTED_CHUNKS = (1, 2, 4, 6)

_cached_chunks: int | None = None


def reset_chunk_cache() -> None:
    """Test helper: drop the cached value so the environment is re-read."""
    global _cached_chunks
    _cached_chunks = None


def ffn_chunk_count() -> int:
    """Return the configured chunk count (cached; validated)."""
    global _cached_chunks
    if _cached_chunks is not None:
        return _cached_chunks
    raw = os.environ.get(ENV_FFN_CHUNKS, "1").strip() or "1"
    try:
        chunks = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_FFN_CHUNKS}={raw!r} must be an integer") from exc
    if chunks not in SUPPORTED_CHUNKS:
        raise ValueError(f"{ENV_FFN_CHUNKS}={raw!r} must be one of {SUPPORTED_CHUNKS}")
    _cached_chunks = chunks
    return chunks


def _linear(layer: Any, x: torch.Tensor) -> torch.Tensor:
    """Call a vLLM linear (returns ``(output, bias)``) or a plain module."""
    result = layer(x)
    if isinstance(result, tuple):
        return result[0]
    return result


def chunked_ffn_forward(
    x: torch.Tensor,
    fc1: Any,
    act_fn: Any,
    fc2: Any,
    *,
    out_features: int,
    chunks: int | None = None,
) -> torch.Tensor:
    """Run ``fc2(act(fc1(x)))`` over ``chunks`` token-row slices.

    ``chunks <= 1`` is the original code path (no allocation, no slicing).
    """
    total = int(x.shape[0])
    if chunks is None:
        chunks = ffn_chunk_count()
    if chunks <= 1 or total <= 0:
        hidden = _linear(fc1, x)
        hidden = act_fn(hidden)
        return _linear(fc2, hidden)
    # A single row (or fewer rows than chunks) cannot be split usefully.
    if total < chunks:
        hidden = _linear(fc1, x)
        hidden = act_fn(hidden)
        return _linear(fc2, hidden)

    out = x.new_empty((total, int(out_features)))
    step = -(-total // chunks)  # ceil division: the tail chunk may be shorter
    for start in range(0, total, step):
        end = min(start + step, total)
        hidden = _linear(fc1, x[start:end])
        hidden = act_fn(hidden)
        part = _linear(fc2, hidden)
        out[start:end].copy_(part)
        del hidden, part
    return out
