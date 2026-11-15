# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact Q-row chunking for the packed local attention kernel (MiniMax-H3 opt-in).

Why this exists
---------------
The 4K/30 s MiniMax-H3 decode is limited by the attention phase, where the
local kernel receives the *whole* post-all-to-all sequence at once: Q is
``[1, T, N, D]`` with ``T`` up to ~65k tokens, and the kernel materialises its
own per-launch workspace on top of Q/K/V and the logits. Splitting Q into
``chunks`` contiguous row ranges and running the same kernel once per range
keeps the KV side untouched (every chunk still attends to the full valid KV
prefix) and therefore computes the *same* attention: softmax is row-local, so
row blocks are independent given full KV.

Contract (and why it is safe to be strict)
------------------------------------------
Chunking is only exact when each chunk can be described to the backend as a
single-document packed attention over the same KV prefix. That is exactly the
contract ``FlashAttentionImpl._resolve_packed_seq_npu`` accepts:
``cu_seqlens_q == [0, rows]``, ``max_seqlen_q == rows``, ``cu_seqlens_k ==
[0, used_k]``, ``max_seqlen_k == used_k``, single packed batch, mask-free.
Everything else - a dense mask (the ``full_qk`` path), a piecewise span plan,
joint (cross) Q/K/V, paged KV, a ring strategy, quantized KV - either changes
the row-local semantics or needs the mask to be sliced along with Q, so this
module raises :class:`NotImplementedError` instead of silently falling back.
A silent fallback would be indistinguishable from a working chunked run in a
memory profile while producing different numbers.

The switch is ``H3_ATTN_Q_CHUNKS`` (default ``1`` = completely unchanged
code path; the caller only calls into this module when the count is > 1).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Callable

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)

logger = init_logger(__name__)

ENV_Q_CHUNKS = "H3_ATTN_Q_CHUNKS"
#: Accepted chunk counts. 1 is the untouched path; the others are divisors that
#: keep the per-chunk row count close to (but never below) one row.
ALLOWED_Q_CHUNKS: tuple[int, ...] = (1, 2, 4, 6)

_PACKED_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
_FLOAT_KV_CACHE_DTYPES = frozenset({None, "float", "auto"})

_cached_raw: str | None = None
_cached_chunks: int = 1


def reset_q_chunk_cache() -> None:
    """Drop the parsed-env cache (tests, and env changes between requests)."""
    global _cached_raw, _cached_chunks
    _cached_raw = None
    _cached_chunks = 1


def q_chunk_count() -> int:
    """Return the configured Q chunk count (``1`` when the env var is unset)."""
    global _cached_raw, _cached_chunks
    raw = os.environ.get(ENV_Q_CHUNKS)
    if raw is None or raw == "":
        _cached_raw, _cached_chunks = None, 1
        return 1
    raw = raw.strip()
    if raw == _cached_raw:
        return _cached_chunks
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - message asserted in tests
        raise ValueError(
            f"{ENV_Q_CHUNKS} must be an integer in {ALLOWED_Q_CHUNKS}, got {raw!r}"
        ) from exc
    if value not in ALLOWED_Q_CHUNKS:
        raise ValueError(
            f"{ENV_Q_CHUNKS} must be one of {ALLOWED_Q_CHUNKS}, got {value}"
        )
    _cached_raw, _cached_chunks = raw, value
    return value


def chunk_bounds(rows: int, chunks: int) -> tuple[tuple[int, int], ...]:
    """Split ``rows`` into at most ``chunks`` contiguous ``[start, end)`` ranges."""
    if rows <= 0:
        return ()
    if chunks <= 1:
        return ((0, rows),)
    step = -(-rows // chunks)  # ceil division: never emit a zero-row chunk
    return tuple(
        (start, min(start + step, rows)) for start in range(0, rows, step)
    )


@dataclass(frozen=True)
class QChunkPlan:
    """Everything the driver needs; built once per layer forward."""

    chunks: int
    total_q: int
    used_q: int
    used_k: int
    chunk_rows: int
    bounds: tuple[tuple[int, int], ...]

    def describe(self) -> dict[str, object]:
        return {
            "chunks": self.chunks,
            "total_q": self.total_q,
            "used_q": self.used_q,
            "used_k": self.used_k,
            "chunk_rows": self.chunk_rows,
            "rows_per_chunk": [end - start for start, end in self.bounds],
        }


def _unsupported(message: str) -> NotImplementedError:
    return NotImplementedError(
        f"{ENV_Q_CHUNKS} Q-row chunking is not applicable here: {message}. "
        f"Set {ENV_Q_CHUNKS}=1 (default) to run the untouched kernel path."
    )


def plan_q_chunking(
    *,
    chunks: int,
    attn_metadata: AttentionMetadata | None,
    query: torch.Tensor,
    key: torch.Tensor,
    backend_name: str | None,
    backend_supports_mask_free: bool,
) -> QChunkPlan:
    """Validate the exactness contract and derive the chunk geometry.

    ``chunks == 1`` is accepted so tests and the equivalence harness can drive
    the identity case through the same code path; the layer itself only calls
    here when ``q_chunk_count() > 1``.

    Raises:
        NotImplementedError: when chunking would not be bit-exact.
    """
    if chunks < 1:
        raise ValueError(f"chunks must be >= 1, got {chunks}")
    if attn_metadata is None:
        raise _unsupported("the layer received no attention metadata")
    if query.ndim != 4 or key.ndim != 4:
        raise _unsupported(f"expected BSND Q/K, got query{tuple(query.shape)} key{tuple(key.shape)}")
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise _unsupported("packed chunking requires a single packed batch (B == 1)")
    if backend_name != "FLASH_ATTN":
        raise _unsupported(f"the backend is {backend_name!r}, not the masking-free NPU FLASH_ATTN path")
    if not backend_supports_mask_free:
        raise _unsupported("the backend does not advertise supports_packed_mask_free()")
    extra = attn_metadata.extra or {}
    if not extra.get("npu_attn_varlen"):
        raise _unsupported("metadata does not opt into the NPU packed varlen path (npu_attn_varlen)")
    missing = [key_name for key_name in _PACKED_KEYS if key_name not in extra]
    if missing:
        raise _unsupported(f"metadata is missing packed keys {missing}")
    if attn_metadata.attn_mask is not None:
        raise _unsupported(
            "the forward carries an attn_mask; that is the full_qk/masked path, whose mask "
            "would have to be sliced together with Q and whose padding rows are not zero"
        )
    if attn_metadata.full_attn_spans is not None:
        raise _unsupported("piecewise full_attn_spans are set")
    if attn_metadata.joint_query is not None or attn_metadata.joint_key is not None:
        raise _unsupported("joint (cross) Q/K/V are set")
    kv_cache_dtype = extra.get("kv_cache_dtype")
    if kv_cache_dtype not in _FLOAT_KV_CACHE_DTYPES:
        raise _unsupported(f"kv_cache_dtype={kv_cache_dtype!r} is quantized")
    cu_q = extra["cu_seqlens_q"]
    cu_k = extra["cu_seqlens_k"]
    used_q = extra["max_seqlen_q"]
    used_k = extra["max_seqlen_k"]
    if not isinstance(used_q, int) or not isinstance(used_k, int):
        raise _unsupported("max_seqlen_q/max_seqlen_k are not host ints")
    if not isinstance(cu_q, torch.Tensor) or not isinstance(cu_k, torch.Tensor):
        raise _unsupported("cu_seqlens_q/cu_seqlens_k are not tensors")
    # The producer passes either the canonical two-element [0, length] view or
    # the [0, real, total] cumulative form (MiniMax-H3 uses the latter for the
    # parent metadata, the former for each chunk).
    if (
        cu_q.ndim != 1
        or cu_k.ndim != 1
        or cu_q.numel() not in (2, 3)
        or cu_k.numel() not in (2, 3)
    ):
        raise _unsupported(
            "cu_seqlens must be canonical [0, length] or [0, real, total] views, got "
            f"q={cu_q.numel()} k={cu_k.numel()} elements"
        )
    total_q, total_k = int(query.shape[1]), int(key.shape[1])
    if not (0 < used_q <= total_q) or not (0 < used_k <= total_k):
        raise _unsupported(
            f"valid lengths out of range (used_q={used_q}, total_q={total_q}, "
            f"used_k={used_k}, total_k={total_k})"
        )
    bounds = chunk_bounds(used_q, chunks)
    if not bounds:
        raise _unsupported(f"used_q={used_q} leaves nothing to attend to")
    return QChunkPlan(
        chunks=chunks,
        total_q=total_q,
        used_q=used_q,
        used_k=used_k,
        chunk_rows=max(end - start for start, end in bounds),
        bounds=bounds,
    )


def build_chunk_metadata(
    attn_metadata: AttentionMetadata,
    plan: QChunkPlan,
    *,
    start: int,
    rows: int,
    device: torch.device,
) -> AttentionMetadata:
    """Describe one Q row range as a single-document packed attention."""
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, plan.used_k], dtype=torch.int32, device=device)
    extra = dict(attn_metadata.extra or {})
    extra.update(
        {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": int(rows),
            "max_seqlen_k": int(plan.used_k),
            "valid_kv_length": int(plan.used_k),
            "npu_attn_varlen": True,
            "q_chunk_index": int(start // max(1, plan.chunk_rows)),
            "q_chunk_count": int(plan.chunks),
            "q_chunk_start": int(start),
        }
    )
    packed = attn_metadata.packed_padding
    chunk_packed = (
        PackedPaddingMetadata(
            q_length=int(rows),
            kv_length=int(plan.used_k),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )
        if packed is not None
        else None
    )
    return dataclasses.replace(
        attn_metadata,
        attn_mask=None,
        full_attn_spans=None,
        packed_padding=chunk_packed,
        extra=extra,
    )


def run_chunked_attention(
    *,
    plan: QChunkPlan,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
    kernel: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Run ``kernel`` once per Q row range and assemble the full-length output.

    Rows past ``used_q`` (the structural padding document) are left as zeros -
    exactly what the masked reference produces, where padding rows attend to
    nothing and softmax returns zeros rather than NaNs.
    """
    # One allocation for the final output, reused across chunks: chunking must
    # not add a second full-length buffer.
    out = query.new_zeros((query.shape[0], plan.total_q, query.shape[2], query.shape[3]))
    key_view = key[:, : plan.used_k]
    value_view = value[:, : plan.used_k]
    for index, (start, end) in enumerate(plan.bounds):
        rows = end - start
        if rows <= 0:  # pragma: no cover - chunk_bounds never emits empty ranges
            continue
        chunk_meta = build_chunk_metadata(
            attn_metadata, plan, start=start, rows=rows, device=query.device
        )
        part = kernel(
            query[:, start:end].contiguous(),
            key_view,
            value_view,
            chunk_meta,
        )
        if part.shape[1] != rows:  # pragma: no cover - defensive
            raise RuntimeError(
                f"chunked attention returned {part.shape[1]} rows for the range "
                f"[{start}, {end})"
            )
        out[:, start:end] = part.to(out.dtype)
        del part
        if index == 0 and logger.isEnabledFor(10):
            logger.debug(
                "%s: first Q chunk rows=%d used_k=%d plan=%s",
                ENV_Q_CHUNKS,
                rows,
                plan.used_k,
                plan.describe(),
            )
    return out
