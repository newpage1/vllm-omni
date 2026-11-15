# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in phase memory probe for the MiniMax-H3 DiT (design item D0).

The probe is disabled by default and, when disabled, must not touch the
execution path: call sites use :func:`phase`, which returns a shared no-op
context manager in that case (no allocator queries, no synchronisation).

Environment contract (all read once per process, values are immutable after
the first read):

``H3_MEM_PHASES``
    ``0``/``1`` (default ``0``). When enabled, every instrumented phase emits
    a ``BEGIN`` record, synchronises the NPU, and emits an ``END`` record.
``H3_MEM_RANKS``
    Comma separated global ranks to instrument (default ``0,7``). Unselected
    ranks keep a small in-process ring buffer and only dump it if the probe is
    asked to flush on an error.
``H3_MEM_BLOCKS``
    Comma separated block indices to instrument (default ``0``). ``*`` or an
    empty value selects every block.
``H3_MEM_JSONL_DIR``
    Output directory (default ``/tmp/h3_mem_phases``). One
    ``h3_mem_phases_rank<N>.jsonl`` file per selected rank. The historical
    name ``H3_MEM_DIR`` is still accepted as a deprecated alias (it is used
    only when ``H3_MEM_JSONL_DIR`` is unset).
``H3_MEM_RING``
    Number of most recent records kept in memory for error dumps
    (default ``256``).

Record fields follow the t79 design::

    rank, step, block, stage, edge, allocated_bytes, reserved_bytes,
    peak_allocated_bytes, peak_reserved_bytes, peak_scope,
    request_base_allocated_bytes, peak_delta_bytes, peak_delta_scope,
    parent, shape, dtype, storage_ptr, storage_bytes, ts, pid

Peak semantics (t86/F2). ``peak_allocated_bytes`` / ``peak_reserved_bytes`` are
the raw allocator peaks from ``torch.npu.max_memory_allocated()`` /
``max_memory_reserved()``. Those counters are *process lifetime* values: the
probe deliberately does not call ``torch.npu.reset_peak_memory_stats()`` (it
would perturb every other consumer of the allocator and add a per-request
reset), so ``peak_scope`` is ``process_lifetime``. The request-relative view is
``peak_delta_bytes = max(0, peak_allocated_bytes - request_base_allocated_bytes)``
with ``peak_delta_scope = "since_request_start"``; the baseline is the
``allocated_bytes`` captured by :func:`reset_peaks` at request start.

``parent`` names the enclosing phase when phases nest (``attn.local_kernel``
and the other ``attn.*`` records live inside ``block.attn``; ``ffn.*`` lives
inside ``block.mlp``). Nested stages must never be summed: the child's bytes
are already part of the parent's window.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import nullcontext
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_PHASES = "H3_MEM_PHASES"
ENV_RANKS = "H3_MEM_RANKS"
ENV_BLOCKS = "H3_MEM_BLOCKS"
ENV_DIR = "H3_MEM_JSONL_DIR"
# Historical name used by the t79/t81 notes before the field was renamed to
# match the constant; kept as a deprecated alias so old launch scripts work.
ENV_DIR_LEGACY = "H3_MEM_DIR"
ENV_RING = "H3_MEM_RING"

DEFAULT_DIR = "/tmp/h3_mem_phases"
DEFAULT_RANKS = "0,7"
DEFAULT_BLOCKS = "0"
DEFAULT_RING = 256

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}

_STATE_LOCK = threading.Lock()
_INITIALISED = False
_ENABLED = False
_RANKS: frozenset[int] = frozenset()
_BLOCKS: frozenset[int] | None = frozenset({0})
_OUTPUT_DIR = DEFAULT_DIR
_RING_SIZE = DEFAULT_RING
_STEP = -1
_BLOCK = -1
_RANK: int | None = None
_PEAK_ALLOCATED = 0
_PEAK_RESERVED = 0
# Allocated bytes captured when the request started; the request-relative peak
# delta is measured against this baseline (the raw allocator peaks themselves
# are process-lifetime values, see the module docstring).
_REQUEST_BASE_ALLOCATED = 0
_REQUEST_BASE_RESERVED = 0
_LEGACY_DIR_WARNED = False

# Enclosing phase for nested instrumentation points. A child's window is fully
# contained in its parent's, so per-stage bytes must never be summed.
STAGE_PARENTS: dict[str, str] = {
    "attn.pre_attention": "block.attn",
    "attn.ring_kernel": "block.attn",
    "attn.local_kernel": "block.attn",
    "attn.post_attention": "block.attn",
    "ffn.fc1": "block.mlp",
    "ffn.act": "block.mlp",
    "ffn.fc2": "block.mlp",
    "ffn.chunked": "block.mlp",
}


def stage_parent(stage: str) -> str | None:
    """Return the enclosing phase name for a nested stage (or ``None``)."""
    return STAGE_PARENTS.get(stage)

# The NPU runtime is absent (or unusable) on CPU-only hosts; probing must never
# fail a request there, so a failing allocator/synchronise call disables the
# corresponding query for the rest of the process.
_NPU_MEMORY_OK = True
_NPU_SYNC_OK = True

_ring: list[dict[str, Any]] = []
_begin_count = 0
_end_count = 0
_last_begin: dict[str, Any] | None = None
_last_end: dict[str, Any] | None = None
_file_handle = None
_file_rank: int | None = None
_dumped_on_error = False


def _parse_bool(value: str, *, env: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{env}={value!r} is not a boolean; use 0/1")


def _parse_int_set(value: str, *, env: str, wildcard: bool) -> frozenset[int] | None:
    text = value.strip()
    if wildcard and text in {"*", "all"}:
        return None
    if not text:
        return frozenset() if not wildcard else None
    parsed: set[int] = set()
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            index = int(item)
        except ValueError as exc:
            raise ValueError(f"{env}={value!r} must be a comma separated integer list") from exc
        if index < 0:
            raise ValueError(f"{env}={value!r} must not contain negative indices")
        parsed.add(index)
    return frozenset(parsed)


def _ensure_initialised() -> None:
    global _INITIALISED, _ENABLED, _RANKS, _BLOCKS, _OUTPUT_DIR, _RING_SIZE
    global _LEGACY_DIR_WARNED
    if _INITIALISED:
        return
    with _STATE_LOCK:
        if _INITIALISED:
            return
        _ENABLED = _parse_bool(os.environ.get(ENV_PHASES, "0"), env=ENV_PHASES)
        ranks = _parse_int_set(os.environ.get(ENV_RANKS, DEFAULT_RANKS), env=ENV_RANKS, wildcard=True)
        _RANKS = ranks if ranks is not None else frozenset()
        blocks = _parse_int_set(os.environ.get(ENV_BLOCKS, DEFAULT_BLOCKS), env=ENV_BLOCKS, wildcard=True)
        _BLOCKS = blocks
        configured_dir = os.environ.get(ENV_DIR)
        if not configured_dir:
            legacy_dir = os.environ.get(ENV_DIR_LEGACY)
            if legacy_dir:
                if not _LEGACY_DIR_WARNED:
                    logger.warning(
                        "[H3_MEM_PHASES] %s is deprecated; use %s instead (value %r)",
                        ENV_DIR_LEGACY,
                        ENV_DIR,
                        legacy_dir,
                    )
                    _LEGACY_DIR_WARNED = True
                configured_dir = legacy_dir
        _OUTPUT_DIR = configured_dir or DEFAULT_DIR
        try:
            ring = int(os.environ.get(ENV_RING, str(DEFAULT_RING)))
        except ValueError as exc:
            raise ValueError(f"{ENV_RING} must be an integer") from exc
        _RING_SIZE = max(1, ring)
        _INITIALISED = True


def reset_state_for_tests() -> None:
    """Test helper: forget cached environment state and in-memory records."""
    global _INITIALISED, _ENABLED, _RANKS, _BLOCKS, _OUTPUT_DIR, _RING_SIZE
    global _STEP, _BLOCK, _RANK, _PEAK_ALLOCATED, _PEAK_RESERVED
    global _REQUEST_BASE_ALLOCATED, _REQUEST_BASE_RESERVED, _LEGACY_DIR_WARNED
    global _begin_count, _end_count, _last_begin, _last_end, _file_handle, _file_rank, _dumped_on_error
    global _NPU_MEMORY_OK, _NPU_SYNC_OK
    with _STATE_LOCK:
        _NPU_MEMORY_OK = True
        _NPU_SYNC_OK = True
        _INITIALISED = False
        _ENABLED = False
        _RANKS = frozenset()
        _BLOCKS = frozenset({0})
        _OUTPUT_DIR = DEFAULT_DIR
        _RING_SIZE = DEFAULT_RING
        _STEP = -1
        _BLOCK = -1
        _RANK = None
        _PEAK_ALLOCATED = 0
        _PEAK_RESERVED = 0
        _REQUEST_BASE_ALLOCATED = 0
        _REQUEST_BASE_RESERVED = 0
        _LEGACY_DIR_WARNED = False
        _ring.clear()
        _begin_count = 0
        _end_count = 0
        _last_begin = None
        _last_end = None
        _file_handle = None
        _file_rank = None
        _dumped_on_error = False


def probe_enabled() -> bool:
    _ensure_initialised()
    return _ENABLED


def probe_active() -> bool:
    """True when this rank should emit records."""
    _ensure_initialised()
    return _ENABLED and _global_rank() in _RANKS


def block_selected(block: int | None = None) -> bool:
    _ensure_initialised()
    if not _ENABLED:
        return False
    if _BLOCKS is None:
        return True
    index = _BLOCK if block is None else block
    return index in _BLOCKS


def _global_rank() -> int:
    global _RANK
    if _RANK is not None:
        return _RANK
    rank = 0
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_available() and distributed.is_initialized():
        try:
            rank = int(distributed.get_rank())
        except Exception:  # noqa: BLE001 - probing must never fail a request
            rank = 0
    _RANK = rank
    return rank


def set_step(step: int) -> None:
    """Tag subsequent records with the denoise step (no-op when disabled)."""
    global _STEP
    if not probe_enabled():
        return
    _STEP = int(step)


def set_block(block: int) -> None:
    global _BLOCK
    if not probe_enabled():
        return
    _BLOCK = int(block)


def current_step() -> int:
    return _STEP


def reset_peaks() -> None:
    """Capture the request baseline and reset the per-request shadow peaks.

    The allocator peaks are process-lifetime counters and are *not* reset here
    (see the module docstring); this only records the ``allocated_bytes`` /
    ``reserved_bytes`` baseline used for ``peak_delta_bytes``.
    """
    global _PEAK_ALLOCATED, _PEAK_RESERVED, _REQUEST_BASE_ALLOCATED, _REQUEST_BASE_RESERVED
    if not probe_enabled():
        return
    snapshot = memory_snapshot()
    _REQUEST_BASE_ALLOCATED = snapshot["allocated_bytes"]
    _REQUEST_BASE_RESERVED = snapshot["reserved_bytes"]
    _PEAK_ALLOCATED = 0
    _PEAK_RESERVED = 0


def request_baseline() -> dict[str, int]:
    """Request-start allocated/reserved baseline captured by reset_peaks()."""
    return {
        "request_base_allocated_bytes": _REQUEST_BASE_ALLOCATED,
        "request_base_reserved_bytes": _REQUEST_BASE_RESERVED,
    }


def _npu_module():
    return getattr(torch, "npu", None)


def memory_snapshot() -> dict[str, int]:
    global _NPU_MEMORY_OK
    allocated = reserved = peak_allocated = peak_reserved = 0
    npu = _npu_module()
    if _NPU_MEMORY_OK and npu is not None and hasattr(npu, "memory_allocated"):
        try:
            allocated = int(npu.memory_allocated())
            reserved = int(npu.memory_reserved())
            peak_allocated = int(npu.max_memory_allocated())
            peak_reserved = int(npu.max_memory_reserved())
        except Exception:  # noqa: BLE001 - CPU-only hosts have no NPU allocator
            _NPU_MEMORY_OK = False
            allocated = reserved = peak_allocated = peak_reserved = 0
            logger.debug("[H3_MEM_PHASES] NPU memory queries disabled after a failure", exc_info=True)
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def synchronize() -> None:
    global _NPU_SYNC_OK
    if not _NPU_SYNC_OK:
        return
    npu = _npu_module()
    if npu is not None and hasattr(npu, "synchronize"):
        try:
            npu.synchronize()
        except Exception:  # noqa: BLE001 - never fail a request from diagnostics
            _NPU_SYNC_OK = False
            logger.debug("[H3_MEM_PHASES] NPU synchronise disabled after a failure", exc_info=True)


def _tensor_fields(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return {"shape": None, "dtype": None, "storage_ptr": None, "storage_bytes": None}
    try:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_bytes = int(storage.nbytes())
    except Exception:  # noqa: BLE001
        storage_ptr = storage_bytes = None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "storage_ptr": storage_ptr,
        "storage_bytes": storage_bytes,
    }


def _writer_for_rank(rank: int):
    global _file_handle, _file_rank
    if _file_handle is not None and _file_rank == rank:
        return _file_handle
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(_OUTPUT_DIR, f"h3_mem_phases_rank{rank}.jsonl")
    _file_handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 - long lived by design
    _file_rank = rank
    return _file_handle


def _emit(record: dict[str, Any], *, to_disk: bool) -> None:
    global _begin_count, _end_count, _last_begin, _last_end
    _ring.append(record)
    if len(_ring) > _RING_SIZE:
        del _ring[: len(_ring) - _RING_SIZE]
    if record["edge"] == "BEGIN":
        _begin_count += 1
        _last_begin = record
    else:
        _end_count += 1
        _last_end = record
    if not to_disk:
        return
    handle = _writer_for_rank(record["rank"])
    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=False))
    handle.write("\n")
    handle.flush()


def _record(stage: str, edge: str, *, tensor: torch.Tensor | None, block: int | None, extra: dict | None) -> dict:
    snapshot = memory_snapshot()
    global _PEAK_ALLOCATED, _PEAK_RESERVED
    if snapshot["peak_allocated_bytes"] > _PEAK_ALLOCATED:
        _PEAK_ALLOCATED = snapshot["peak_allocated_bytes"]
    if snapshot["peak_reserved_bytes"] > _PEAK_RESERVED:
        _PEAK_RESERVED = snapshot["peak_reserved_bytes"]
    peak_delta = max(0, snapshot["peak_allocated_bytes"] - _REQUEST_BASE_ALLOCATED)
    record: dict[str, Any] = {
        "rank": _global_rank(),
        "step": _STEP,
        "block": _BLOCK if block is None else int(block),
        "stage": stage,
        "edge": edge,
        **snapshot,
        # The raw allocator peaks are process-lifetime counters; the request
        # view is peak_delta_bytes (see the module docstring).
        "peak_scope": "process_lifetime",
        "request_base_allocated_bytes": _REQUEST_BASE_ALLOCATED,
        "request_base_reserved_bytes": _REQUEST_BASE_RESERVED,
        "peak_delta_bytes": peak_delta,
        "peak_delta_scope": "since_request_start",
        "parent": stage_parent(stage),
        **_tensor_fields(tensor),
        "ts": time.time(),
        "pid": os.getpid(),
    }
    if extra:
        record.update(extra)
    return record


class _Phase:
    """Context manager that emits BEGIN/END around one instrumented region."""

    __slots__ = ("stage", "tensor", "block", "extra")

    def __init__(self, stage: str, tensor: torch.Tensor | None, block: int | None, extra: dict | None) -> None:
        self.stage = stage
        self.tensor = tensor
        self.block = block
        self.extra = extra

    def __enter__(self):
        _emit(_record(self.stage, "BEGIN", tensor=self.tensor, block=self.block, extra=self.extra), to_disk=True)
        synchronize()
        return self

    def __exit__(self, exc_type, exc, tb):
        synchronize()
        extra = self.extra
        if exc_type is not None:
            extra = dict(extra or {})
            extra["error"] = f"{exc_type.__name__}: {exc}"
        _emit(_record(self.stage, "END", tensor=self.tensor, block=self.block, extra=extra), to_disk=True)
        if exc_type is not None:
            dump_on_error(exc)
        return False


_NULL_PHASE = nullcontext()


def phase(
    stage: str,
    *,
    tensor: torch.Tensor | None = None,
    block: int | None = None,
    extra: dict | None = None,
):
    """Return a probe context manager, or a no-op when the probe is off."""
    if not probe_enabled() or not probe_active():
        return _NULL_PHASE
    if not block_selected(block):
        return _NULL_PHASE
    return _Phase(stage, tensor, block, extra)


def note(stage: str, *, block: int | None = None, extra: dict | None = None) -> None:
    """Emit a single END-style record without a matched BEGIN."""
    if not probe_enabled() or not probe_active() or not block_selected(block):
        return
    _emit(_record(stage, "END", tensor=None, block=block, extra=extra), to_disk=True)


def dump_on_error(exc: BaseException | None) -> None:
    """Persist the in-memory ring of the last records for post-mortem."""
    global _dumped_on_error
    if not probe_enabled() or _dumped_on_error:
        return
    _dumped_on_error = True
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(_OUTPUT_DIR, f"h3_mem_oom_rank{_global_rank()}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "rank": _global_rank(),
                        "kind": "probe_error",
                        "error": None if exc is None else f"{type(exc).__name__}: {exc}",
                        "last_begin": _last_begin,
                        "last_end": _last_end,
                        "begin_count": _begin_count,
                        "end_count": _end_count,
                    },
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
            for item in _ring:
                handle.write(json.dumps(item, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
        logger.warning(
            "[H3_MEM_PHASES] flushed %d records to %s after %s",
            len(_ring),
            path,
            None if exc is None else type(exc).__name__,
        )
    except Exception:  # noqa: BLE001 - diagnostics must never mask the original error
        logger.debug("[H3_MEM_PHASES] failed to dump the probe ring", exc_info=True)


def counters() -> dict[str, int]:
    """Diagnostics for tests: record counters and ring size."""
    return {
        "begin": _begin_count,
        "end": _end_count,
        "ring": len(_ring),
        "last_begin_stage_len": 0 if _last_begin is None else len(_last_begin.get("stage", "")),
    }


def recent_records() -> list[dict[str, Any]]:
    """Diagnostics for tests: a copy of the in-memory record ring."""
    return [dict(record) for record in _ring]
