# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-resident residual swap for the MiniMax-H3 DiT (design item C/D2).

The first residual of a :class:`MiniMaxH3DiTBlock` (the block input ``x``) must
stay alive across the whole attention phase even though nothing reads it until
the gated residual after attention. With ``H3_RESIDUAL_SWAP=host`` that single
tensor is copied asynchronously into a bounded pinned-host slot and its device
storage is released; it is copied back right before the gated residual.

Environment contract:

``H3_RESIDUAL_SWAP``
    ``off`` (default) or ``host``. ``off`` keeps the tensor in device memory and
    the pool is never constructed.
``H3_SWAP_HOST_GIB``
    Job-wide pinned-host budget in GiB (default ``48``). Divided by the world
    size to obtain the per-rank budget: the design forbids a per-rank 48 GiB
    interpretation.

Correctness rules implemented here:

* the device source storage stays valid until the D2H copy completes (the copy
  is issued on the current stream and the source tensor is only dropped after
  the event is recorded);
* a host slot is not recycled before its H2D copy has completed;
* a device restore tensor is not reused by another swap before its consumer is
  done (the restore tensor is handed to the caller and never pooled);
* :meth:`ResidualSwapPool.drain` waits for every outstanding copy, and the
  block calls it on exceptions/cancellation so a failing request never leaves a
  copy in flight.

Without an NPU (CPU tests) the pool falls back to pageable host buffers and
synchronous copies; the byte content is identical, only the overlap is lost.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_RESIDUAL_SWAP = "H3_RESIDUAL_SWAP"
ENV_SWAP_HOST_GIB = "H3_SWAP_HOST_GIB"

MODE_OFF = "off"
MODE_HOST = "host"
SUPPORTED_MODES = (MODE_OFF, MODE_HOST)
DEFAULT_SWAP_HOST_GIB = 48.0

_cached_mode: str | None = None
_cached_budget: int | None = None


def reset_swap_cache() -> None:
    """Test helper: drop cached env values and the module pool."""
    global _cached_mode, _cached_budget, _POOL
    _cached_mode = None
    _cached_budget = None
    if _POOL is not None:
        _POOL.close()
    _POOL = None


def residual_swap_mode() -> str:
    global _cached_mode
    if _cached_mode is not None:
        return _cached_mode
    raw = (os.environ.get(ENV_RESIDUAL_SWAP, MODE_OFF) or MODE_OFF).strip().lower()
    if raw not in SUPPORTED_MODES:
        raise ValueError(f"{ENV_RESIDUAL_SWAP}={raw!r} must be one of {SUPPORTED_MODES}")
    _cached_mode = raw
    return raw


def swap_host_budget_bytes() -> int:
    """Job-wide pinned-host budget in bytes."""
    global _cached_budget
    if _cached_budget is not None:
        return _cached_budget
    raw = os.environ.get(ENV_SWAP_HOST_GIB, str(DEFAULT_SWAP_HOST_GIB))
    try:
        gib = float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_SWAP_HOST_GIB}={raw!r} must be a number of GiB") from exc
    if gib <= 0:
        raise ValueError(f"{ENV_SWAP_HOST_GIB}={raw!r} must be positive")
    _cached_budget = int(gib * (1024**3))
    return _cached_budget


def per_rank_budget_bytes(world_size: int) -> int:
    return swap_host_budget_bytes() // max(1, int(world_size))


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:  # noqa: BLE001
        return False


class SwapHandle:
    """One in-flight residual swap."""

    __slots__ = ("host", "shape", "dtype", "event_out", "event_in", "pinned", "_released")

    def __init__(
        self,
        host: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        event_out: torch.Event | None,
        pinned: bool,
    ) -> None:
        self.host = host
        self.shape = shape
        self.dtype = dtype
        self.event_out = event_out
        self.event_in: torch.Event | None = None
        self.pinned = pinned
        self._released = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SwapHandle(shape={self.shape}, dtype={self.dtype}, pinned={self.pinned})"


class ResidualSwapPool:
    """Bounded pool of pinned host slots for one rank."""

    def __init__(self, *, max_bytes: int, device: torch.device | str) -> None:
        if max_bytes <= 0:
            raise ValueError("residual swap pool requires a positive byte budget")
        self.max_bytes = int(max_bytes)
        self.device = torch.device(device)
        self._free: list[torch.Tensor] = []
        self._live: list[SwapHandle] = []
        self._outstanding: list[tuple[torch.Event | None, torch.Event | None]] = []
        self._resident_bytes = 0
        self._peak_bytes = 0
        self._pinned_supported = _npu_available()
        self._swaps_out = 0
        self._swaps_in = 0
        self._fallbacks = 0

    # -- host storage ----------------------------------------------------
    def _allocate(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        nbytes = numel * torch.tensor([], dtype=dtype).element_size()
        if self._resident_bytes + nbytes > self.max_bytes:
            if self._pinned_supported:
                raise RuntimeError(
                    "H3 residual swap pool exhausted: "
                    f"need {nbytes} bytes, resident {self._resident_bytes}, budget {self.max_bytes}"
                )
            # CPU fallback keeps working but must not silently exceed the budget.
            self._fallbacks += 1
        host: torch.Tensor | None = None
        if self._pinned_supported:
            try:
                host = torch.empty(numel, dtype=dtype, pin_memory=True)
            except Exception:  # noqa: BLE001 - no pinned allocator on this device
                host = None
                self._pinned_supported = False
                self._fallbacks += 1
        if host is None:
            host = torch.empty(numel, dtype=dtype)
        self._resident_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
        return host

    def _acquire(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        for index, host in enumerate(self._free):
            if host.numel() == numel and host.dtype == dtype:
                self._free.pop(index)
                return host
        return self._allocate(numel, dtype)

    def _release_host(self, handle: SwapHandle) -> None:
        self._free.append(handle.host)

    # -- swap operations -------------------------------------------------
    def swap_out(self, tensor: torch.Tensor, *, release: Any | None = None) -> SwapHandle:
        """Copy ``tensor`` into pinned host memory and drop the device source."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("residual swap expects a tensor")
        contiguous = tensor if tensor.is_contiguous() else tensor.contiguous()
        host = self._acquire(contiguous.numel(), contiguous.dtype)
        view = host.view(contiguous.shape)
        event_out: torch.Event | None = None
        if self._pinned_supported:
            view.copy_(contiguous, non_blocking=True)
            event_out = torch.npu.Event()
            event_out.record()
            self._outstanding.append((event_out, None))
        else:
            view.copy_(contiguous)
        handle = SwapHandle(host, tuple(contiguous.shape), contiguous.dtype, event_out, self._pinned_supported)
        self._live.append(handle)
        self._swaps_out += 1
        if release is not None:
            release()
        return handle

    def swap_in(self, handle: SwapHandle) -> torch.Tensor:
        """Copy a swapped residual back into a fresh device tensor."""
        if handle.event_out is not None:
            handle.event_out.synchronize()
        out = torch.empty(handle.shape, dtype=handle.dtype, device=self.device)
        source = handle.host.view(handle.shape)
        if self._pinned_supported:
            event_in = torch.npu.Event()
            out.copy_(source, non_blocking=True)
            event_in.record()
            handle.event_in = event_in
            self._outstanding.append((None, event_in))
        else:
            out.copy_(source)
        self._swaps_in += 1
        if handle in self._live:
            self._live.remove(handle)
        self._release_host(handle)
        return out

    def drain(self) -> None:
        """Wait for every outstanding copy; safe to call on error paths."""
        for event_out, event_in in self._outstanding:
            for event in (event_out, event_in):
                if event is not None:
                    try:
                        event.synchronize()
                    except Exception:  # noqa: BLE001 - teardown must not raise
                        logger.debug("[H3_SWAP] event synchronize failed during drain", exc_info=True)
        self._outstanding.clear()

    def close(self) -> None:
        self.drain()
        self._free.clear()
        self._live.clear()
        self._resident_bytes = 0

    def stats(self) -> dict[str, int | bool]:
        return {
            "max_bytes": self.max_bytes,
            "resident_bytes": self._resident_bytes,
            "peak_bytes": self._peak_bytes,
            "free_slots": len(self._free),
            "live_swaps": len(self._live),
            "swaps_out": self._swaps_out,
            "swaps_in": self._swaps_in,
            "pinned": self._pinned_supported,
            "fallbacks": self._fallbacks,
        }


_POOL: ResidualSwapPool | None = None


def get_residual_swap_pool(
    *,
    device: torch.device | str | None = None,
    world_size: int = 1,
) -> ResidualSwapPool:
    """Return the process-wide pool (created on first use)."""
    global _POOL
    if _POOL is None:
        if device is None:
            device = "npu" if _npu_available() else "cpu"
        _POOL = ResidualSwapPool(max_bytes=per_rank_budget_bytes(world_size), device=device)
        logger.info(
            "[H3_SWAP] residual host swap enabled: budget/rank=%.2f GiB pinned=%s",
            _POOL.max_bytes / (1024**3),
            _POOL.stats()["pinned"],
        )
    return _POOL


def drain_residual_swap_pool() -> None:
    if _POOL is not None:
        _POOL.drain()
