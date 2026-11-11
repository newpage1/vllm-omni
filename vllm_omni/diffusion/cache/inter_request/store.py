# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-request cache storage for MiniMax-H3 packed denoise state.

Each DiT rank owns one store (``rank-NN/`` on disk), because the packed H3 rows
are already replicated on every rank.  Entries hold:

* ``checkpoints``: packed ``(video_rows, audio_rows)`` snapshots taken after a
  denoise step, stored as CPU FP32;
* ``video_latent`` / ``audio_latent``: the pre-VAE latents of the finished
  request, so a byte-identical request can skip the whole DiT loop.

The store is a small LRU with a per-rank memory budget.  When a persistent
directory is configured the entries survive a server restart, and the manifest
(including the CLIP prompt embedding) is loaded eagerly so semantic lookup does
not have to read every checkpoint file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

from .config import InterRequestCacheConfig

logger = init_logger(__name__)

SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"


def _cpu_fp32(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)


def _cpu_like(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy to CPU keeping the original dtype (final latents)."""
    if tensor is None:
        return None
    return tensor.detach().to(device="cpu", copy=True)


def _nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass
class H3StepCheckpoint:
    """Packed H3 rows after a denoise step (``next_step`` is the next step)."""

    next_step: int
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    schema_version: int = SCHEMA_VERSION

    def nbytes(self) -> int:
        return _nbytes(self.video_rows) + _nbytes(self.audio_rows)


@dataclass
class H3CacheEntry:
    key: str
    prompt: str
    meta: dict[str, Any]
    embedding: torch.Tensor | None = None
    checkpoints: list[H3StepCheckpoint] = field(default_factory=list)
    video_latent: torch.Tensor | None = None
    audio_latent: torch.Tensor | None = None
    created_at: float = 0.0
    restored: bool = False

    def has_final_latents(self) -> bool:
        return self.video_latent is not None and self.audio_latent is not None

    def nbytes(self) -> int:
        return (
            sum(checkpoint.nbytes() for checkpoint in self.checkpoints)
            + _nbytes(self.video_latent)
            + _nbytes(self.audio_latent)
        )

    def checkpoint_at(self, step: int) -> tuple[H3StepCheckpoint | None, int | None]:
        """Return the checkpoint at ``step``, else the closest earlier one."""
        exact: H3StepCheckpoint | None = None
        earlier: H3StepCheckpoint | None = None
        for checkpoint in self.checkpoints:
            if checkpoint.next_step == step:
                exact = checkpoint
                break
            if checkpoint.next_step < step and (
                earlier is None or checkpoint.next_step > earlier.next_step
            ):
                earlier = checkpoint
        chosen = exact if exact is not None else earlier
        return chosen, (None if chosen is None else chosen.next_step)


class H3InterRequestCacheStore:
    """Per-rank LRU store with optional on-disk persistence."""

    def __init__(self, config: InterRequestCacheConfig, *, rank: int = 0, world_size: int = 1) -> None:
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._lock = threading.Lock()
        self._resident: OrderedDict[str, H3CacheEntry] = OrderedDict()
        self._meta: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        self.evictions = 0
        self._dir: str | None = None
        if config.persistent_dir:
            self._dir = os.path.join(os.path.abspath(os.path.expanduser(config.persistent_dir)), f"rank-{self.rank:02d}")
            os.makedirs(self._dir, exist_ok=True)
            self.load_persistent()

    # ------------------------------------------------------------- persistence
    @property
    def dir_path(self) -> str | None:
        return self._dir

    def _manifest_path(self) -> str | None:
        return None if self._dir is None else os.path.join(self._dir, _MANIFEST_NAME)

    def _entry_path(self, key: str) -> str | None:
        return None if self._dir is None else os.path.join(self._dir, f"{key}.pt")

    def load_persistent(self) -> int:
        """Load the entry manifest (metadata only); tensors load lazily."""
        path = self._manifest_path()
        if path is None or not os.path.exists(path):
            return 0
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as exc:  # pragma: no cover - corrupt cache dir
            logger.warning("[inter-request] rank %d: cannot read %s: %s", self.rank, path, exc)
            return 0
        loaded = 0
        for item in payload.get("entries", []):
            key = item.get("key")
            if not isinstance(key, str) or not item.get("meta"):
                continue
            if not os.path.exists(os.path.join(self._dir, f"{key}.pt")):
                continue
            self._meta[key] = item
            loaded += 1
        if loaded:
            logger.info(
                "[inter-request] rank %d: loaded persistent cache manifest %s (%d entries)",
                self.rank,
                path,
                loaded,
            )
        return loaded

    def _write_manifest(self) -> None:
        path = self._manifest_path()
        if path is None:
            return
        payload = {"schema": SCHEMA_VERSION, "entries": list(self._meta.values())}
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)

    def _persist(self, entry: H3CacheEntry) -> None:
        path = self._entry_path(entry.key)
        if path is None:
            return
        payload = {
            "schema": SCHEMA_VERSION,
            "key": entry.key,
            "prompt": entry.prompt,
            "embedding": entry.embedding,
            "meta": entry.meta,
            "created_at": entry.created_at,
            "checkpoints": [
                {
                    "next_step": checkpoint.next_step,
                    "video_rows": checkpoint.video_rows,
                    "audio_rows": checkpoint.audio_rows,
                }
                for checkpoint in entry.checkpoints
            ],
            "video_latent": entry.video_latent,
            "audio_latent": entry.audio_latent,
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self._meta[entry.key] = self._entry_metadata(entry)
        self._write_manifest()

    def _entry_metadata(self, entry: H3CacheEntry) -> dict[str, Any]:
        return {
            "key": entry.key,
            "prompt": entry.prompt,
            "embedding": None if entry.embedding is None else entry.embedding.tolist(),
            "meta": entry.meta,
            "created_at": entry.created_at,
            "nbytes": entry.nbytes(),
            "steps": len(entry.checkpoints),
        }

    def _load_entry(self, key: str) -> H3CacheEntry | None:
        path = self._entry_path(key)
        if path is None or not os.path.exists(path):
            return None
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except (TypeError, RuntimeError):
                payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - corrupt entry file
            logger.warning("[inter-request] rank %d: cannot load %s: %s", self.rank, path, exc)
            return None
        checkpoints = [
            H3StepCheckpoint(
                next_step=int(item["next_step"]),
                video_rows=item["video_rows"],
                audio_rows=item["audio_rows"],
            )
            for item in payload.get("checkpoints", [])
        ]
        return H3CacheEntry(
            key=key,
            prompt=str(payload.get("prompt", "")),
            meta=dict(payload.get("meta") or {}),
            embedding=payload.get("embedding"),
            checkpoints=checkpoints,
            video_latent=payload.get("video_latent"),
            audio_latent=payload.get("audio_latent"),
            created_at=float(payload.get("created_at") or 0.0),
            restored=True,
        )

    # ------------------------------------------------------------------- lookup
    def get(self, key: str) -> H3CacheEntry | None:
        with self._lock:
            entry = self._resident.get(key)
            if entry is not None:
                self._resident.move_to_end(key)
                return entry
            if key not in self._meta:
                return None
        entry = self._load_entry(key)
        if entry is None:
            return None
        with self._lock:
            self._insert_resident(entry)
            self._evict_locked()
        return entry

    def semantic_search(
        self,
        embedding: torch.Tensor,
        shape_key: str,
        threshold: float,
    ) -> tuple[str, float, float]:
        """Best prompt match with the same shape key.

        Returns ``(key, best_similarity, runner_up_similarity)``; ``key`` is an
        empty string when there is no candidate at all.
        """
        query = embedding.to(dtype=torch.float32)
        query = query / query.norm().clamp_min(1e-12)
        best_key = ""
        best = -1.0
        runner_up = -1.0
        with self._lock:
            candidates = list(self._meta.values())
        for item in candidates:
            meta = item.get("meta") or {}
            if meta.get("shape_key") != shape_key:
                continue
            raw = item.get("embedding")
            if not raw:
                continue
            vector = torch.tensor(raw, dtype=torch.float32)
            vector = vector / vector.norm().clamp_min(1e-12)
            similarity = float(torch.dot(query, vector))
            if similarity > best:
                runner_up = best
                best = similarity
                best_key = str(item["key"])
            elif similarity > runner_up:
                runner_up = similarity
        if best < threshold:
            return "", best, runner_up
        return best_key, best, runner_up

    # -------------------------------------------------------------------- write
    def put(self, entry: H3CacheEntry) -> bool:
        if self.config.max_entries <= 0 or self.config.max_memory_bytes <= 0:
            return False
        if not entry.has_final_latents():
            logger.warning("[inter-request] refusing to store %s without final latents", entry.key)
            return False
        if entry.nbytes() > self.config.max_memory_bytes:
            logger.info(
                "[inter-request] entry %s (%.1f MiB) exceeds the per-rank budget (%.1f MiB); not cached",
                entry.key,
                entry.nbytes() / 2**20,
                self.config.max_memory_bytes / 2**20,
            )
            return False
        entry.created_at = entry.created_at or time.time()
        with self._lock:
            self._meta[entry.key] = self._entry_metadata(entry)
            self._meta.move_to_end(entry.key)
            self._insert_resident(entry)
            if self._dir is not None:
                try:
                    self._persist(entry)
                except OSError as exc:  # pragma: no cover - disk full / permission
                    logger.warning("[inter-request] rank %d: cannot persist %s: %s", self.rank, entry.key, exc)
            self._evict_locked()
        return True

    def _insert_resident(self, entry: H3CacheEntry) -> None:
        previous = self._resident.pop(entry.key, None)
        if previous is not None:
            self._resident_bytes -= previous.nbytes()
        self._resident[entry.key] = entry
        self._resident_bytes += entry.nbytes()

    def _evict_locked(self) -> None:
        while self._resident and (
            len(self._resident) > self.config.max_entries
            or self._resident_bytes > self.config.max_memory_bytes
        ):
            key, entry = self._resident.popitem(last=False)
            self._resident_bytes -= entry.nbytes()
            self.evictions += 1
            if self._dir is None:
                self._meta.pop(key, None)
            logger.info(
                "[inter-request] rank %d: evicted %s (%.1f MiB); resident=%d entries %.1f MiB",
                self.rank,
                key,
                entry.nbytes() / 2**20,
                len(self._resident),
                self._resident_bytes / 2**20,
            )

    # -------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._meta),
                "resident": len(self._resident),
                "resident_mib": round(self._resident_bytes / 2**20, 1),
                "hits": self.hits,
                "misses": self.misses,
                "semantic_hits": self.semantic_hits,
                "evictions": self.evictions,
                "persistent_dir": self._dir,
            }

    def summary(self) -> str:
        stats = self.stats()
        return (
            f"rank={self.rank} entries={stats['entries']} resident={stats['resident']} "
            f"resident_mib={stats['resident_mib']} hits={stats['hits']} misses={stats['misses']} "
            f"semantic_hits={stats['semantic_hits']} evictions={stats['evictions']}"
        )


__all__ = [
    "H3CacheEntry",
    "H3InterRequestCacheStore",
    "H3StepCheckpoint",
    "SCHEMA_VERSION",
]
