# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Configuration for the cross-request (inter-request) DiT cache backend.

The backend reuses :class:`~vllm_omni.diffusion.data.DiffusionCacheConfig`, so
settings can be supplied either through the flat ``inter_request_*`` fields of
that dataclass or through a nested ``inter_request`` object inside
``--cache-config``.  The nested form wins when both are present.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

# ``--cache-config '{"inter_request": {...}}'`` and
# ``--cache-config '{"inter_request_cache": {...}}'`` are both accepted.
_NESTED_KEYS = ("inter_request", "inter_request_cache")

# Legacy PR #2 keys are accepted inside the nested object for convenience.
_NESTED_ALIASES = {
    "clip_min_skip": "min_skip_steps",
    "clip_max_skip_ratio": "max_skip_ratio",
    "persistent_cache_dir": "persistent_dir",
    "min_skip": "min_skip_steps",
    "max_skip_ratio": "max_skip_ratio",
    "record_step_latents": "record_steps",
    "cache_run_id": "run_id",
    "run_id": "run_id",
    "cache_reuse": "reuse_persistent",
    "reuse": "reuse_persistent",
    "reuse_persistent_dir": "reuse_persistent",
}


def _sanitize_token(value: object) -> str:
    """Reduce a run id to a safe single path component."""
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return cleaned[:96] or "run"


_MODES = ("exact", "exact_semantic")


def _lookup(config: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from ``DiffusionCacheConfig`` without raising."""
    try:
        return getattr(config, name)
    except AttributeError:
        return default


@lru_cache(maxsize=8)
def _clip_identity(resolved: str) -> str:
    """Best-effort identity of a CLIP checkpoint tree: ``path|bytes|mtime``.

    The semantic path stores 768-d pooled embeddings, which are only comparable
    when they came from the same text encoder.  Including the on-disk weight
    identity in the config hash makes a swapped/replaced checkpoint invalidate
    the whole cache (conservative) instead of silently matching embeddings from
    a different model.  Identity is read once per process; if the path is not
    readable the plain path is used so every rank still agrees.
    """
    total = 0
    newest = 0
    try:
        root = Path(resolved)
        if root.is_dir():
            weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.bin"))
            for item in weights:
                try:
                    stat = item.stat()
                except OSError:
                    continue
                total += int(stat.st_size)
                newest = max(newest, int(stat.st_mtime_ns))
        elif root.is_file():
            stat = root.stat()
            total = int(stat.st_size)
            newest = int(stat.st_mtime_ns)
    except OSError:
        return resolved
    if not total:
        return resolved
    return f"{resolved}|{total}|{newest}"


@dataclass(frozen=True)
class InterRequestCacheConfig:
    """Settings for :class:`InterRequestCacheBackend`."""

    # "exact" only reuses byte-identical keys, "exact_semantic" additionally
    # resumes from step K when the CLIP prompt similarity clears the threshold.
    mode: str = "exact_semantic"
    # Per-rank limits.  Each rank keeps its own checkpoint shard because the
    # packed H3 rows are already replicated on every DiT rank.
    max_entries: int = 8
    max_memory_bytes: int = 2 * 1024**3
    # Record per-step packed rows so a partial hit can resume at step K.
    record_steps: bool = True
    persistent_dir: str | None = None
    clip_model_path: str | None = None
    clip_device: str = "cpu"
    clip_threshold: float = 0.65
    min_skip_steps: int = 5
    max_skip_ratio: float = 0.5
    # Reserved: number of denoise steps to run without intra-request caching
    # after a resume (the intra backend has its own warmup; kept for parity
    # with the design document).
    resume_warmup_steps: int = 4
    # Composite mode: optionally run an intra-request backend underneath.
    intra_backend: str | None = None
    intra_config: dict[str, Any] = field(default_factory=dict)
    # Run-scoped persistence.  By default the on-disk cache lives under
    # ``<persistent_dir>/run-<run_id>/rank-NN`` so a new engine process can
    # never hit entries recorded by an older process: the recorded rows are
    # allocator state, not plain data, and a stale hit replays another
    # process's decode.  ``reuse_persistent`` keeps the legacy flat layout for
    # the cases where cross-restart reuse is explicitly wanted.
    run_id: str | None = None
    reuse_persistent: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"inter_request mode must be one of {_MODES}, got {self.mode!r}")
        if self.max_entries < 0:
            raise ValueError("inter_request max_entries must be >= 0")
        if self.max_memory_bytes < 0:
            raise ValueError("inter_request max_memory_bytes must be >= 0")
        if not 0.0 <= self.clip_threshold <= 1.0:
            raise ValueError("inter_request clip_threshold must be within [0, 1]")
        if self.min_skip_steps < 0:
            raise ValueError("inter_request min_skip_steps must be >= 0")
        if not 0.0 < self.max_skip_ratio <= 1.0:
            raise ValueError("inter_request max_skip_ratio must be within (0, 1]")
        if self.intra_backend is not None and str(self.intra_backend).lower() in ("", "none"):
            object.__setattr__(self, "intra_backend", None)
        if self.run_id is not None and not str(self.run_id).strip():
            object.__setattr__(self, "run_id", None)
        if self.run_id is not None:
            object.__setattr__(self, "run_id", _sanitize_token(self.run_id))

    # ------------------------------------------------------------------ helpers
    @property
    def semantic_enabled(self) -> bool:
        return self.mode == "exact_semantic" and bool(self.clip_model_path)

    @property
    def intra_backend_name(self) -> str | None:
        return None if self.intra_backend is None else str(self.intra_backend).lower()

    @property
    def run_identity(self) -> str:
        """Identifier of the engine run that owns ``persistent_dir``.

        ``run_id`` wins; otherwise ``VLLM_OMNI_RUN_ID`` (exported by
        ``vllm-omni serve`` before the workers start) is used.  The per-process
        fallback keeps the default safe when the backend is built outside the
        CLI: each process then owns its own directory instead of silently
        sharing rows recorded by another process.
        """
        if self.run_id:
            return str(self.run_id)
        env = os.environ.get("VLLM_OMNI_RUN_ID", "").strip()
        if env:
            return _sanitize_token(env)
        return f"pid{os.getpid()}-{int(time.time() * 1000) % 100000000:08d}"

    @property
    def persistent_root(self) -> str | None:
        """Directory that holds this run's ``rank-NN`` subdirectories."""
        if not self.persistent_dir:
            return None
        base = os.path.abspath(os.path.expanduser(str(self.persistent_dir)))
        if self.reuse_persistent:
            return base
        return os.path.join(base, f"run-{self.run_identity}")

    @property
    def clip_identity(self) -> str:
        """Identity of the CLIP text encoder used by the semantic path.

        Empty when the semantic path is disabled, so exact-only deployments do
        not depend on a CLIP model being present.
        """
        if not self.semantic_enabled:
            return ""
        raw = str(self.clip_model_path)
        try:
            resolved = str(Path(raw).expanduser().resolve())
        except OSError:
            resolved = raw
        return _clip_identity(resolved)

    def config_hash(self) -> str:
        """Stable hash of the knobs that change reuse semantics.

        ``clip`` carries the CLIP checkpoint identity: swapping the text
        encoder changes every key, so entries recorded with another encoder
        can never be reused (their pooled embeddings are not comparable).
        """
        payload = {
            "mode": self.mode,
            "clip": self.clip_identity,
            "clip_threshold": round(float(self.clip_threshold), 6),
            "min_skip_steps": int(self.min_skip_steps),
            "max_skip_ratio": round(float(self.max_skip_ratio), 6),
            "record_steps": bool(self.record_steps),
            "max_entries": int(self.max_entries),
            "intra_backend": self.intra_backend_name,
            "schema": 2,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_cache_config(cls, config: Any) -> "InterRequestCacheConfig":
        """Build from a ``DiffusionCacheConfig`` (or any attribute holder)."""
        nested: dict[str, Any] = {}
        extra = _lookup(config, "_extra_params", None)
        if isinstance(extra, Mapping):
            for name in _NESTED_KEYS:
                value = extra.get(name)
                if isinstance(value, Mapping):
                    nested = dict(value)
                    break

        resolved: dict[str, Any] = {}
        for key, value in nested.items():
            resolved[_NESTED_ALIASES.get(key, key)] = value

        # Flat ``inter_request_*`` fields take effect only when the nested
        # object does not spell the same setting out.
        def flat(name: str, target: str) -> None:
            if target in resolved:
                return
            value = _lookup(config, f"inter_request_{name}", None)
            if value is not None:
                resolved[target] = value

        flat("mode", "mode")
        flat("max_entries", "max_entries")
        flat("record_step_latents", "record_steps")
        flat("persistent_cache_dir", "persistent_dir")
        flat("clip_model_path", "clip_model_path")
        flat("clip_device", "clip_device")
        flat("clip_threshold", "clip_threshold")
        flat("clip_min_skip", "min_skip_steps")
        flat("clip_max_skip_ratio", "max_skip_ratio")
        flat("intra_backend", "intra_backend")
        flat("intra_config", "intra_config")
        flat("resume_warmup_steps", "resume_warmup_steps")
        flat("cache_run_id", "run_id")
        flat("cache_reuse", "reuse_persistent")
        max_memory_gb = resolved.pop("max_memory_gb", None)
        if max_memory_gb is None:
            max_memory_gb = _lookup(config, "inter_request_max_memory_gb", None)
        if max_memory_gb is not None and "max_memory_bytes" not in resolved:
            resolved["max_memory_bytes"] = int(float(max_memory_gb) * 1024**3)
        if "max_memory_bytes" in resolved:
            resolved["max_memory_bytes"] = int(resolved["max_memory_bytes"])

        known = {name: value for name, value in resolved.items() if name in cls.__dataclass_fields__}
        unknown = sorted(set(resolved) - set(known))
        if unknown:
            raise ValueError(f"Unknown inter_request cache settings: {unknown}")
        return cls(**known)


__all__ = ["InterRequestCacheConfig"]
