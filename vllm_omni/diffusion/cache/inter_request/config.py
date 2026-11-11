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
from dataclasses import dataclass, field
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
}

_MODES = ("exact", "exact_semantic")


def _lookup(config: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from ``DiffusionCacheConfig`` without raising."""
    try:
        return getattr(config, name)
    except AttributeError:
        return default


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

    # ------------------------------------------------------------------ helpers
    @property
    def semantic_enabled(self) -> bool:
        return self.mode == "exact_semantic" and bool(self.clip_model_path)

    @property
    def intra_backend_name(self) -> str | None:
        return None if self.intra_backend is None else str(self.intra_backend).lower()

    def config_hash(self) -> str:
        """Stable hash of the knobs that change reuse semantics."""
        payload = {
            "mode": self.mode,
            "clip_threshold": round(float(self.clip_threshold), 6),
            "min_skip_steps": int(self.min_skip_steps),
            "max_skip_ratio": round(float(self.max_skip_ratio), 6),
            "record_steps": bool(self.record_steps),
            "max_entries": int(self.max_entries),
            "intra_backend": self.intra_backend_name,
            "schema": 1,
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
