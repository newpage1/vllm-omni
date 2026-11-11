# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cache keys for the MiniMax-H3 cross-request DiT cache.

A key captures everything that changes the denoised result: the normalised
prompt, sampling parameters, the packed latent shape and a fingerprint of the
model/attention/quantisation stack.  ``shape_key`` is the prompt-independent
part (it keeps the seed, because resuming mid-schedule reuses that seed's
initial noise); the semantic (CLIP) path may only reuse an entry with the same
``shape_key``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def normalize_prompt(prompt: str) -> str:
    """Collapse whitespace so trivially different strings share a key."""
    return " ".join(str(prompt).split())


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]


def build_model_fingerprint(od_config: Any, *, world_size: int = 1) -> str:
    """Fingerprint the weight/attention stack that a cached result depends on."""

    def read(name: str, default: Any = None) -> Any:
        try:
            return getattr(od_config, name)
        except AttributeError:
            return default

    def as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            return _stable_json(value)
        return repr(value)

    payload = {
        "model": as_text(read("model")),
        "revision": as_text(read("revision")),
        "model_class": as_text(read("model_class_name")),
        "dtype": as_text(read("dtype")),
        "quantization": as_text(read("quantization_config")),
        "attention_backend": as_text(read("diffusion_attention_backend")),
        "attention_config": as_text(read("diffusion_attention_config")),
        "load_format": as_text(read("diffusion_load_format")),
        "world_size": int(world_size),
        "schema": SCHEMA_VERSION,
    }
    return _fingerprint(payload)


@dataclass(frozen=True)
class H3CacheKey:
    """Hashable description of one H3 denoise request."""

    prompt: str
    seed: int
    task: str
    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int
    num_frames: int
    fps: int
    duration: float
    num_steps: int
    video_shift: float
    audio_shift: float
    sampler: str
    model_fingerprint: str
    cache_config_hash: str
    world_size: int
    shape_key: str = field(default="", compare=False)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.shape_key:
            shape_key = _stable_json(
                {
                    "task": self.task,
                    "latent_shape": [self.latent_t, self.latent_h, self.latent_w],
                    "audio_t": self.audio_t,
                    "num_frames": self.num_frames,
                    "fps": self.fps,
                    "num_steps": self.num_steps,
                    "video_shift": self.video_shift,
                    "audio_shift": self.audio_shift,
                    "sampler": self.sampler,
                    "seed": int(self.seed),
                    "model_fingerprint": self.model_fingerprint,
                }
            )
            object.__setattr__(self, "shape_key", shape_key)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "prompt": self.prompt,
            "seed": int(self.seed),
            "task": self.task,
            "latent_t": int(self.latent_t),
            "latent_h": int(self.latent_h),
            "latent_w": int(self.latent_w),
            "audio_t": int(self.audio_t),
            "num_frames": int(self.num_frames),
            "fps": int(self.fps),
            "duration": round(float(self.duration), 6),
            "num_steps": int(self.num_steps),
            "video_shift": round(float(self.video_shift), 6),
            "audio_shift": round(float(self.audio_shift), 6),
            "sampler": self.sampler,
            "model_fingerprint": self.model_fingerprint,
            "cache_config_hash": self.cache_config_hash,
            "world_size": int(self.world_size),
            "shape_key": self.shape_key,
        }

    def digest(self) -> str:
        return hashlib.sha256(_stable_json(self.payload()).encode("utf-8")).hexdigest()

    def meta(self) -> dict[str, Any]:
        """Prompt-independent metadata stored alongside an entry."""
        return {
            "shape_key": self.shape_key,
            "model_fingerprint": self.model_fingerprint,
            "num_steps": int(self.num_steps),
            "task": self.task,
            "latent_shape": [int(self.latent_t), int(self.latent_h), int(self.latent_w)],
            "audio_t": int(self.audio_t),
            "num_frames": int(self.num_frames),
            "fps": int(self.fps),
            "duration": round(float(self.duration), 6),
            "video_shift": round(float(self.video_shift), 6),
            "audio_shift": round(float(self.audio_shift), 6),
            "sampler": self.sampler,
            "world_size": int(self.world_size),
        }

    def describe(self) -> str:
        return (
            f"task={self.task} seed={self.seed} steps={self.num_steps} "
            f"latent={self.latent_t}x{self.latent_h}x{self.latent_w} audio_t={self.audio_t} "
            f"frames={self.num_frames}@{self.fps}fps shifts=({self.video_shift},{self.audio_shift}) "
            f"sampler={self.sampler} prompt={self.prompt[:48]!r}"
        )


def build_h3_cache_key(
    *,
    od_config: Any,
    prompt: str,
    seed: int,
    task: str,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    num_frames: int,
    num_steps: int,
    video_shift: float,
    audio_shift: float,
    sampler: str,
    fps: int,
    cache_config_hash: str,
    world_size: int = 1,
) -> H3CacheKey:
    normalized = normalize_prompt(prompt)
    if not normalized:
        raise ValueError("MiniMax H3 inter-request cache requires a non-empty prompt")
    return H3CacheKey(
        prompt=normalized,
        seed=int(seed),
        task=str(task),
        latent_t=int(latent_t),
        latent_h=int(latent_h),
        latent_w=int(latent_w),
        audio_t=int(audio_t),
        num_frames=int(num_frames),
        fps=int(fps),
        duration=float(num_frames) / float(fps),
        num_steps=int(num_steps),
        video_shift=float(video_shift),
        audio_shift=float(audio_shift),
        sampler=str(sampler),
        model_fingerprint=build_model_fingerprint(od_config, world_size=world_size),
        cache_config_hash=str(cache_config_hash),
        world_size=int(world_size),
    )


__all__ = [
    "H3CacheKey",
    "SCHEMA_VERSION",
    "build_h3_cache_key",
    "build_model_fingerprint",
    "normalize_prompt",
]
