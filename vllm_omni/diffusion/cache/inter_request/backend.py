# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-request (inter-request) DiT cache backend.

This backend ports the reuse idea of cta-11/vllm-omni PR #2 to the vLLM-Omni
0.32.0 ``CacheBackend`` lifecycle and to MiniMax-H3 T2VA:

* byte-identical request (same key)  -> return the cached pre-VAE latents and
  skip the whole denoise loop (``HIT_EXACT``);
* CLIP prompt similarity >= threshold -> restore the packed video/audio rows of
  step K and continue the Euler loop from K (``HIT_PARTIAL``);
* otherwise -> record per-step packed rows and store the entry on success
  (``MISS``).

The backend can wrap an intra-request backend (currently ``cache_dit``); the
H3 pipeline owns the request-scoped Cache-DiT runtime and this backend only
tells the quality policy which intra backend to install.  Both hit kinds are
logged with an ``[inter-request]`` prefix, next to Cache-DiT's own ``DBCache``
lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend

from .clip import CLIPPromptEmbedder
from .config import InterRequestCacheConfig
from .keys import H3CacheKey, build_h3_cache_key
from .store import H3CacheEntry, H3InterRequestCacheStore, H3StepCheckpoint

logger = init_logger(__name__)

HIT_MISS = 0
HIT_EXACT = 1
HIT_PARTIAL = 2
_HIT_NAMES = {HIT_MISS: "MISS", HIT_EXACT: "HIT_EXACT", HIT_PARTIAL: "HIT_PARTIAL"}

_H3_PIPELINE_ATTR = "_inter_request_cache_backend"
_H3_INTRA_ATTR = "_inter_request_intra_backend"


def dit_rank_world() -> tuple[Any, int, int]:
    """Return ``(process_group, rank, world_size)`` for the DiT ranks."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return None, 0, 1
    from vllm_omni.diffusion.distributed.parallel_state import get_world_group

    group = get_world_group().device_group
    return group, dist.get_rank(group), dist.get_world_size(group)


def compute_skip_steps(
    *,
    num_steps: int,
    similarity: float,
    threshold: float,
    min_skip_steps: int,
    max_skip_ratio: float,
) -> int:
    """Map a prompt similarity to a resume step K.

    ``K`` is monotone in the similarity: ``min_skip_steps`` at the threshold and
    ``int(num_steps * max_skip_ratio)`` (capped at ``num_steps - 1``) for an
    identical prompt.  A similarity below the threshold, or a window too small
    to hold ``min_skip_steps``, disables the skip (``K = 0``).
    """
    if similarity < threshold:
        return 0
    if num_steps < 2:
        return 0
    k_max = min(int(num_steps * float(max_skip_ratio)), num_steps - 1)
    if k_max < min_skip_steps:
        return 0
    span = max(1e-6, 1.0 - float(threshold))
    z = min(1.0, max(0.0, (float(similarity) - float(threshold)) / span))
    return int(min(k_max, min_skip_steps + int(round(z * (k_max - min_skip_steps)))))


@dataclass
class InterRequestDecision:
    kind: int
    start_step: int = 0
    similarity: float = 0.0
    runner_up: float = 0.0
    entry: H3CacheEntry | None = None
    checkpoint: H3StepCheckpoint | None = None
    reason: str = ""
    matched_key: str = ""

    @property
    def is_hit(self) -> bool:
        return self.kind != HIT_MISS

    @property
    def name(self) -> str:
        return _HIT_NAMES.get(self.kind, "UNKNOWN")


class InterRequestSession:
    """Per-``diffuse`` bookkeeping: decision, staged rows, commit/abort."""

    def __init__(
        self,
        backend: "InterRequestCacheBackend",
        pipeline: Any,
        key: H3CacheKey,
        decision: InterRequestDecision,
    ) -> None:
        self.backend = backend
        self.pipeline = pipeline
        self.key = key
        self.decision = decision
        self._checkpoints: list[H3StepCheckpoint] = []
        self._resumed_prefix: list[H3StepCheckpoint] = []
        self._finished = False
        if decision.kind == HIT_PARTIAL and decision.entry is not None:
            start = decision.start_step
            for checkpoint in decision.entry.checkpoints:
                if checkpoint.next_step < start:
                    self._resumed_prefix.append(checkpoint)

    # ------------------------------------------------------------------ record
    def record_after_step(self, step: int, video_rows: torch.Tensor, audio_rows: torch.Tensor) -> None:
        if not self.backend.ir.record_steps or self._finished:
            return
        self._checkpoints.append(
            H3StepCheckpoint(
                next_step=int(step) + 1,
                video_rows=video_rows.detach().to(device="cpu", dtype=torch.float32, copy=True),
                audio_rows=audio_rows.detach().to(device="cpu", dtype=torch.float32, copy=True),
            )
        )

    # ------------------------------------------------------------------ finish
    def exact_latents(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.decision.entry
        if entry is None or entry.video_latent is None or entry.audio_latent is None:
            raise RuntimeError("[inter-request] exact hit without cached final latents")
        self._finished = True
        return entry.video_latent.to(device=device), entry.audio_latent.to(device=device)

    def commit(self, video_latent: torch.Tensor, audio_latent: torch.Tensor) -> None:
        if self._finished:
            return
        self._finished = True
        if not self.backend.ir.record_steps and self.decision.kind != HIT_PARTIAL:
            return
        checkpoints = self._merge_checkpoints()
        entry = H3CacheEntry(
            key=self.key.digest(),
            prompt=self.key.prompt,
            meta=self.key.meta(),
            embedding=self.backend.embedding_for(self.key.prompt),
            checkpoints=checkpoints,
            video_latent=video_latent.detach().to(device="cpu", copy=True),
            audio_latent=audio_latent.detach().to(device="cpu", copy=True),
        )
        stored = self.backend.store.put(entry)
        if stored:
            size_mib = entry.nbytes() / 2**20
            logger.info(
                "[inter-request] rank %d: recorded entry key=%s steps=%d (%s..%s) size=%.1f MiB "
                "prompt=%r total checkpoints=%d %s",
                self.backend.rank,
                entry.key[:16],
                len(entry.checkpoints),
                entry.checkpoints[0].next_step if entry.checkpoints else "-",
                entry.checkpoints[-1].next_step if entry.checkpoints else "-",
                size_mib,
                entry.prompt[:48],
                len(entry.checkpoints),
                self.backend.store.summary(),
            )

    def abort(self) -> None:
        self._finished = True
        self._checkpoints.clear()

    def _merge_checkpoints(self) -> list[H3StepCheckpoint]:
        merged: dict[int, H3StepCheckpoint] = {
            checkpoint.next_step: checkpoint for checkpoint in self._resumed_prefix
        }
        for checkpoint in self._checkpoints:
            merged[checkpoint.next_step] = checkpoint
        return [merged[step] for step in sorted(merged)]


class InterRequestCacheBackend(CacheBackend):
    """Cross-request DiT cache with an optional intra-request backend."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.ir = InterRequestCacheConfig.from_cache_config(config)
        self.store: H3InterRequestCacheStore | None = None
        self._embedder: CLIPPromptEmbedder | None = None
        self._rank = 0
        self._world_size = 1
        self._request_steps: int | None = None
        self._requests = 0

    # --------------------------------------------------------------- lifecycle
    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def enable(self, pipeline: Any) -> None:
        if not hasattr(pipeline, "_cache_dit_runtime") or not hasattr(pipeline, "od_config"):
            raise ValueError(
                "inter_request cache backend currently supports the MiniMax-H3 diffusion pipeline only; "
                f"got {type(pipeline).__name__}"
            )
        group, rank, world_size = dit_rank_world()
        self._rank = rank
        self._world_size = world_size
        self.store = H3InterRequestCacheStore(self.ir, rank=rank, world_size=world_size)
        if self.ir.semantic_enabled and (rank == 0 or world_size == 1):
            self._embedder = CLIPPromptEmbedder(self.ir.clip_model_path, device=self.ir.clip_device)
        setattr(pipeline, _H3_PIPELINE_ATTR, self)
        setattr(pipeline, _H3_INTRA_ATTR, self.ir.intra_backend_name)
        self.enabled = True
        logger.info(
            "[inter-request] backend enabled on %s: rank=%d/%d mode=%s threshold=%.3f "
            "min_skip=%d max_skip_ratio=%.3f record_steps=%s max_entries=%d "
            "persistent_dir=%s clip=%s intra=%s",
            type(pipeline).__name__,
            rank,
            world_size,
            self.ir.mode,
            self.ir.clip_threshold,
            self.ir.min_skip_steps,
            self.ir.max_skip_ratio,
            self.ir.record_steps,
            self.ir.max_entries,
            self.ir.persistent_dir,
            self.ir.clip_model_path,
            self.ir.intra_backend_name or "none",
        )
        if self.ir.intra_backend_name:
            logger.info(
                "[inter-request] composite mode: inter-request (cross-request whole-DiT/step-K reuse) "
                "+ %s (intra-request block skip); cache-dit summary is enabled through the H3 quality policy",
                self.ir.intra_backend_name,
            )
        if self.store is not None and self.store.stats()["entries"]:
            logger.info("[inter-request] cache store ready: %s", self.store.summary())

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        # A request boundary must NOT clear the cross-request store.  The
        # intra backend is refreshed by the pipeline itself once the resume
        # decision is known (it needs N - K, not N).
        self._request_steps = int(num_inference_steps) if num_inference_steps is not None else None
        self._requests += 1
        if verbose:
            logger.info(
                "[inter-request] request #%d boundary: num_inference_steps=%s intra=%s store={%s}",
                self._requests,
                self._request_steps,
                self.ir.intra_backend_name or "none",
                "none" if self.store is None else self.store.summary(),
            )

    def disable(self, pipeline: Any) -> None:
        if getattr(pipeline, _H3_PIPELINE_ATTR, None) is self:
            delattr(pipeline, _H3_PIPELINE_ATTR)
        if getattr(pipeline, _H3_INTRA_ATTR, None) is not None:
            setattr(pipeline, _H3_INTRA_ATTR, None)
        self.enabled = False
        logger.info("[inter-request] backend disabled; store={%s}", self.store.summary() if self.store else "none")

    # --------------------------------------------------------------------- key
    def embedding_for(self, prompt: str) -> torch.Tensor | None:
        if self._embedder is None:
            return None
        return self._embedder.encode(prompt)

    def begin_request(
        self,
        pipeline: Any,
        *,
        prompt: str,
        task: str,
        seed: int,
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
    ) -> InterRequestSession:
        if self.store is None or not self.enabled:
            raise RuntimeError("[inter-request] backend is not enabled")
        key = build_h3_cache_key(
            od_config=pipeline.od_config,
            prompt=prompt,
            seed=seed,
            task=task,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            sampler=sampler,
            fps=fps,
            cache_config_hash=self.ir.config_hash(),
            world_size=self._world_size,
        )
        decision = self._plan(pipeline, key)
        return InterRequestSession(self, pipeline, key, decision)

    # ---------------------------------------------------------------- planning
    def _plan(self, pipeline: Any, key: H3CacheKey) -> InterRequestDecision:
        is_leader = self._world_size == 1 or self._rank == 0
        local = self._plan_local(key) if is_leader else InterRequestDecision(HIT_MISS, reason="non-leader")
        decision = self._sync_decision(pipeline, local)
        if decision.kind == HIT_MISS:
            return decision
        entry = decision.entry
        if entry is None:
            raise RuntimeError(
                f"[inter-request] rank {self._rank} could not restore the cache entry selected by rank 0 "
                f"(matched_key={decision.matched_key[:16] or '<unknown>'}); per-rank stores diverged"
            )
        if decision.kind == HIT_EXACT:
            if not entry.has_final_latents():
                raise RuntimeError(
                    f"[inter-request] exact hit has no final latents (key={entry.key[:16]})"
                )
            decision.start_step = int(key.num_steps)
            logger.info(
                "[inter-request] rank %d: HIT_EXACT key=%s prompt=%r -> skip all %d DiT steps %s",
                self._rank,
                entry.key[:16],
                entry.prompt[:48],
                key.num_steps,
                key.describe(),
            )
            return decision
        checkpoint, actual_step = entry.checkpoint_at(int(decision.start_step))
        if checkpoint is None:
            logger.warning(
                "[inter-request] rank %d: partial hit but no recorded checkpoint at step %d "
                "(checkpoints=%d); falling back to a full run",
                self._rank,
                decision.start_step,
                len(entry.checkpoints),
            )
            return InterRequestDecision(
                HIT_MISS, similarity=decision.similarity, runner_up=decision.runner_up,
                reason="no-checkpoint",
            )
        if actual_step != decision.start_step:
            logger.info(
                "[inter-request] rank %d: requested resume step %d, using nearest recorded step %d",
                self._rank,
                decision.start_step,
                actual_step,
            )
        decision.start_step = int(actual_step)
        decision.checkpoint = checkpoint
        logger.info(
            "[inter-request] rank %d: HIT_PARTIAL key=%s matched=%s prompt=%r similarity=%.4f "
            "-> resume at step %d/%d (skip %d DiT steps) runner_up=%.4f %s",
            self._rank,
            key.digest()[:16],
            entry.key[:16],
            entry.prompt[:48],
            decision.similarity,
            decision.start_step,
            key.num_steps,
            decision.start_step,
            decision.runner_up,
            key.describe(),
        )
        return decision

    def _plan_local(self, key: H3CacheKey) -> InterRequestDecision:
        if self.store is None:
            return InterRequestDecision(HIT_MISS, reason="store-disabled")
        if self._world_size > 1 and self._rank != 0:
            return InterRequestDecision(HIT_MISS, reason="non-leader")
        entry = self.store.get(key.digest())
        if entry is not None and entry.has_final_latents():
            self.store.hits += 1
            return InterRequestDecision(HIT_EXACT, similarity=1.0, entry=entry)
        if self.ir.semantic_enabled and self._embedder is not None:
            embedding = self._embedder.encode(key.prompt)
            hit_key, best, runner_up = self.store.semantic_search(
                embedding, key.shape_key, self.ir.clip_threshold
            )
            if hit_key:
                skip = compute_skip_steps(
                    num_steps=key.num_steps,
                    similarity=best,
                    threshold=self.ir.clip_threshold,
                    min_skip_steps=self.ir.min_skip_steps,
                    max_skip_ratio=self.ir.max_skip_ratio,
                )
                if skip > 0:
                    self.store.semantic_hits += 1
                    return InterRequestDecision(
                        HIT_PARTIAL,
                        start_step=skip,
                        similarity=best,
                        runner_up=runner_up,
                        matched_key=hit_key,
                    )
                logger.info(
                    "[inter-request] rank %d: semantic candidate sim=%.4f but skip=%d is disabled "
                    "(min_skip=%d, max_skip_ratio=%.3f, num_steps=%d)",
                    self._rank,
                    best,
                    skip,
                    self.ir.min_skip_steps,
                    self.ir.max_skip_ratio,
                    key.num_steps,
                )
            elif best >= 0.0:
                logger.info(
                    "[inter-request] rank %d: MISS key=%s best_prompt_similarity=%.4f < threshold=%.3f "
                    "(runner_up=%.4f) %s",
                    self._rank,
                    key.digest()[:16],
                    best,
                    self.ir.clip_threshold,
                    runner_up,
                    key.describe(),
                )
        self.store.misses += 1
        logger.info(
            "[inter-request] rank %d: MISS key=%s record_steps=%s %s store={%s}",
            self._rank,
            key.digest()[:16],
            self.ir.record_steps,
            key.describe(),
            self.store.summary(),
        )
        return InterRequestDecision(HIT_MISS, reason="no-entry")

    def _sync_decision(self, pipeline: Any, local: InterRequestDecision) -> InterRequestDecision:
        """Broadcast rank 0's decision (and matched key) so every DiT rank takes the same branch."""
        group, rank, world_size = dit_rank_world()
        if world_size == 1 or group is None:
            return local
        device = getattr(pipeline, "device", None) or torch.device("npu")
        payload = torch.zeros(4, dtype=torch.long, device=device)
        matched: list[str] = [local.entry.key if local.entry is not None else local.matched_key]
        if rank == 0:
            payload[0] = int(local.kind)
            payload[1] = int(local.start_step)
            payload[2] = int(round(float(local.similarity) * 10000))
            payload[3] = int(round(float(local.runner_up) * 10000))
            matched = [local.entry.key if local.entry is not None else local.matched_key]
        import torch.distributed as dist

        # ``device_group`` ranks are global ranks unless the group was created
        # otherwise; resolve the leader explicitly so the broadcast is correct
        # for any USP layout.
        leader = dist.get_global_rank(group, 0)
        dist.broadcast(payload, src=leader, group=group)
        dist.broadcast_object_list(matched, src=leader, group=group, device=device)
        kind = int(payload[0].item())
        start_step = int(payload[1].item())
        similarity = float(payload[2].item()) / 10000.0
        runner_up = float(payload[3].item()) / 10000.0
        matched_key = str(matched[0]) if matched else ""
        if rank == 0:
            # The leader's semantic branch advertises the matched key without
            # holding the entry; materialise it here so both the checkpoint
            # restore and the commit-prefix merge see the same object.
            if local.entry is None and kind != HIT_MISS and matched_key and self.store is not None:
                local.entry = self.store.get(matched_key)
            local.matched_key = matched_key
            return local
        entry = None
        if kind != HIT_MISS and matched_key and self.store is not None:
            entry = self.store.get(matched_key)
        return InterRequestDecision(
            kind,
            start_step=start_step,
            similarity=similarity,
            runner_up=runner_up,
            entry=entry,
            matched_key=matched_key,
        )


__all__ = [
    "HIT_EXACT",
    "HIT_MISS",
    "HIT_PARTIAL",
    "InterRequestCacheBackend",
    "InterRequestDecision",
    "InterRequestSession",
    "compute_skip_steps",
    "dit_rank_world",
]
