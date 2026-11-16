# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Group-first DLO hooks must re-materialize an offloaded block.

``offload_layer`` replaces a block's weights with zero-sized placeholders while
the shared slot keeps this group's tag.  A stage that never runs in between (a
cross-request EXACT cache hit skips the whole DiT stage) leaves that state in
place, so the next stage's first block must re-prefetch instead of trusting the
slot tag: otherwise the block runs with empty weights and the operator fails
with "Parameter gamma of RmsNorm has incorrect shape [[0]]".
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadHook,
)

pytestmark = pytest.mark.cpu


class _PrevHookStub:
    """Minimal predecessor hook that records the prefetch requests."""

    def __init__(self) -> None:
        self.prefetch_calls: list[tuple[int, bool]] = []
        self.waits: list[int] = []
        self._prefetched_slot: int | None = None
        self.ready_events: list[object | None] = [None, None]

    def prefetch_layer(self, slot: int, non_blocking: bool = True) -> None:
        self.prefetch_calls.append((slot, non_blocking))
        self._prefetched_slot = slot

    def _wait_for_weights(self, slot: int) -> None:
        self.waits.append(slot)


def _hook(*, materialized: bool, slot_group: int = 1, current_slot: int = 0) -> tuple[DistributedLayerwiseOffloadHook, _PrevHookStub, list]:
    hook = object.__new__(DistributedLayerwiseOffloadHook)
    prev = _PrevHookStub()
    own_calls: list[tuple[int, bool]] = []
    hook._is_group_first = True
    hook._prev_hook = prev
    hook.current_slot = current_slot
    hook._group_id = 1
    hook._shared_slot_group = [slot_group, slot_group]
    hook._shared_slot_owner = [id(prev), id(prev)]
    probe = torch.zeros(4)
    if not materialized:
        probe.data = torch.empty(0)
    hook._materialization_probe = probe
    hook.prefetch_layer = lambda slot, non_blocking=True: own_calls.append((slot, non_blocking))
    return hook, prev, own_calls


def test_offloaded_block_is_refetched_even_when_the_slot_keeps_our_group():
    hook, prev, own = _hook(materialized=False, slot_group=1)

    hook.pre_forward(None)

    assert prev.prefetch_calls == [(0, False)], "an offloaded block must be re-fetched"
    assert prev.waits == [0]
    assert own == [(1, True)], "the next slot is still prefetched for the ring"


def test_materialized_block_still_skips_the_sync_prefetch():
    hook, prev, own = _hook(materialized=True, slot_group=1)

    hook.pre_forward(None)

    assert prev.prefetch_calls == [], "a live block must not pay for a redundant copy"
    assert prev.waits == [0]
    assert own == [(1, True)]


def test_foreign_group_still_triggers_a_refetch():
    hook, prev, _ = _hook(materialized=True, slot_group=0)

    hook.pre_forward(None)

    assert prev.prefetch_calls == [(0, False)]


def test_owner_mismatch_still_triggers_a_refetch():
    hook, prev, _ = _hook(materialized=True, slot_group=1)
    hook._shared_slot_owner = [-1, -1]

    hook.pre_forward(None)

    assert prev.prefetch_calls == [(0, False)]


def test_materialization_probe_tracks_storage_release():
    tensor = torch.zeros(8)
    hook, _, _ = _hook(materialized=True)
    hook._materialization_probe = tensor
    assert hook.is_materialized is True
    tensor.data = torch.empty(0)
    assert hook.is_materialized is False
    meta = torch.empty((0,), device="meta")
    hook._materialization_probe = meta
    assert hook.is_materialized is False
