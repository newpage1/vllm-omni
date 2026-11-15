# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the MiniMax-H3 residual host swap pool (design item C/D2)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.residual_swap import (
    ENV_RESIDUAL_SWAP,
    ENV_SWAP_HOST_GIB,
    MODE_HOST,
    MODE_OFF,
    ResidualSwapPool,
    get_residual_swap_pool,
    per_rank_budget_bytes,
    reset_swap_cache,
    residual_swap_mode,
    swap_host_budget_bytes,
)

pytestmark = pytest.mark.cpu

HIDDEN = 64
GIB = 1024**3


@pytest.fixture(autouse=True)
def _clean_swap_env(monkeypatch):
    reset_swap_cache()
    monkeypatch.delenv(ENV_RESIDUAL_SWAP, raising=False)
    monkeypatch.delenv(ENV_SWAP_HOST_GIB, raising=False)
    yield
    reset_swap_cache()


def _pool(mb: int = 64) -> ResidualSwapPool:
    return ResidualSwapPool(max_bytes=mb * 1024 * 1024, device="cpu")


def _bnf_residual() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(37, HIDDEN, dtype=torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_roundtrip_is_bitwise_identical(dtype):
    pool = _pool()
    source = _bnf_residual().to(dtype)
    expected = source.clone()
    handle = pool.swap_out(source)
    source = None
    restored = pool.swap_in(handle)
    assert restored.dtype == dtype
    assert restored.shape == expected.shape
    assert torch.equal(restored, expected)
    # Bitwise identity, not merely allclose.
    assert torch.equal(restored.view(torch.uint8), expected.view(torch.uint8))
    pool.close()


def test_pool_reuses_the_host_slot():
    pool = _pool()
    first = _bnf_residual()
    handle = pool.swap_out(first)
    pool.swap_in(handle)
    stats_after_first = pool.stats()
    second = _bnf_residual()
    handle = pool.swap_out(second)
    pool.swap_in(handle)
    stats_after_second = pool.stats()
    assert stats_after_first["resident_bytes"] == stats_after_second["resident_bytes"]
    assert stats_after_second["swaps_out"] == 2
    assert stats_after_second["free_slots"] == 1
    pool.close()


def test_budget_is_enforced_for_pinned_pools():
    pool = _pool(mb=1)
    big = torch.zeros(1024 * 1024, dtype=torch.float32)  # 4 MiB
    if not pool.stats()["pinned"]:
        pytest.skip("pageable CPU fallback does not enforce the pinned budget")
    with pytest.raises(RuntimeError, match="residual swap pool exhausted"):
        pool.swap_out(big)
    pool.close()


def test_drain_is_safe_without_pending_copies():
    pool = _pool()
    pool.drain()
    pool.drain()
    pool.close()


def test_release_callback_drops_the_caller_reference():
    pool = _pool()
    source = _bnf_residual()
    holder = [source]
    handle = pool.swap_out(source, release=lambda: holder.__setitem__(0, None))
    assert holder[0] is None
    restored = pool.swap_in(handle)
    assert torch.equal(restored, source)
    pool.close()


def test_simulated_block_matches_without_swap():
    """off vs host must produce a bitwise identical block output."""
    pool = _pool()
    torch.manual_seed(23)
    weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16) * 0.02
    x = _bnf_residual()
    gate = torch.randn(x.shape[0], HIDDEN, dtype=torch.bfloat16) * 0.1

    def block(residual: torch.Tensor) -> torch.Tensor:
        # norm -> attention stand-in -> gated residual
        normed = torch.nn.functional.rms_norm(residual, (HIDDEN,))
        attended = normed @ weight
        return residual + gate * attended

    expected = block(x.clone())

    derivative = x.clone()
    normed = torch.nn.functional.rms_norm(derivative, (HIDDEN,))
    handle = pool.swap_out(derivative, release=lambda: None)
    derivative = None
    attended = normed @ weight
    restored = pool.swap_in(handle)
    actual = restored + gate * attended
    assert torch.equal(actual, expected)
    pool.close()


def test_off_mode_is_the_default_and_allocates_nothing(monkeypatch):
    assert residual_swap_mode() == MODE_OFF
    pool = get_residual_swap_pool()
    assert pool.stats()["resident_bytes"] == 0
    assert pool.stats()["swaps_out"] == 0
    reset_swap_cache()


def test_host_mode_parsing_and_budget(monkeypatch):
    monkeypatch.setenv(ENV_RESIDUAL_SWAP, "host")
    reset_swap_cache()
    assert residual_swap_mode() == MODE_HOST
    monkeypatch.setenv(ENV_SWAP_HOST_GIB, "48")
    reset_swap_cache()
    assert swap_host_budget_bytes() == 48 * GIB
    assert per_rank_budget_bytes(8) == 6 * GIB
    assert per_rank_budget_bytes(0) == 48 * GIB
    monkeypatch.setenv(ENV_RESIDUAL_SWAP, "device")
    reset_swap_cache()
    with pytest.raises(ValueError):
        residual_swap_mode()
    reset_swap_cache()


def test_pool_exhaustion_is_reported_when_pinned(monkeypatch):
    """A too-small pinned budget must fail loudly instead of silently paging."""
    monkeypatch.setenv(ENV_SWAP_HOST_GIB, "1")
    reset_swap_cache()
    pool = get_residual_swap_pool(device="cpu", world_size=8)  # 128 MiB/rank
    assert pool.max_bytes == GIB // 8
    if not pool.stats()["pinned"]:
        pytest.skip("pageable CPU fallback does not enforce the pinned budget")
    tensors = [torch.empty(16 * 1024 * 1024 // 4) for _ in range(4)]  # 4 x 16 MiB
    with pytest.raises(RuntimeError):
        for tensor in tensors:
            pool.swap_out(tensor)
    reset_swap_cache()
