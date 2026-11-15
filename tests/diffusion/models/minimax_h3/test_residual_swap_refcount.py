# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Refcount regression for the D2 residual host swap (t86/F1).

``H3_RESIDUAL_SWAP=host`` only saves device memory when the block input is
actually released before attention starts. The pre-t86 loop called
``block(holder[0], ...)``; ``nn.Module.__call__`` forwards ``*args`` through
``_call_impl``, whose frame keeps the ``args`` tuple alive for the whole
``forward``. That tuple is a strong reference the block cannot drop, so
clearing ``x``/``residual``/``holder[0]`` inside the block left the storage
pinned and the swap saved nothing.

These tests drive the real :class:`MiniMaxH3DiTBlock.forward` (only the numeric
kernels are stubbed; they are irrelevant to reference counting) through two
calling conventions:

* :func:`test_block_call_boundary_releases_the_device_storage` uses the t86
  holder convention (``_call_block_with_swap``) and requires the storage to be
  gone *inside* attention;
* :func:`test_legacy_argument_convention_pins_the_storage` keeps the old
  positional-argument convention and asserts that it does **not** release.

Both observe the tensor through a weak reference: any surviving alias (the
``nn.Module`` argument tuple, a stale local, the probe itself, ...) keeps the
weakref alive. The tests never bind the block input to a local that outlives the
call, exactly like the t86 loop (``hidden = None`` before calling): a lingering
local would be the same bug one frame up.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as transformer
from vllm_omni.diffusion.models.minimax_h3 import residual_swap as swap

pytestmark = pytest.mark.cpu

HIDDEN = 32
TOKENS = 6


class _RefProbe:
    """Observes whether the block input is still alive at the swap point."""

    def __init__(self) -> None:
        self.ref: weakref.ReferenceType[torch.Tensor] | None = None
        self.alive_inside_attention: bool | None = None
        self.payload: torch.Tensor | None = None

    def watch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the tensor while keeping only a weak reference to it."""
        self.payload = tensor.detach().clone()
        self.ref = weakref.ref(tensor)
        return tensor

    def record(self) -> None:
        assert self.ref is not None
        self.alive_inside_attention = self.ref() is not None

    @property
    def freed_inside_attention(self) -> bool:
        assert self.alive_inside_attention is not None, "attention phase never ran"
        return not self.alive_inside_attention

    def describe(self) -> str:
        return (
            f"freed_inside_attention={int(self.freed_inside_attention)} "
            f"alive_inside_attention={int(bool(self.alive_inside_attention))}"
        )


class _AttentionStub(nn.Module):
    """Stands in for MiniMaxH3Attention and samples the refcount while it runs."""

    def __init__(self, probe: _RefProbe, *, fail: bool = False) -> None:
        super().__init__()
        self.probe = probe
        self.fail = fail
        self.calls = 0

    def forward(self, h: torch.Tensor, **_: object) -> torch.Tensor:
        self.calls += 1
        gc.collect()
        # Nothing downstream needs the block input itself: the gated residual
        # reads the restored host copy, so returning zeros keeps the block
        # output equal to the residual bytes and creates no hidden alias.
        self.probe.record()
        if self.fail:
            raise RuntimeError("attention exploded")
        return torch.zeros_like(h)


class _AdalnStub(nn.Module):
    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(t_emb.new_zeros(t_emb.shape[0], HIDDEN) for _ in range(6))


class _MlpStub(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h


class _NormStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))
        self.variance_epsilon = 1e-6


def _install_kernel_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the NPU kernels with shape-compatible CPU stand-ins."""

    def _norm_scale_shift(x, weight, shift, scale, combined_indices, eps):
        # A fresh tensor, like the real kernel: returning ``x`` would alias the
        # block input and hide the bug this file is about.
        return x.clone()

    def _gate_norm_scale_shift(residual, gate, h, weight, shift, scale, combined_indices, eps):
        return residual, h

    def _gate(residual, gate, h, combined_indices):
        return residual + h

    monkeypatch.setattr(transformer, "rms_norm_indexed_scale_shift", _norm_scale_shift)
    monkeypatch.setattr(transformer, "indexed_gate_rms_norm_scale_shift", _gate_norm_scale_shift)
    monkeypatch.setattr(transformer, "indexed_gate", _gate)


def _make_block(probe: _RefProbe, *, fail: bool = False) -> transformer.MiniMaxH3DiTBlock:
    block = transformer.MiniMaxH3DiTBlock.__new__(transformer.MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    block.norm1 = _NormStub()
    block.norm2 = _NormStub()
    block.attn = _AttentionStub(probe, fail=fail)
    block.mlp = _MlpStub()
    block.adaln_proj = _AdalnStub()
    return block


def _block_kwargs() -> dict[str, object]:
    return {
        "t_emb": torch.zeros(1, 8),
        "combined_indices": torch.zeros(TOKENS, dtype=torch.long),
        "rope_table": torch.zeros(1),
        "cu_seqlens": torch.tensor([0, TOKENS]),
        "max_seqlen": TOKENS,
        "packed_total": TOKENS,
    }


def _host_swap_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv(swap.ENV_RESIDUAL_SWAP, swap.MODE_HOST)
    swap.reset_swap_cache()
    return _block_kwargs()


def _drive_with_holder(block, holder, kwargs):
    """Call the block the way the t86 loop does (holder convention).

    On a pre-t86 tree the helper does not exist; fall back to the legacy call so
    this same test reports ``freed=0`` there instead of an AttributeError.
    """
    helper = getattr(transformer, "_call_block_with_swap", None)
    if helper is None:
        return _drive_with_positional_arg(block, holder, kwargs)
    return helper(block, holder, kwargs), holder


def _drive_with_positional_arg(block, holder, kwargs):
    """Pre-t86 convention: the tensor travels as a positional argument."""
    out = block(holder[0], release_input=transformer._make_input_release(holder), **kwargs)
    return out, holder


def test_block_call_boundary_releases_the_device_storage(monkeypatch):
    _install_kernel_stubs(monkeypatch)
    probe = _RefProbe()
    block = _make_block(probe)
    kwargs = _host_swap_kwargs(monkeypatch)
    holder = [probe.watch(torch.randn(TOKENS, HIDDEN))]

    out, holder = _drive_with_holder(block, holder, kwargs)

    print(f"[t86] holder convention: {probe.describe()}")
    assert probe.freed_inside_attention, (
        "the block input was still alive inside attention "
        f"(freed={int(probe.freed_inside_attention)}); "
        "the caller kept a strong reference to the swapped residual"
    )
    assert not any(holder), "the holder must be consumed by the block"
    assert torch.equal(out, probe.payload), "the swap must preserve the tensor bytes"
    swap.reset_swap_cache()


def test_legacy_argument_convention_pins_the_storage(monkeypatch):
    """Reproduction of the pre-t86 behaviour: the args tuple keeps the input."""
    _install_kernel_stubs(monkeypatch)
    probe = _RefProbe()
    block = _make_block(probe)
    kwargs = _host_swap_kwargs(monkeypatch)
    holder = [probe.watch(torch.randn(TOKENS, HIDDEN))]

    out, holder = _drive_with_positional_arg(block, holder, kwargs)

    print(f"[t86] legacy convention: {probe.describe()}")
    assert not any(holder)
    assert not probe.freed_inside_attention, (
        "expected the legacy positional call to keep the block input alive inside attention"
    )
    assert torch.equal(out, probe.payload)
    swap.reset_swap_cache()


def test_host_and_off_paths_are_bitwise_identical(monkeypatch):
    _install_kernel_stubs(monkeypatch)
    torch.manual_seed(7)
    reference = torch.randn(TOKENS, HIDDEN)
    kwargs = _block_kwargs()

    monkeypatch.delenv(swap.ENV_RESIDUAL_SWAP, raising=False)
    swap.reset_swap_cache()
    off_probe = _RefProbe()
    off_holder = [off_probe.watch(reference.clone())]
    off_out, _ = _drive_with_holder(_make_block(off_probe), off_holder, kwargs)

    on_kwargs = _host_swap_kwargs(monkeypatch)
    on_probe = _RefProbe()
    on_holder = [on_probe.watch(reference.clone())]
    on_out, _ = _drive_with_holder(_make_block(on_probe), on_holder, on_kwargs)

    assert torch.equal(off_out, on_out)
    # Without the swap the input is still alive at the attention point: that is
    # exactly what the host mode is supposed to change.
    assert off_probe.alive_inside_attention is True
    assert on_probe.freed_inside_attention, on_probe.describe()
    swap.reset_swap_cache()


def test_exception_path_drops_references_and_drains(monkeypatch):
    _install_kernel_stubs(monkeypatch)
    probe = _RefProbe()
    block = _make_block(probe, fail=True)
    kwargs = _host_swap_kwargs(monkeypatch)
    holder = [probe.watch(torch.randn(TOKENS, HIDDEN))]

    with pytest.raises(RuntimeError, match="attention exploded"):
        _drive_with_holder(block, holder, kwargs)

    pool = swap.get_residual_swap_pool(device="cpu", world_size=1)
    assert pool.stats()["outstanding_copies"] == 0, "no copy may stay in flight after a failure"
    swap.reset_swap_cache()


def test_pool_budget_is_split_by_world_size(monkeypatch):
    monkeypatch.setenv(swap.ENV_RESIDUAL_SWAP, swap.MODE_HOST)
    monkeypatch.setenv(swap.ENV_SWAP_HOST_GIB, "48")
    swap.reset_swap_cache()
    pool = swap.get_residual_swap_pool(device="cpu", world_size=8)
    assert pool.max_bytes == swap.per_rank_budget_bytes(8) == 6 * 1024**3
    assert pool.stats()["world_size"] == 8
    # A later caller with a different world size must not silently re-budget.
    same = swap.get_residual_swap_pool(device="cpu", world_size=1)
    assert same is pool
    assert same.stats()["world_size"] == 8
    swap.reset_swap_cache()


def test_holder_requires_a_tensor(monkeypatch):
    _install_kernel_stubs(monkeypatch)
    probe = _RefProbe()
    block = _make_block(probe)
    with pytest.raises(ValueError, match="input_holder was empty"):
        transformer._call_block_with_swap(block, [], _host_swap_kwargs(monkeypatch))
    swap.reset_swap_cache()
