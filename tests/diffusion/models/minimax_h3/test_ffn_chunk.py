# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the MiniMax-H3 chunked FFN helper (design item B1/D1)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.ffn_chunk import (
    ENV_FFN_CHUNKS,
    chunked_ffn_forward,
    ffn_chunk_count,
    reset_chunk_cache,
)

pytestmark = pytest.mark.cpu

HIDDEN = 256
FFN = 512


class _FakeGateUp(torch.nn.Module):
    """Mimics MergedColumnParallelLinear([ffn, ffn]) -> (output, bias)."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.gate = torch.nn.Parameter(torch.randn(HIDDEN, FFN, generator=generator, dtype=torch.bfloat16) * 0.02)
        self.up = torch.nn.Parameter(torch.randn(HIDDEN, FFN, generator=generator, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor):
        return torch.cat((x @ self.gate, x @ self.up), dim=-1), None


class _FakeAct(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class _FakeDown(torch.nn.Module):
    """Mimics RowParallelLinear -> (output, bias)."""

    def __init__(self, seed: int = 1) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(FFN, HIDDEN, generator=generator, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor):
        return x @ self.weight, None


@pytest.fixture()
def layers():
    torch.manual_seed(7)
    return _FakeGateUp(), _FakeAct(), _FakeDown()


def _reference(x: torch.Tensor, fc1, act, fc2) -> torch.Tensor:
    hidden, _ = fc1(x)
    hidden = act(hidden)
    out, _ = fc2(hidden)
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    base = expected.float().norm().clamp_min(1e-12)
    return float(diff / base)


@pytest.mark.parametrize("chunks", [2, 4, 6])
def test_chunked_matches_reference_bf16(layers, chunks):
    fc1, act, fc2 = layers
    x = torch.randn(131, HIDDEN, dtype=torch.bfloat16)
    expected = _reference(x, fc1, act, fc2)
    actual = chunked_ffn_forward(x, fc1, act, fc2, out_features=HIDDEN, chunks=chunks)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert _rel_l2(actual, expected) <= 5e-3
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_chunk_one_is_the_original_path(layers):
    fc1, act, fc2 = layers
    x = torch.randn(64, HIDDEN, dtype=torch.bfloat16)
    expected = _reference(x, fc1, act, fc2)
    actual = chunked_ffn_forward(x, fc1, act, fc2, out_features=HIDDEN, chunks=1)
    assert torch.equal(actual, expected)


def test_non_divisible_tail_rows(layers):
    """7 rows over 4 chunks: the tail chunk is shorter and must still be exact."""
    fc1, act, fc2 = layers
    x = torch.randn(7, HIDDEN, dtype=torch.bfloat16)
    expected = _reference(x, fc1, act, fc2)
    actual = chunked_ffn_forward(x, fc1, act, fc2, out_features=HIDDEN, chunks=4)
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_degenerate_row_counts(layers, rows):
    fc1, act, fc2 = layers
    x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
    expected = _reference(x, fc1, act, fc2)
    actual = chunked_ffn_forward(x, fc1, act, fc2, out_features=HIDDEN, chunks=4)
    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_env_parsing(monkeypatch):
    reset_chunk_cache()
    monkeypatch.delenv(ENV_FFN_CHUNKS, raising=False)
    assert ffn_chunk_count() == 1
    for value, expected in (("2", 2), ("4", 4), ("6", 6), ("1", 1)):
        reset_chunk_cache()
        monkeypatch.setenv(ENV_FFN_CHUNKS, value)
        assert ffn_chunk_count() == expected
    for bad in ("3", "0", "eight"):
        reset_chunk_cache()
        monkeypatch.setenv(ENV_FFN_CHUNKS, bad)
        with pytest.raises(ValueError):
            ffn_chunk_count()
    reset_chunk_cache()


def test_no_list_of_chunks_is_built(layers, monkeypatch):
    """Chunked execution must not keep every chunk result alive (no list/cat)."""
    fc1, act, fc2 = layers
    x = torch.randn(129, HIDDEN, dtype=torch.bfloat16)
    seen: list[torch.Tensor] = []
    original = fc2.forward

    def _spy(tensor: torch.Tensor):
        seen.append(tensor)
        return original(tensor)

    monkeypatch.setattr(fc2, "forward", _spy)
    out = chunked_ffn_forward(x, fc1, act, fc2, out_features=HIDDEN, chunks=4)
    assert len(seen) == 4
    # Every intermediate chunk is strictly smaller than the full output.
    assert all(item.shape[0] < out.shape[0] for item in seen[:-1])
