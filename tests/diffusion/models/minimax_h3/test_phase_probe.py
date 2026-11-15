# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the MiniMax-H3 D0 phase probe (design item D0)."""

from __future__ import annotations

import json
from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.utils import phase_probe

pytestmark = pytest.mark.cpu

REQUIRED_FIELDS = (
    "rank",
    "step",
    "block",
    "stage",
    "edge",
    "allocated_bytes",
    "reserved_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "shape",
    "dtype",
    "storage_ptr",
    "storage_bytes",
    "peak_scope",
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    for env in (phase_probe.ENV_PHASES, phase_probe.ENV_RANKS, phase_probe.ENV_BLOCKS):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(phase_probe.ENV_DIR, str(tmp_path))
    phase_probe.reset_state_for_tests()
    yield
    phase_probe.reset_state_for_tests()


class _SyncSpy:
    def __init__(self) -> None:
        self.calls = 0
        self.available = False

    def is_available(self) -> bool:  # pragma: no cover - mirror of torch.npu API
        return self.available

    def synchronize(self) -> None:
        self.calls += 1


def test_default_is_disabled_and_never_synchronises(monkeypatch):
    spy = _SyncSpy()
    monkeypatch.setattr(phase_probe, "_npu_module", lambda: spy)
    assert phase_probe.probe_enabled() is False
    cm = phase_probe.phase("qkv.proj", tensor=torch.zeros(4, 4))
    assert isinstance(cm, type(nullcontext())) or cm is not None
    with cm:
        pass
    assert spy.calls == 0
    assert phase_probe.counters()["begin"] == 0


def test_enabled_emits_begin_end_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    monkeypatch.setenv(phase_probe.ENV_BLOCKS, "0")
    phase_probe.reset_state_for_tests()
    spy = _SyncSpy()
    monkeypatch.setattr(phase_probe, "_npu_module", lambda: spy)
    assert phase_probe.probe_enabled() is True

    tensor = torch.zeros(8, 16, dtype=torch.bfloat16)
    phase_probe.set_step(3)
    phase_probe.set_block(0)
    with phase_probe.phase("qkv.proj", tensor=tensor):
        pass

    path = tmp_path / "h3_mem_phases_rank0.jsonl"
    assert path.exists()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert [record["edge"] for record in records] == ["BEGIN", "END"]
    for record in records:
        for field in REQUIRED_FIELDS:
            assert field in record, field
        assert record["stage"] == "qkv.proj"
        assert record["step"] == 3
        assert record["block"] == 0
        assert record["peak_scope"] == "request_cumulative"
        assert record["shape"] == [8, 16]
        assert record["dtype"] == "bfloat16"
        assert record["storage_bytes"] >= 8 * 16 * 2
    assert records[0]["storage_ptr"] is not None
    assert spy.calls == 2  # one synchronise before BEGIN, one before END


def test_block_filter_skips_unselected_blocks(monkeypatch):
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    monkeypatch.setenv(phase_probe.ENV_BLOCKS, "0")
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(7)
    with phase_probe.phase("ffn.fc1"):
        pass
    assert phase_probe.counters()["begin"] == 0


def test_rank_filter_skips_unselected_ranks(monkeypatch):
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "7")
    phase_probe.reset_state_for_tests()
    monkeypatch.setattr(phase_probe, "_global_rank", lambda: 0)
    with phase_probe.phase("ffn.fc1"):
        pass
    assert phase_probe.counters()["begin"] == 0


def test_error_dump_contains_the_original_exception(monkeypatch, tmp_path):
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(0)
    with pytest.raises(RuntimeError, match="NPU out of memory"):
        with phase_probe.phase("block.attn", extra={"note": "unit"}):
            raise RuntimeError("NPU out of memory")
    dump = tmp_path / "h3_mem_oom_rank0.jsonl"
    assert dump.exists()
    lines = [json.loads(line) for line in dump.read_text().splitlines()]
    assert lines[0]["kind"] == "probe_error"
    assert "NPU out of memory" in (lines[0]["error"] or "")
    assert lines[0]["last_begin"]["stage"] == "block.attn"
    # The END record of the failing phase is written before the dump.
    assert lines[-1]["edge"] == "END"
    assert "error" in lines[-1]
