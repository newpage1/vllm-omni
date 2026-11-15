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
    "peak_scope",
    "request_base_allocated_bytes",
    "peak_delta_bytes",
    "peak_delta_scope",
    "parent",
    "shape",
    "dtype",
    "storage_ptr",
    "storage_bytes",
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
        assert record["peak_scope"] == "process_lifetime"
        assert record["peak_delta_scope"] == "since_request_start"
        assert record["parent"] is None
        assert record["shape"] == [8, 16]
        assert record["dtype"] == "bfloat16"
        assert record["storage_bytes"] >= 8 * 16 * 2
    assert records[0]["storage_ptr"] is not None
    assert spy.calls == 2  # one synchronise before BEGIN, one before END


def test_nested_stage_records_its_parent(monkeypatch):
    """t86/O3: nested stages carry their enclosing phase, they are not additive."""
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(0)
    with phase_probe.phase("attn.local_kernel"):
        pass
    with phase_probe.phase("ffn.fc1"):
        pass
    with phase_probe.phase("qkv.proj"):
        pass
    records = phase_probe.recent_records()
    parents = {record["stage"]: record["parent"] for record in records if record["edge"] == "BEGIN"}
    assert parents == {
        "attn.local_kernel": "block.attn",
        "ffn.fc1": "block.mlp",
        "qkv.proj": None,
    }
    assert phase_probe.stage_parent("block.attn") is None


def test_peak_delta_is_measured_from_the_request_baseline(monkeypatch):
    """t86/F2: allocator peaks stay process-wide; peak_delta_bytes is per request."""
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(0)

    snapshots = iter(
        [
            # reset_peaks() baseline
            {
                "allocated_bytes": 1000,
                "reserved_bytes": 2000,
                "peak_allocated_bytes": 5000,
                "peak_reserved_bytes": 6000,
            },
            # first record
            {
                "allocated_bytes": 1500,
                "reserved_bytes": 2500,
                "peak_allocated_bytes": 5000,
                "peak_reserved_bytes": 6000,
            },
            # second record: allocator peak grew, request view grows too
            {
                "allocated_bytes": 4000,
                "reserved_bytes": 4500,
                "peak_allocated_bytes": 9000,
                "peak_reserved_bytes": 9500,
            },
        ]
    )
    monkeypatch.setattr(phase_probe, "memory_snapshot", lambda: dict(next(snapshots)))
    phase_probe.reset_peaks()
    with phase_probe.phase("block.attn"):
        pass

    records = [r for r in phase_probe.recent_records() if r["edge"] == "BEGIN"]
    assert records[0]["peak_allocated_bytes"] == 5000  # untouched process peak
    assert records[0]["request_base_allocated_bytes"] == 1000
    assert records[0]["peak_delta_bytes"] == 4000
    assert records[0]["peak_scope"] == "process_lifetime"


def test_legacy_env_dir_still_works(monkeypatch, tmp_path):
    """t86/O1: H3_MEM_DIR is a deprecated alias for H3_MEM_JSONL_DIR."""
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    monkeypatch.delenv(phase_probe.ENV_DIR, raising=False)
    legacy_dir = tmp_path / "legacy"
    monkeypatch.setenv(phase_probe.ENV_DIR_LEGACY, str(legacy_dir))
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(0)
    with phase_probe.phase("qkv.proj"):
        pass
    assert (legacy_dir / "h3_mem_phases_rank0.jsonl").exists()


def test_jsonl_dir_wins_over_the_legacy_alias(monkeypatch, tmp_path):
    monkeypatch.setenv(phase_probe.ENV_PHASES, "1")
    monkeypatch.setenv(phase_probe.ENV_RANKS, "0")
    modern = tmp_path / "modern"
    monkeypatch.setenv(phase_probe.ENV_DIR, str(modern))
    monkeypatch.setenv(phase_probe.ENV_DIR_LEGACY, str(tmp_path / "legacy"))
    phase_probe.reset_state_for_tests()
    phase_probe.set_block(0)
    with phase_probe.phase("qkv.proj"):
        pass
    assert (modern / "h3_mem_phases_rank0.jsonl").exists()
    assert not (tmp_path / "legacy" / "h3_mem_phases_rank0.jsonl").exists()


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
