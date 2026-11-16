# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run-scoped persistence for the inter-request DiT cache.

The recorded rows are allocator state copied out of a live process, so an
engine restart must not replay rows that a *different* process recorded.  These
tests pin that contract: the default layout is ``<dir>/run-<id>/rank-NN``, and
the legacy flat layout is opt-in through ``reuse_persistent``.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from vllm_omni.diffusion.cache.inter_request import InterRequestCacheConfig
from vllm_omni.diffusion.cache.inter_request.store import (
    H3CacheEntry,
    H3InterRequestCacheStore,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = pytest.mark.cpu


def _entry(key: str = "k1", prompt: str = "a cat on a lawn") -> H3CacheEntry:
    return H3CacheEntry(
        key=key,
        prompt=prompt,
        meta={"shape_key": "sk-1", "num_steps": 4},
        video_latent=torch.arange(8, dtype=torch.float32),
        audio_latent=torch.arange(4, dtype=torch.float32),
    )


def test_default_layout_is_run_scoped(tmp_path):
    base = tmp_path / "cache"
    first = H3InterRequestCacheStore(InterRequestCacheConfig(persistent_dir=str(base), run_id="run-a"))
    assert first.put(_entry()) is True
    assert first.dir_path == str(base / "run-run-a" / "rank-00")
    assert (base / "run-run-a" / "rank-00" / "manifest.json").exists()

    # A restart gets a different run id, so the previous rows stay invisible.
    second = H3InterRequestCacheStore(InterRequestCacheConfig(persistent_dir=str(base), run_id="run-b"))
    assert second.dir_path == str(base / "run-run-b" / "rank-00")
    assert second.get("k1") is None


def test_same_run_id_reloads_the_manifest(tmp_path):
    base = tmp_path / "cache"
    first = H3InterRequestCacheStore(InterRequestCacheConfig(persistent_dir=str(base), run_id="run-a"))
    assert first.put(_entry())

    again = H3InterRequestCacheStore(InterRequestCacheConfig(persistent_dir=str(base), run_id="run-a"))
    loaded = again.get("k1")
    assert loaded is not None
    assert loaded.restored is True
    assert torch.equal(loaded.video_latent, _entry().video_latent)


def test_reuse_persistent_keeps_the_flat_layout(tmp_path):
    base = tmp_path / "cache"
    first = H3InterRequestCacheStore(
        InterRequestCacheConfig(persistent_dir=str(base), run_id="run-a", reuse_persistent=True)
    )
    assert first.put(_entry())
    assert first.dir_path == str(base / "rank-00")

    # Explicit reuse: another run id sees the same flat directory on purpose.
    again = H3InterRequestCacheStore(
        InterRequestCacheConfig(persistent_dir=str(base), run_id="run-b", reuse_persistent=True)
    )
    assert again.dir_path == str(base / "rank-00")
    assert again.get("k1") is not None


def test_env_run_id_scopes_the_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_RUN_ID", "nightly/2026-11-17")
    config = InterRequestCacheConfig(persistent_dir=str(tmp_path / "cache"))
    assert config.run_identity == "nightly_2026-11-17"
    assert config.persistent_root == str(tmp_path / "cache" / "run-nightly_2026-11-17")


def test_process_fallback_is_unique_per_process(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_RUN_ID", raising=False)
    config = InterRequestCacheConfig(persistent_dir=str(tmp_path / "cache"))
    assert config.run_identity.startswith("pid")
    assert config.persistent_root.endswith(("run-" + config.run_identity))


def test_run_id_is_sanitized(tmp_path):
    config = InterRequestCacheConfig(persistent_dir=str(tmp_path), run_id="  a/b c..d  ")
    assert config.run_id == "a_b_c..d"


def test_empty_run_id_falls_back_to_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_RUN_ID", "env-3")
    config = InterRequestCacheConfig(persistent_dir=str(tmp_path), run_id="   ")
    assert config.run_id is None
    assert config.run_identity == "env-3"


def test_flat_diffusion_fields_are_read(tmp_path):
    cache_config = DiffusionCacheConfig(
        inter_request_persistent_cache_dir=str(tmp_path),
        inter_request_cache_run_id="flat-9",
    )
    config = InterRequestCacheConfig.from_cache_config(cache_config)
    assert config.run_id == "flat-9"
    assert config.reuse_persistent is False
    assert config.persistent_root == str(tmp_path / "run-flat-9")


def test_flat_reuse_flag_is_read(tmp_path):
    cache_config = DiffusionCacheConfig(
        inter_request_persistent_cache_dir=str(tmp_path),
        inter_request_cache_run_id="flat-9",
        inter_request_cache_reuse=True,
    )
    config = InterRequestCacheConfig.from_cache_config(cache_config)
    assert config.reuse_persistent is True
    assert config.persistent_root == str(tmp_path)


def test_nested_block_can_request_reuse(tmp_path):
    cache_config = DiffusionCacheConfig.from_dict(
        {"inter_request": {"persistent_dir": str(tmp_path), "reuse_persistent": True}}
    )
    config = InterRequestCacheConfig.from_cache_config(cache_config)
    assert config.reuse_persistent is True


def test_unknown_scope_key_is_rejected():
    with pytest.raises(ValueError):
        InterRequestCacheConfig.from_cache_config(
            DiffusionCacheConfig.from_dict({"inter_request": {"not_a_setting": 1}})
        )


def test_run_scope_does_not_change_the_config_hash(tmp_path):
    a = InterRequestCacheConfig(persistent_dir=str(tmp_path), run_id="run-a")
    b = InterRequestCacheConfig(persistent_dir=str(tmp_path), run_id="run-b", reuse_persistent=True)
    assert a.config_hash() == b.config_hash()


def test_projection_flags_reach_the_cache_config(tmp_path):
    """The two new CLI flags are plain projection fields merged on init."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    od_config = OmniDiffusionConfig(
        model="dummy",
        cache_backend="inter_request",
        inter_request_persistent_cache_dir=str(tmp_path),
        inter_request_cache_run_id="cli-run-1",
        inter_request_cache_reuse=True,
    )
    assert od_config.cache_config.inter_request_cache_run_id == "cli-run-1"
    assert od_config.cache_config.inter_request_cache_reuse is True

    config = InterRequestCacheConfig.from_cache_config(od_config.cache_config)
    assert config.run_identity == "cli-run-1"
    assert config.reuse_persistent is True
    assert config.persistent_root == str(tmp_path)


def test_serve_registers_the_scope_flags():
    """``serve`` must expose both flags; building the parser loads vLLM plugins."""
    import inspect

    from vllm_omni.entrypoints.cli import serve

    source = inspect.getsource(serve)
    assert '"--inter-request-cache-run-id"' in source
    assert '"--inter-request-cache-reuse"' in source
    assert 'os.environ["VLLM_OMNI_RUN_ID"]' in source
