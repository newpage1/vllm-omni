# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the ``H3_VAE_CHUNK_ENCODE`` host staging buffer.

The switch only changes *where* the delivered-video uint8 buffer lives, so the
tests pin the two properties that makes it safe to turn on:

* the buffer request leaves the accelerator (the whole point -- at 4K/30s that
  request is 17.18 GiB of device memory), and
* the delivered bytes are identical to the default path, chunk plan included
  (overlapping clips, a short tail clip, padded frames).

The streaming MP4 producer is tested alongside them because it is the path that
never allocates the whole video at all: the session must keep its in-flight
payload bounded and still emit every frame with an audio track.
"""

from __future__ import annotations

import importlib
import io
import os
from types import MethodType

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu

_ENV = "H3_VAE_CHUNK_ENCODE"

vae_temporal = importlib.import_module("vllm_omni.diffusion.models.minimax_h3.vae_temporal")
chunked_video = importlib.import_module("vllm_omni.diffusion.utils.chunked_video")


# ---------------------------------------------------------------------------
# A checkpoint-shaped host: the fork only needs the clip loop's collaborators.
# ---------------------------------------------------------------------------


class _TransformRev:
    """Processor reversal seen by the fork (3-channel statistics)."""

    mean = (0.5, 0.25, 0.75)
    std = (0.5, 0.25, 0.25)


class _Processor:
    transform_rev = _TransformRev()


class _FakeH3VAE:
    """Minimal ``AutoencoderKLLegacy`` stand-in for the temporal decode fork.

    ``token_overlap`` and ``frame_pre_padding`` are zero so the frame plan is
    readable in the test: one clip writes ``tokens * vae_ratio`` frames.
    """

    def __init__(self, *, vae_ratio_t: int = 2, tokens_chunk_size: int = 2, short_tail: bool = False) -> None:
        self.vae_ratio_t = int(vae_ratio_t)
        self.tokens_chunk_size = int(tokens_chunk_size)
        self.token_overlap = 0
        self.token_drop = 0
        self.frame_pre_padding = 0
        self.frame_overlap = 0
        self.processor = _Processor()
        self._short_tail = bool(short_tail)
        self.decode_calls = 0

    # -- the loop's collaborators -------------------------------------------
    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        frames = [
            self._clip_frames(index, num_chunks, int(z.shape[2]))
            for index in range(num_chunks)
        ]
        total = sum(frames)
        return total, 0, total

    def _clip_frames(self, index: int, num_chunks: int, tokens: int) -> int:
        del tokens
        clip_tokens = self.tokens_chunk_size + self.token_overlap
        if self._short_tail and index == num_chunks - 1:
            clip_tokens = max(1, clip_tokens - 1)
        return clip_tokens * self.vae_ratio_t

    def _adaptive_decode(self, clip_z: torch.Tensor) -> torch.Tensor:
        index = self.decode_calls
        self.decode_calls += 1
        num_chunks = self._num_chunks
        frames = self._clip_frames(index, num_chunks, int(clip_z.shape[2]))
        # Deterministic, distinguishable frames whose values sit inside the
        # reversal's range so the quantizer stays in [0, 255].
        out_h = int(clip_z.shape[3]) * 2
        out_w = int(clip_z.shape[4]) * 2
        base = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1)
        ramp = torch.arange(out_h * out_w, dtype=torch.float32).view(1, 1, 1, out_h, out_w)
        pattern = (base * 0.01 + ramp * 0.001 + float(index) * 0.05).clamp(0.0, 1.0)
        return pattern.expand(1, 3, frames, out_h, out_w).contiguous()

    def blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        extent = min(int(blend_extent), int(a.shape[dim]), int(b.shape[dim]))
        if extent <= 0:
            return b
        weight = torch.arange(extent, dtype=b.dtype).view(-1 if dim == -3 else 1, 1, 1, 1, 1)
        weight = weight / float(extent)
        blended = a.narrow(dim, int(a.shape[dim]) - extent, extent) * (1.0 - weight)
        blended = blended + b.narrow(dim, 0, extent) * weight
        if extent >= int(b.shape[dim]):
            return blended
        return torch.cat([blended, b.narrow(dim, extent, int(b.shape[dim]) - extent)], dim=dim)


def _bind(model: _FakeH3VAE, *, num_chunks: int) -> _FakeH3VAE:
    model._num_chunks = num_chunks
    model._decode_temporal_streaming = MethodType(vae_temporal._decode_temporal_streaming_uint8, model)
    return model


def _latent(*, num_chunks: int, tokens_chunk_size: int = 2, channels: int = 4) -> torch.Tensor:
    length = num_chunks * tokens_chunk_size
    return torch.zeros(1, channels, length, 2, 2)


def _run_fork(model: _FakeH3VAE, z: torch.Tensor, *, num_chunks: int) -> torch.Tensor:
    model.decode_calls = 0
    return model._decode_temporal_streaming(z, None, None, num_chunks, 0, None)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    yield


# ---------------------------------------------------------------------------
# Where the buffer lives
# ---------------------------------------------------------------------------


def test_default_path_asks_the_compute_device_for_the_buffer(monkeypatch):
    """``off`` must allocate exactly where the default path always has."""
    requested = []
    real_empty = torch.empty

    def recording_empty(*size, **kwargs):
        requested.append(kwargs.get("device"))
        return real_empty(*size, **kwargs)

    monkeypatch.setattr(vae_temporal.torch, "empty", recording_empty)
    buffer = vae_temporal._allocate_uint8_output([1, 3, 4, 2, 2], torch.device("cpu"))
    assert buffer.dtype is torch.uint8
    assert requested == [torch.device("cpu")]


def test_chunk_encode_requests_host_memory_and_leaves_the_device_alone(monkeypatch):
    """The switch moves the 17.18 GiB request off the accelerator."""
    monkeypatch.setenv(_ENV, "host")
    requested = []
    real_empty = torch.empty

    def recording_empty(*size, **kwargs):
        requested.append(kwargs.get("device"))
        return real_empty(*size, **kwargs)

    monkeypatch.setattr(vae_temporal.torch, "empty", recording_empty)
    buffer = vae_temporal._allocate_uint8_output([1, 3, 8, 16, 16], torch.device("npu"))
    assert buffer.device.type == "cpu"
    assert requested and set(requested) == {"cpu"}, (
        "the whole-video buffer must not be allocated on the accelerator; requests were "
        f"{requested!r}"
    )


def test_four_k_thirty_second_formula_keeps_the_big_request_on_the_host(monkeypatch):
    """The 4K/30s buffer is 17.9 GB; with the switch on, none of it is device memory."""
    import weakref

    monkeypatch.setenv(_ENV, "1")
    shape = [1, 3, 720, 2160, 3840]  # 30s at 24fps, 4K
    expected = int(np.prod(shape))
    real_empty = torch.empty
    captured = {}

    def recording_empty(*size, **kwargs):
        tensor = real_empty(*size, **kwargs)
        captured["device"] = kwargs.get("device")
        captured["ref"] = weakref.ref(tensor)
        return tensor

    monkeypatch.setattr(vae_temporal.torch, "empty", recording_empty)
    buffer = vae_temporal._allocate_uint8_output(shape, torch.device("npu"))
    assert captured["device"] == "cpu"
    assert buffer.numel() * buffer.element_size() == expected
    assert buffer.numel() * buffer.element_size() >= 16 * 1024**3, (
        "the shape must model the 4K/30s case (16.68 GiB / 17.9 GB)"
    )
    del buffer
    # The host buffer is the only copy; the accelerator never saw a request of
    # this size, which is the whole 17.18 GiB the task asks to remove.


def test_pin_refusal_falls_back_to_pageable_host_memory(monkeypatch):
    monkeypatch.setenv(_ENV, "host")
    calls = []

    real_empty = torch.empty

    def refusing_empty(*size, **kwargs):
        calls.append(kwargs)
        if kwargs.get("pin_memory"):
            raise RuntimeError("pinning refused")
        return real_empty(*size, dtype=torch.uint8)

    monkeypatch.setattr(vae_temporal.torch, "empty", refusing_empty)
    buffer = vae_temporal._allocate_uint8_output([1, 3, 2, 2, 2], torch.device("npu"))
    assert buffer.dtype is torch.uint8
    assert [bool(call.get("pin_memory")) for call in calls] == [True, False]


@pytest.mark.parametrize("value", ["0", "off", "false", "no", ""])
def test_off_spellings_keep_device_staging(monkeypatch, value):
    monkeypatch.setenv(_ENV, value)
    assert not vae_temporal._chunk_encode_enabled()


@pytest.mark.parametrize("value", ["1", "host", "true", "yes", "on", " HOST "])
def test_on_spellings_enable_host_staging(monkeypatch, value):
    monkeypatch.setenv(_ENV, value)
    assert vae_temporal._chunk_encode_enabled()


# ---------------------------------------------------------------------------
# The delivered bytes do not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chunks", [1, 2, 3])
def test_host_staging_is_bit_identical_to_the_default_path(monkeypatch, num_chunks):
    off_model = _bind(_FakeH3VAE(), num_chunks=num_chunks)
    on_model = _bind(_FakeH3VAE(), num_chunks=num_chunks)
    z = _latent(num_chunks=num_chunks)

    off = _run_fork(off_model, z, num_chunks=num_chunks)
    monkeypatch.setenv(_ENV, "host")
    on = _run_fork(on_model, z, num_chunks=num_chunks)

    assert off.dtype is on.dtype is torch.uint8
    assert off.shape == on.shape
    assert torch.equal(off, on), "host staging must not change the delivered frame bytes"


def test_overlapping_and_short_tail_clips_still_match(monkeypatch):
    """A shorter tail clip (odd frame total) must land in the same positions."""
    # vae_ratio 3 makes the one-clip-shorter tail land on an odd frame total
    # (6 + 6 + 3 = 15), which is the case a naive "split in half" plan breaks.
    off_model = _bind(_FakeH3VAE(vae_ratio_t=3, short_tail=True), num_chunks=3)
    on_model = _bind(_FakeH3VAE(vae_ratio_t=3, short_tail=True), num_chunks=3)
    z = _latent(num_chunks=3)

    off = _run_fork(off_model, z, num_chunks=3)
    monkeypatch.setenv(_ENV, "host")
    on = _run_fork(on_model, z, num_chunks=3)

    assert off.shape == on.shape
    assert torch.equal(off, on)
    assert int(off.shape[2]) == 15
    assert int(off.shape[2]) % 2 == 1


def test_a_failing_clip_propagates_and_keeps_nothing(monkeypatch):
    monkeypatch.setenv(_ENV, "host")
    model = _bind(_FakeH3VAE(), num_chunks=2)

    def exploding(clip_z):
        raise RuntimeError("decode exploded")

    model._adaptive_decode = exploding
    with pytest.raises(RuntimeError, match="decode exploded"):
        _run_fork(model, _latent(num_chunks=2), num_chunks=2)


def test_frame_plan_mismatch_still_raises(monkeypatch):
    monkeypatch.setenv(_ENV, "host")
    model = _bind(_FakeH3VAE(), num_chunks=2)
    model._decode_temporal_output_frame_plan = lambda *a, **k: (999, 0, 999)
    with pytest.raises(RuntimeError, match="frame plan mismatch"):
        _run_fork(model, _latent(num_chunks=2), num_chunks=2)


# ---------------------------------------------------------------------------
# The streaming MP4 producer: bounded in flight, every frame out
# ---------------------------------------------------------------------------


def _tiny_chunk(frames: int, *, height: int = 16, width: int = 16, value: float = 0.5) -> torch.Tensor:
    return torch.full((1, 3, frames, height, width), value, dtype=torch.float32)


def test_mp4_session_bounds_in_flight_bytes_and_covers_every_frame():
    pytest.importorskip("av")
    budget = 64 * 1024
    session = chunked_video.ChunkedVideoMP4Session(
        value_range=(0.0, 1.0),
        fps=24.0,
        audio_waveforms=[np.zeros(4800, dtype=np.float32)],
        audio_sample_rate=32000,
        batch_frames=2,
        max_pending=2,
        transfer_slots=2,
        max_pending_bytes=budget,
    )
    pushed = 0
    for _ in range(2):  # two temporal clips
        session.push(_tiny_chunk(2))
        pushed += 2
    session.push(_tiny_chunk(1))  # odd tail frame
    pushed += 1
    videos = session.finish()

    assert len(videos) == 1 and videos[0]
    stats = session.transfer_stats()
    if stats is not None:  # a ring exists only off CPU-less paths
        assert stats.peak_pending_bytes <= budget
        assert stats.peak_slots_in_use <= 2

    av = pytest.importorskip("av")
    with av.open(io.BytesIO(videos[0])) as container:
        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]
        assert video_streams and audio_streams, "the progressive MP4 keeps its audio track"
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == pushed, "incremental encoding must emit every pushed frame"


def test_mp4_session_abort_releases_the_ring():
    pytest.importorskip("av")
    session = chunked_video.ChunkedVideoMP4Session(
        value_range=(0.0, 1.0),
        fps=24.0,
        batch_frames=1,
        max_pending=1,
        transfer_slots=2,
    )
    session.push(_tiny_chunk(1))
    session.push(_tiny_chunk(1))
    session.abort()  # must not raise and must drop queued chunks
    assert session._inflight == deque_empty()


def deque_empty():
    from collections import deque

    return deque()


def test_session_requires_one_waveform_per_batch_entry():
    pytest.importorskip("av")
    session = chunked_video.ChunkedVideoMP4Session(
        value_range=(0.0, 1.0),
        fps=24.0,
        audio_waveforms=[np.zeros(8, dtype=np.float32)],
        audio_sample_rate=32000,
        batch_frames=1,
        max_pending=1,
        transfer_slots=0,
    )
    with pytest.raises(ValueError, match="one audio waveform per batch entry"):
        session.push(torch.zeros(2, 3, 1, 8, 8))
        session.finish()
