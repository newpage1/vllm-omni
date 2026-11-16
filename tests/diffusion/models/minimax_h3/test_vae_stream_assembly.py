# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the streamed (preallocated-buffer) tile assembly.

The rank-local tiled decode assembles one decoded clip frame from the tiles.
The stock assembly concatenates a row of tiles, then concatenates the rows, so
the frame exists twice at the peak. ``H3_VAE_STREAM_ASSEMBLY=1`` copies every
blended/trimmed tile into a single preallocated frame instead.

These tests drive the real adapter functions against a fake checkpoint that
reproduces ``split_tiles`` and ``blend`` verbatim, and check:

* element-exact equality with a verbatim mirror of the checkpoint's
  concatenating ``tiled_decode`` (several geometries, including edge tiles);
* the concatenations really disappear (``torch.cat`` call accounting);
* the live intermediate bytes shrink (weakref + ``gc`` accounting);
* the switch defaults to off and the default path stays the concatenating one.
"""

from __future__ import annotations

import gc
import math
import weakref

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import rank_local_decode

pytestmark = pytest.mark.cpu


class _FakeCheckpointVideoVAE:
    """Minimal stand-in for ``AutoencoderKLLegacy`` (decode side only)."""

    def __init__(self, *, tile_size: int, overlap_min: int, ratio: int = 2, channels: int = 3):
        # The real decoder maps tile pixels to latent positions by integer
        # division, so a tile that is not a multiple of the upscale factor can
        # never be requested (its ``tile_len // ratio`` would lose pixels).
        assert tile_size % ratio == 0, "tile_size must be a multiple of the upscale factor"
        self.vae_ratio = ratio
        self.decoder_tile_size = tile_size
        self.tile_size = tile_size
        self.decoder_tile_overlap_min = overlap_min
        self.tile_overlap_min = overlap_min
        self.channels = channels
        self.decode_calls: list[tuple[int, int]] = []

    # -- checkpoint methods reproduced verbatim -------------------------
    def split_tiles(self, input_len, is_decoder=False):
        tile_size = self.decoder_tile_size if is_decoder else self.tile_size
        tile_overlap_min = (
            self.decoder_tile_overlap_min if is_decoder else self.tile_overlap_min
        )

        if tile_size >= input_len:
            return [0], [input_len], []

        n = math.ceil(input_len / tile_size)
        while True:
            overlaps = [tile_overlap_min] * (n - 1)
            remaining = tile_size * n - sum(overlaps) - input_len

            if remaining < 0:
                n += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (n - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(n - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        tile_len = [tile_size] * n
        return tile_start_idx, tile_len, overlaps

    def blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        a_overlap = a[tuple(slice_a)]

        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        b_overlap = b[tuple(slice_b)]

        blended = a_overlap * weight_a + b_overlap * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            b_rest = b[tuple(slice_b_rest)]
            return torch.cat([blended, b_rest], dim=dim)
        return blended

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.decode_calls.append((int(z.shape[-2]), int(z.shape[-1])))
        upsampled = z.repeat_interleave(self.vae_ratio, dim=-1).repeat_interleave(
            self.vae_ratio, dim=-2
        )
        # A fixed zero-mean pattern keeps every tile distinct, so an ordering
        # mistake in the assembly cannot hide behind identical tiles.
        ramp = torch.arange(upsampled.numel(), dtype=upsampled.dtype, device=upsampled.device)
        ramp = (ramp % 17).view_as(upsampled) / 17.0
        return upsampled + ramp


class _FakeAdapter:
    """Stand-in for ``MiniMaxH3VideoVAE`` (only what the decoder touches)."""

    def __init__(self, model: _FakeCheckpointVideoVAE, *, parallel_state=None):
        self.model = model
        self.parallel_size = 1
        self._parallel_state = parallel_state or {
            "sp_rank": 0,
            "sp_size": 1,
            "sp_process_group": None,
        }

    def _native_parallel_state(self):
        return self._parallel_state


def _reference_tiled_decode(model: _FakeCheckpointVideoVAE, z: torch.Tensor) -> torch.Tensor:
    """Verbatim mirror of the checkpoint's concatenating ``tiled_decode``."""

    ratio = model.vae_ratio
    height = int(z.shape[-2]) * ratio
    width = int(z.shape[-1]) * ratio
    y_idx, y_len, y_overlap = model.split_tiles(height, True)
    x_idx, x_len, x_overlap = model.split_tiles(width, True)

    z_tiles: list[list[torch.Tensor]] = []
    for i, y_start in enumerate(y_idx):
        row: list[torch.Tensor] = []
        for j, x_start in enumerate(x_idx):
            y0 = y_start // ratio
            y1 = y0 + y_len[i] // ratio
            x0 = x_start // ratio
            x1 = x0 + x_len[j] // ratio
            row.append(model.decode(z[..., y0:y1, x0:x1]))
        z_tiles.append(row)

    rows: list[list[torch.Tensor]] = []
    for i, row in enumerate(z_tiles):
        rows.append(list(row))

    result_rows: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        result_row: list[torch.Tensor] = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


GEOMETRIES = [
    # (latent_h, latent_w, tile, overlap, ratio) -- the frame is 2x the latent.
    (10, 12, 6, 2, 2),  # 3x3 tiles, non-zero offsets
    (12, 12, 6, 2, 2),  # exact multiple, no remainder
    (5, 8, 8, 2, 2),  # single tile per axis
    (14, 10, 6, 3, 2),  # uneven overlaps on both axes
    (7, 9, 6, 1, 2),  # asymmetric remainder
    (9, 9, 6, 3, 3),  # different upscale factor (tile/overlap divisible by it)
]


def _latent(h: int, w: int, *, channels: int = 24, frames: int = 2) -> torch.Tensor:
    values = torch.arange(frames * channels * h * w, dtype=torch.float32)
    return (values % 41).view(1, channels, frames, h, w) / 41.0 - 0.5


def _run(vae: _FakeAdapter, z: torch.Tensor) -> torch.Tensor:
    return rank_local_decode._rank_local_tiled_decode(vae, z)


@pytest.mark.parametrize(("latent_h", "latent_w", "tile", "overlap", "ratio"), GEOMETRIES)
def test_stream_assembly_matches_the_concatenated_reference(
    monkeypatch, latent_h, latent_w, tile, overlap, ratio
):
    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, "1")
    model = _FakeCheckpointVideoVAE(tile_size=tile, overlap_min=overlap, ratio=ratio)
    vae = _FakeAdapter(model)
    z = _latent(latent_h, latent_w, frames=2)

    expected = _reference_tiled_decode(model, z)
    actual = _run(vae, z)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), "streamed assembly must be bit-identical"


@pytest.mark.parametrize(("latent_h", "latent_w", "tile", "overlap", "ratio"), GEOMETRIES)
def test_default_path_still_uses_the_concatenating_assembly(
    monkeypatch, latent_h, latent_w, tile, overlap, ratio
):
    monkeypatch.delenv(rank_local_decode.STREAM_ASSEMBLY_ENV, raising=False)
    model = _FakeCheckpointVideoVAE(tile_size=tile, overlap_min=overlap, ratio=ratio)
    vae = _FakeAdapter(model)
    z = _latent(latent_h, latent_w, frames=2)

    expected = _reference_tiled_decode(model, z)
    actual = _run(vae, z)
    assert torch.equal(actual, expected)


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(rank_local_decode.STREAM_ASSEMBLY_ENV, raising=False)
    assert rank_local_decode.stream_assembly_requested() is False


@pytest.mark.parametrize("value", ["0", "false", "off", "", "no", "please"])
def test_switch_rejects_non_truthy_values(monkeypatch, value):
    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, value)
    assert rank_local_decode.stream_assembly_requested() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_switch_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, value)
    assert rank_local_decode.stream_assembly_requested() is True


def _count_cat_calls(monkeypatch, vae, z):
    calls = {"cat": 0}
    original_cat = torch.cat

    def counting_cat(tensors, *args, **kwargs):
        calls["cat"] += 1
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)
    out = _run(vae, z)
    return out, calls["cat"]


def test_streamed_assembly_removes_the_row_and_frame_concatenations(monkeypatch):
    latent_h, latent_w, tile, overlap = 14, 10, 6, 3
    model = _FakeCheckpointVideoVAE(tile_size=tile, overlap_min=overlap, ratio=2)
    vae = _FakeAdapter(model)
    z = _latent(latent_h, latent_w, frames=2)

    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, "1")
    streamed, stream_cats = _count_cat_calls(monkeypatch, vae, z)

    monkeypatch.delenv(rank_local_decode.STREAM_ASSEMBLY_ENV, raising=False)
    concatenated, concat_cats = _count_cat_calls(monkeypatch, vae, z)

    assert torch.equal(streamed, concatenated)
    rows = len(model.split_tiles(int(z.shape[-2]) * 2, True)[0])
    # The default path spends one cat per tile row plus one for the frame;
    # the streamed path only keeps the cats inside ``blend``.
    assert concat_cats - stream_cats == rows + 1


def test_streamed_assembly_keeps_fewer_intermediates_alive(monkeypatch):
    """The streamed path must not materialise whole rows and a whole frame.

    The accounting registers every intermediate the assembly creates (blend
    results and ``torch.cat`` results) in a ``weakref`` list and samples the live
    bytes right after each creation. The default path allocates one concatenated
    tensor per tile row plus the returned frame; the streamed path only keeps
    blend results for the previous row and copies into the preallocated frame.
    """

    latent_h, latent_w, tile, overlap = 14, 10, 6, 3
    model = _FakeCheckpointVideoVAE(tile_size=tile, overlap_min=overlap, ratio=2)
    vae = _FakeAdapter(model)
    z = _latent(latent_h, latent_w, frames=2)

    live: list[weakref.ReferenceType] = []
    peak = {"stream": 0, "concat": 0}
    state = {"mode": "stream"}

    def record(tensor: torch.Tensor) -> None:
        live.append(weakref.ref(tensor))
        gc.collect()
        alive = [ref() for ref in live if ref() is not None]
        current = sum(t.nbytes for t in alive)
        peak[state["mode"]] = max(peak[state["mode"]], current)

    original_blend = model.blend
    original_cat = torch.cat

    def tracking_blend(a, b, blend_extent, dim):
        out = original_blend(a, b, blend_extent, dim)
        record(out)
        return out

    def tracking_cat(tensors, *args, **kwargs):
        out = original_cat(tensors, *args, **kwargs)
        record(out)
        return out

    monkeypatch.setattr(model, "blend", tracking_blend)
    monkeypatch.setattr(torch, "cat", tracking_cat)

    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, "1")
    streamed = _run(vae, z)

    live.clear()
    state["mode"] = "concat"
    monkeypatch.delenv(rank_local_decode.STREAM_ASSEMBLY_ENV, raising=False)
    concatenated = _run(vae, z)

    assert torch.equal(streamed, concatenated)
    frame_bytes = streamed.nbytes
    assert peak["stream"] < frame_bytes
    assert peak["concat"] >= frame_bytes
    assert peak["stream"] < peak["concat"]


def test_streamed_assembly_rejects_a_missing_tile():
    model = _FakeCheckpointVideoVAE(tile_size=6, overlap_min=2, ratio=2)
    vae = _FakeAdapter(model)
    z = _latent(10, 10, frames=1)
    y_idx, y_len, y_overlap = model.split_tiles(int(z.shape[-2]) * 2, True)
    x_idx, x_len, x_overlap = model.split_tiles(int(z.shape[-1]) * 2, True)
    tiles: list[torch.Tensor] = []
    for i in range(len(y_idx)):
        for j in range(len(x_idx)):
            tiles.append(model.decode(z[..., : y_len[i] // 2, : x_len[j] // 2].contiguous()))
    tiles[1] = None  # type: ignore[call-overload]

    with pytest.raises(RuntimeError, match=r"missing tile"):
        rank_local_decode._assemble_streamed(
            vae,
            tiles,  # type: ignore[arg-type]
            i_max=len(y_idx),
            j_max=len(x_idx),
            y_len=y_len,
            x_len=x_len,
            y_overlap=y_overlap,
            x_overlap=x_overlap,
        )


def test_streamed_assembly_rejects_a_short_row():
    model = _FakeCheckpointVideoVAE(tile_size=6, overlap_min=2, ratio=2)
    vae = _FakeAdapter(model)
    z = _latent(10, 10, frames=1)
    y_idx, y_len, y_overlap = model.split_tiles(int(z.shape[-2]) * 2, True)
    x_idx, x_len, x_overlap = model.split_tiles(int(z.shape[-1]) * 2, True)
    tiles = [torch.zeros(1, 3, 1, 6, 6) for _ in range(max(len(y_idx) * len(x_idx) - 1, 1))]

    with pytest.raises(RuntimeError, match=r"expected \d+ tiles in row"):
        rank_local_decode._assemble_streamed(
            vae,
            tiles,
            i_max=len(y_idx),
            j_max=len(x_idx),
            y_len=y_len,
            x_len=x_len,
            y_overlap=y_overlap,
            x_overlap=x_overlap,
        )


def test_streamed_assembly_rejects_an_empty_tile_list():
    model = _FakeCheckpointVideoVAE(tile_size=6, overlap_min=2, ratio=2)
    vae = _FakeAdapter(model)
    with pytest.raises(RuntimeError, match=r"received no tiles"):
        rank_local_decode._assemble_streamed(
            vae,
            [],
            i_max=1,
            j_max=1,
            y_len=[6],
            x_len=[6],
            y_overlap=[],
            x_overlap=[],
        )


def test_log_line_reports_the_assembly_mode(monkeypatch, caplog):
    monkeypatch.setenv(rank_local_decode.STREAM_ASSEMBLY_ENV, "1")
    model = _FakeCheckpointVideoVAE(tile_size=6, overlap_min=2, ratio=2)
    vae = _FakeAdapter(model)
    z = _latent(10, 10, frames=1)

    with caplog.at_level("INFO"):
        _run(vae, z)

    assert any("assembly=stream" in record.getMessage() for record in caplog.records)
