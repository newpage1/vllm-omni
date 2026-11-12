# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank-local uint8 tile assembly for the MiniMax-H3 video VAE decode.

The checkpoint's tiled decoder (``AutoencoderKL.tiled_decode``) decodes the
tiles each rank owns, then gathers **every rank's float32 tiles to every
rank** (``_all_gather_tiled_results``), blends them, and concatenates a
full-frame float32 video before the caller quantizes it to uint8 a few
milliseconds later. On the canonical 1344x768 geometry the video decode is the
single largest memory spike of a request (``t55``: 25.4 GB per card against a
6.45 GB ready baseline), so the transient float32 tile, gather and frame
buffers are the first thing to remove.

With ``VLLM_OMNI_VAE_RANK_LOCAL_DECODE=1`` the adapter replaces the
checkpoint's ``tiled_decode`` (instance binding, same pattern as the XPU
persistent-gather override in ``vae.py``) with the implementation below:

* every rank decodes only its own round-robin tiles and converts each tile to
  uint8 **before** it reaches any collective, so float32 tile results never
  accumulate;
* tiles travel as uint8 (a quarter of the bytes) into a single grow-only
  workspace, instead of a fresh float32 receive buffer per call;
* the frame is assembled tile by tile into a uint8 tensor; only the overlap
  strips are promoted to float32 for blending.

Tile count, ownership, blend geometry and the returned layout mirror the
checkpoint, so the result matches the stock path up to uint8 rounding inside
the blend seams (``notes/rank_local_decode_proto.md`` records the measured
PSNR/SSIM against the stock decode).

The switch is deliberately one-way per process: the first decode installs the
override and every later decode reuses it. ``decode_with_chunks`` restores the
checkpoint method because the streaming MP4 session consumes float frames.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

RANK_LOCAL_DECODE_ENV = "VLLM_OMNI_VAE_RANK_LOCAL_DECODE"

_TRUTHY = {"1", "true", "yes", "on"}

_UINT8_GATHER_FALLBACK_WARNED = False


def rank_local_decode_requested() -> bool:
    """Whether ``VLLM_OMNI_VAE_RANK_LOCAL_DECODE`` asks for the prototype."""

    return os.environ.get(RANK_LOCAL_DECODE_ENV, "").strip().lower() in _TRUTHY


def rank_local_decode_installed(vae) -> bool:
    return bool(getattr(vae, "_rank_local_decode_installed", False))


def install_rank_local_tiled_decode(vae) -> bool:
    """Bind the rank-local tiled decoder on ``vae``'s checkpoint instance.

    Returns True when the rank-local path is active for this decode. No-op
    (and False) unless the environment switch is set and the checkpoint still
    exposes the stock ``tiled_decode``. Binding is per instance because the
    checkpoint class is remote code shared by every component loaded from the
    same checkpoint.
    """

    if rank_local_decode_installed(vae):
        return True
    if not rank_local_decode_requested():
        return False
    model = getattr(vae, "model", None)
    if model is None or not callable(getattr(model, "tiled_decode", None)):
        return False
    if not callable(getattr(model, "decode", None)) or not callable(
        getattr(model, "split_tiles", None)
    ):
        return False

    vae._checkpoint_tiled_decode = model.tiled_decode
    vae._checkpoint_blend = model.blend
    model._checkpoint_blend = model.blend
    vae._rank_local_decode_installed = True

    def _tiled_decode(z: torch.Tensor) -> torch.Tensor:
        return _rank_local_tiled_decode(vae, z)

    def _blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        return _uint8_aware_blend(model, a, b, blend_extent, dim)

    model.tiled_decode = _tiled_decode
    model.blend = _blend
    logger.info(
        "[H3_VAE_RANK_LOCAL] rank-local uint8 tiled decode installed "
        "(parallel_size=%s, tile_size=%s, tile_overlap_min=%s)",
        getattr(vae, "parallel_size", 1),
        getattr(model, "decoder_tile_size", None),
        getattr(model, "decoder_tile_overlap_min", None),
    )
    return True


def uninstall_rank_local_tiled_decode(vae) -> bool:
    """Restore the checkpoint methods (used by the float-frames chunk path)."""

    if not rank_local_decode_installed(vae):
        return False
    model = getattr(vae, "model", None)
    original_tiled = getattr(vae, "_checkpoint_tiled_decode", None)
    original_blend = getattr(vae, "_checkpoint_blend", None)
    if model is not None and original_tiled is not None:
        model.tiled_decode = original_tiled
    if model is not None and original_blend is not None:
        model.blend = original_blend
    vae._rank_local_decode_installed = False
    vae._checkpoint_tiled_decode = None
    vae._checkpoint_blend = None
    logger.info("[H3_VAE_RANK_LOCAL] restored the checkpoint tiled decode for the chunk path")
    return True


def _device_mem_mib() -> tuple[float, float]:
    """``(allocated, max_allocated)`` MiB on the active accelerator, if any."""

    for namespace in ("npu", "cuda", "xpu"):
        module = getattr(torch, namespace, None)
        if module is None:
            continue
        try:
            return (
                float(module.memory_allocated()) / (1024.0 * 1024.0),
                float(module.max_memory_allocated()) / (1024.0 * 1024.0),
            )
        except Exception:  # noqa: BLE001 - diagnostics must never fail a decode
            continue
    return (0.0, 0.0)


def _quantize_tile(vae, tile: torch.Tensor) -> torch.Tensor:
    """Revert and quantize one decoded tile to a contiguous uint8 tensor."""

    # Same op order as the complete path: the checkpoint reverts in the decode
    # dtype, and _prepare_minimax_h3_video_output widens to fp32 for the
    # clamp/scale/round quantizer.
    work = vae._revert_decoded_inplace(tile)
    work = work.float().clamp_(0.0, 1.0).mul_(255.0).round_()
    return work.to(torch.uint8)


def _uint8_aware_blend(
    model,
    a: torch.Tensor,
    b: torch.Tensor,
    blend_extent: int,
    dim: int,
) -> torch.Tensor:
    """``model.blend`` that stays in uint8 when both inputs are uint8."""

    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        return model._checkpoint_blend(a, b, blend_extent, dim)

    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    if blend_extent <= 0:
        return b

    positions = torch.arange(blend_extent, device=b.device, dtype=torch.float32)
    weight_a = (1.0 - positions / blend_extent).to(torch.float32)
    weight_b = (positions / blend_extent).to(torch.float32)
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    weight_a = weight_a.view(shape)
    weight_b = weight_b.view(shape)

    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-blend_extent, None)
    a_overlap = a[tuple(slice_a)].to(torch.float32)

    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, blend_extent)

    blended = (a_overlap * weight_a + b[tuple(slice_b)].to(torch.float32) * weight_b)
    blended = blended.clamp_(0.0, 255.0).round_().to(torch.uint8)

    if blend_extent < b.shape[dim]:
        slice_b_rest = [slice(None)] * b.ndim
        slice_b_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
    return blended


def _gather_uint8_tiles(
    vae,
    local_tiles: list[tuple[int, torch.Tensor]],
    num_tiles: int,
    sp_rank: int,
    sp_size: int,
) -> list[torch.Tensor]:
    """Gather uint8 tiles into global tile order.

    Ownership is round-robin, so every rank knows every other rank's task
    count from ``num_tiles`` and ``sp_size``: the leading dimension is padded
    to ``max_tasks`` and one ``all_gather_into_tensor`` fills a single
    grow-only uint8 workspace. Every rank ends up with all tiles (uint8), so
    the downstream blend/trim code stays symmetric across ranks.
    """

    state = vae._native_parallel_state()
    group = state.get("sp_process_group")

    if sp_size <= 1 or group is None:
        results: list[torch.Tensor | None] = [None] * num_tiles
        for index, tile in local_tiles:
            results[index] = tile
        return results  # type: ignore[return-value]

    if not local_tiles:
        raise ValueError(f"Found empty tasks on sp rank {sp_rank}")

    max_tasks = -(-num_tiles // sp_size)
    if len(local_tiles) > max_tasks:
        raise ValueError(
            f"sp rank {sp_rank} holds {len(local_tiles)} tiles but round-robin "
            f"ownership of {num_tiles} tiles across {sp_size} ranks allows at most {max_tasks}"
        )

    first = local_tiles[0][1]
    if len(local_tiles) == max_tasks:
        stacked = torch.stack([tile for _, tile in local_tiles], dim=0)
    else:
        stacked = torch.empty((max_tasks, *first.shape), dtype=torch.uint8, device=first.device)
        for slot, (_, tile) in enumerate(local_tiles):
            stacked[slot].copy_(tile)
        stacked[len(local_tiles) :].zero_()

    need_bytes = sp_size * stacked.numel()
    workspace = getattr(vae, "_rank_local_gather_workspace", None)
    if (
        workspace is None
        or workspace.device != stacked.device
        or workspace.numel() < need_bytes
    ):
        # Drop the previous block before allocating its replacement, so one
        # geometry never keeps two workspaces resident.
        vae._rank_local_gather_workspace = None
        workspace = torch.empty(need_bytes, dtype=torch.uint8, device=stacked.device)
        vae._rank_local_gather_workspace = workspace
        vae._rank_local_gather_reused = False
    else:
        vae._rank_local_gather_reused = True

    buffer = workspace[:need_bytes].view(sp_size, *stacked.shape)
    vae._rank_local_gather_mode = "uint8"
    try:
        dist.all_gather_into_tensor(buffer, stacked, group=group)
    except (RuntimeError, TypeError):
        # Not every HCCL/NCCL build accepts a uint8 payload. Fall back to a
        # 4-byte view of the same bytes: pad the stack to a multiple of four
        # bytes, gather as int32, and view the received block back as uint8.
        global _UINT8_GATHER_FALLBACK_WARNED
        if not _UINT8_GATHER_FALLBACK_WARNED:
            _UINT8_GATHER_FALLBACK_WARNED = True
            logger.warning(
                "[H3_VAE_RANK_LOCAL] uint8 all_gather_into_tensor rejected; "
                "falling back to an int32 view of the same bytes"
            )
        flat = stacked.reshape(-1)
        pad = (-flat.numel()) % 4
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        as_int = flat.view(torch.int32)
        need_words = sp_size * as_int.numel()
        words = getattr(vae, "_rank_local_gather_words", None)
        if words is None or words.numel() < need_words:
            vae._rank_local_gather_words = None
            words = torch.empty(need_words, dtype=torch.int32, device=stacked.device)
            vae._rank_local_gather_words = words
        word_view = words[:need_words].view(sp_size, as_int.numel())
        dist.all_gather_into_tensor(word_view, as_int, group=group)
        # Padding to a 4-byte multiple can add up to three trailing bytes, so
        # slice the received block back to the exact tile payload before the
        # reshape.
        flat_bytes = word_view.view(torch.uint8)[:, : stacked.numel()]
        buffer = flat_bytes.reshape(sp_size, *stacked.shape)
        vae._rank_local_gather_mode = "int32_fallback"

    results = [None] * num_tiles
    for rank in range(sp_size):
        rank_tasks = -(-(num_tiles - rank) // sp_size)
        for k in range(rank_tasks):
            results[k * sp_size + rank] = buffer[rank][k]
    if any(tile is None for tile in results):
        raise RuntimeError(
            f"rank-local tile gather left gaps on sp rank {sp_rank}: "
            f"{sum(tile is None for tile in results)}/{num_tiles} tiles missing"
        )
    return results  # type: ignore[return-value]


def _rank_local_tiled_decode(vae, z: torch.Tensor) -> torch.Tensor:
    """Mirror of ``AutoencoderKL.tiled_decode`` with uint8 tile transport."""

    model = vae.model
    ratio = int(model.vae_ratio)
    parallel = bool(getattr(model, "parallel_tiling", False))

    state = vae._native_parallel_state()
    sp_rank = int(state.get("sp_rank") or 0) if parallel else 0
    sp_size = int(state.get("sp_size") or 1) if parallel else 1

    height = int(z.shape[-2]) * ratio
    width = int(z.shape[-1]) * ratio
    y_idx, y_len, y_overlap = model.split_tiles(height, True)
    x_idx, x_len, x_overlap = model.split_tiles(width, True)

    i_max, j_max = len(y_idx), len(x_idx)
    num_tiles = i_max * j_max
    local_indices = list(range(sp_rank, num_tiles, sp_size))

    before_alloc, before_peak = _device_mem_mib()
    local_tiles: list[tuple[int, torch.Tensor]] = []
    tile_peak = before_peak
    for index in local_indices:
        i, j = divmod(index, j_max)
        y0 = y_idx[i] // ratio
        y1 = y0 + y_len[i] // ratio
        x0 = x_idx[j] // ratio
        x1 = x0 + x_len[j] // ratio
        decoded = model.decode(z[..., y0:y1, x0:x1])
        local_tiles.append((index, _quantize_tile(vae, decoded)))
        del decoded
        tile_peak = max(tile_peak, _device_mem_mib()[1])

    gather_peak = tile_peak
    all_decoded = _gather_uint8_tiles(vae, local_tiles, num_tiles, sp_rank, sp_size)
    gather_peak = max(gather_peak, _device_mem_mib()[1])
    del local_tiles

    rows: list[list[torch.Tensor | None]] = [[None] * j_max for _ in range(i_max)]
    for index, tile in enumerate(all_decoded):
        rows[index // j_max][index % j_max] = tile

    result_rows: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        result_row: list[torch.Tensor] = []
        for j, tile in enumerate(row):
            assert tile is not None
            if i > 0:
                tile = vae.model.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)  # type: ignore[arg-type]
            if j > 0:
                tile = vae.model.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)  # type: ignore[arg-type]
            if i < len(rows) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    dec = torch.cat(result_rows, dim=-2)

    after_alloc, after_peak = _device_mem_mib()
    logger.info(
        "[H3_VAE_RANK_LOCAL] tiles=%d local=%d/%d %s collective=%s bytes=%.2fMiB "
        "alloc_mib=%.1f->%.1f peak_gain_mib=%.1f tile_peak_mib=%.1f gather_peak_mib=%.1f",
        num_tiles,
        len(local_indices),
        sp_size,
        "gather=reuse" if getattr(vae, "_rank_local_gather_reused", False) else "gather=alloc",
        getattr(vae, "_rank_local_gather_mode", "local"),
        float(dec.numel()) / (1024.0 * 1024.0),
        before_alloc,
        after_alloc,
        after_peak - before_peak,
        tile_peak - before_peak,
        gather_peak - before_peak,
    )
    return dec
