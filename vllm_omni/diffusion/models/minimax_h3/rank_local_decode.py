# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank-local tile assembly for the MiniMax-H3 video VAE decode.

The checkpoint's tiled decoder (``AutoencoderKL.tiled_decode``) decodes the
tiles each rank owns, then gathers **every rank's tiles to every rank**
(``_all_gather_tiled_results``), blends them, and concatenates a full-frame
video. The vLLM-Omni adapter already consumes that clip as uint8
(``vae_temporal._decode_temporal_streaming_uint8`` reverts and quantizes each
temporal part while streaming into a uint8 buffer), so the only buffers this
module can remove are the intermediate float buffers of the tile assembly.

With ``VLLM_OMNI_VAE_RANK_LOCAL_DECODE=1`` the adapter replaces the
checkpoint's ``tiled_decode`` (instance binding, same pattern as the XPU
persistent-gather override in ``vae.py``) with the implementation below:

* every rank decodes only its own round-robin tiles;
* the tiles travel through a single grow-only workspace instead of a freshly
  allocated stacked send buffer plus receive buffer per call;
* the frame is assembled tile by tile with exactly the checkpoint's blend and
  trim order.

Numerics are intentionally identical to the checkpoint: tiles stay in the
decoder dtype, the blend weights are computed in that dtype, and the returned
clip is the same float tensor the stock path returns. The revert and the
quantizer stay where they are -- inside ``write_part`` of the streaming
temporal decoder -- because running them here would apply them twice.

The switch is one-way per process: the first decode installs the override and
every later decode reuses it. ``decode_with_chunks`` restores the checkpoint
method because the chunked MP4 session slices temporally on its own.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

RANK_LOCAL_DECODE_ENV = "VLLM_OMNI_VAE_RANK_LOCAL_DECODE"

_TRUTHY = {"1", "true", "yes", "on"}

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
    vae._rank_local_decode_installed = True

    def _tiled_decode(z: torch.Tensor) -> torch.Tensor:
        return _rank_local_tiled_decode(vae, z)

    model.tiled_decode = _tiled_decode
    logger.info(
        "[H3_VAE_RANK_LOCAL] rank-local tiled decode installed "
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
    if model is not None and original_tiled is not None:
        model.tiled_decode = original_tiled
    vae._rank_local_decode_installed = False
    vae._checkpoint_tiled_decode = None
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


def _gather_tiles(
    vae,
    local_tiles: list[tuple[int, torch.Tensor]],
    num_tiles: int,
    sp_rank: int,
    sp_size: int,
) -> list[torch.Tensor]:
    """Gather tiles into global tile order through one grow-only workspace.

    Ownership is round-robin, so every rank knows every other rank's task
    count from ``num_tiles`` and ``sp_size``: the leading dimension is padded
    to ``max_tasks`` and a single ``all_gather_into_tensor`` fills a workspace
    that is reused across clips instead of a fresh stacked send buffer plus
    receive buffer per call. Every rank ends up with all tiles, which the
    symmetric blend below needs.
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
    dtype = first.dtype
    if len(local_tiles) == max_tasks:
        stacked = torch.stack([tile for _, tile in local_tiles], dim=0)
    else:
        stacked = torch.zeros((max_tasks, *first.shape), dtype=dtype, device=first.device)
        for slot, (_, tile) in enumerate(local_tiles):
            stacked[slot].copy_(tile)

    need_elements = sp_size * stacked.numel()
    workspace = getattr(vae, "_rank_local_gather_workspace", None)
    if (
        workspace is None
        or workspace.device != stacked.device
        or workspace.dtype != dtype
        or workspace.numel() < need_elements
    ):
        # Drop the previous block before allocating its replacement, so one
        # geometry never keeps two workspaces resident.
        vae._rank_local_gather_workspace = None
        workspace = torch.empty(need_elements, dtype=dtype, device=stacked.device)
        vae._rank_local_gather_workspace = workspace
        vae._rank_local_gather_reused = False
    else:
        vae._rank_local_gather_reused = True

    buffer = workspace[:need_elements].view(sp_size, *stacked.shape)
    vae._rank_local_gather_mode = str(dtype).replace("torch.", "")
    dist.all_gather_into_tensor(buffer, stacked, group=group)

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
        local_tiles.append((index, decoded))
        del decoded
        tile_peak = max(tile_peak, _device_mem_mib()[1])

    gather_peak = tile_peak
    all_decoded = _gather_tiles(vae, local_tiles, num_tiles, sp_rank, sp_size)
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
        "[H3_VAE_RANK_LOCAL] tiles=%d local=%d/%d %s transport=%s bytes=%.2fMiB "
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
