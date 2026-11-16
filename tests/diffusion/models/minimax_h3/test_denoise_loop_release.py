"""t120: the denoise loop must not carry per-step temporaries into the next step.

The loop keeps four families of per-step buffers alive by default because the
``for`` body rebinds them only *after* the next ``model(**fk)`` call returns:

* ``v_video`` / ``v_audio``       - raw DiT velocity rows (decoder dtype)
* ``mv_video_t`` / ``mv_audio_t`` - FP32 gather at the update rows
* ``x0_video`` / ``x0_audio``     - flow-to-x0 prediction
* ``new_target`` / ``new_audio``  - solver output copied into the packed rows

For a 4K/30s request those four families add up to ~2.3 GB of device memory
that would stay resident across the next DiT forward.  t120 releases each of
them as soon as its last consumer ran.  This test drives the *real* loop with
stubbed dependencies and asserts, at every subsequent forward, that all of the
previous iteration's buffers are already collectable.
"""

from __future__ import annotations

import gc
import importlib
import weakref
from dataclasses import dataclass, field

import pytest
import torch

pytestmark = pytest.mark.cpu

_ROWS = 8
_VIDEO_WIDTH = 96
_AUDIO_WIDTH = 32
_SIGMAS = [1.0, 0.5, 0.0]


def _loop_module():
    return importlib.import_module("vllm_omni.diffusion.models.minimax_h3.denoise_loop")


class _Recorder:
    """Collects weak references to every per-step buffer the loop creates."""

    def __init__(self) -> None:
        self.pending: list[tuple[str, weakref.ReferenceType]] = []
        self.alive_at_forward: list[dict[str, bool]] = []
        self.alive_in_solver: list[dict[str, bool]] = []

    def watch(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        self.pending.append((name, weakref.ref(tensor)))
        return tensor

    def alive(self, prefix: str | None = None) -> dict[str, bool]:
        gc.collect()
        alive = {name: ref() is not None for name, ref in self.pending if ref() is not None}
        if prefix is not None:
            alive = {name: value for name, value in alive.items() if name.startswith(prefix)}
        return alive

    def check_before_forward(self) -> None:
        alive = self.alive()
        self.alive_at_forward.append(alive)
        assert not alive, f"previous step's buffers are still alive: {sorted(alive)}"

    def survivors(self) -> list[str]:
        gc.collect()
        return sorted(name for name, ref in self.pending if ref() is not None)


@dataclass
class _FakePositive:
    update_mask_dev: torch.Tensor
    audio_update_mask_dev: torch.Tensor
    locked_audio_rows: torch.Tensor | None = None
    forward_kwargs_calls: int = 0

    def forward_kwargs(self, **kwargs):
        self.forward_kwargs_calls += 1
        return dict(kwargs)

    def prepare_rope_table(self, model) -> None:  # pragma: no cover - NPU only
        raise AssertionError("CPU path must not prepare a rope table")


class _FakeEdit:
    """Stands in for MiniMaxH3LatentEdit (``video_edit`` / ``audio_edit``)."""

    def __init__(self, recorder: _Recorder, tag: str) -> None:
        self.recorder = recorder
        self.tag = tag
        self.calls = 0

    def to(self, *, device=None, dtype=None):
        return self

    def x0(self, state: torch.Tensor, velocity: torch.Tensor, t) -> torch.Tensor:
        self.calls += 1
        self.recorder.watch(f"{self.tag}.velocity", velocity)
        return state * 0.25 + velocity * 0.75


class _FakeSolver:
    def __init__(self, recorder: _Recorder, tag: str) -> None:
        self.recorder = recorder
        self.tag = tag
        self.step_index = 0
        self.calls = 0

    def step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        assert step == self.step_index, "the loop must consume solver steps in order"
        self.calls += 1
        self.step_index = step + 1
        self.recorder.watch(f"{self.tag}.x0", denoised)
        out = torch.empty_like(state)
        out.copy_(state * 0.5 + denoised * 0.5)
        return self.recorder.watch(f"{self.tag}.new_target", out)


def _install_stubs(monkeypatch, recorder, solvers, *, use_edit: bool):
    mod = _loop_module()

    video_rows = torch.randn(_ROWS, _VIDEO_WIDTH, dtype=torch.float32)
    audio_rows = torch.randn(_ROWS, _AUDIO_WIDTH, dtype=torch.float32)
    positive = _FakePositive(
        update_mask_dev=torch.ones(_ROWS, dtype=torch.bool),
        audio_update_mask_dev=torch.ones(_ROWS, dtype=torch.bool),
    )

    monkeypatch.setattr(
        mod,
        "minimax_h3_prepare_denoise_rows",
        lambda **kwargs: (video_rows.clone(), audio_rows.clone(), None, None),
    )
    monkeypatch.setattr(
        mod,
        "minimax_h3_prepare_edit_rows",
        lambda rows, mask, edit, t, cond_t, *, sigma=None: (rows, None),
    )

    def fake_rf_v_to_x0(state: torch.Tensor, velocity: torch.Tensor, t) -> torch.Tensor:
        recorder.watch("mv", velocity)
        return state + velocity * 0.1 * float(t.item())

    monkeypatch.setattr(mod, "minimax_h3_rf_v_to_x0", fake_rf_v_to_x0)
    monkeypatch.setattr(mod, "minimax_h3_publish_denoise_progress", lambda *a, **k: None)
    monkeypatch.setattr(mod, "check_request_cancellation", lambda *a, **k: None)
    monkeypatch.setattr(mod.phase_probe, "reset_peaks", lambda: None)
    monkeypatch.setattr(mod.phase_probe, "set_step", lambda step: None)

    queue = list(solvers)
    monkeypatch.setattr(mod, "create_h3_sample_solver", lambda name, sigmas: queue.pop(0))

    def model(**kwargs):
        recorder.check_before_forward()
        v = recorder.watch("v_video", torch.full_like(video_rows, 0.5))
        a = recorder.watch("v_audio", torch.full_like(audio_rows, -0.5))
        return v, a

    video_edit = _FakeEdit(recorder, "video_edit") if use_edit else None
    audio_edit = _FakeEdit(recorder, "audio_edit") if use_edit else None
    return mod, model, positive, video_edit, audio_edit


def _run(mod, model, positive, video_edit, audio_edit):
    return mod.minimax_h3_denoise_loop(
        model=model,
        positive=positive,
        initial_video_rows=torch.randn(_ROWS, _VIDEO_WIDTH),
        initial_audio_rows=torch.randn(_ROWS, _AUDIO_WIDTH),
        keyframe_cond_rows=None,
        video_edit=video_edit,
        audio_edit=audio_edit,
        sigmas_video=list(_SIGMAS),
        sigmas_audio=list(_SIGMAS),
        device=torch.device("cpu"),
        sampler="euler",
    )


@pytest.mark.parametrize("use_edit", [False, True], ids=["plain", "edit_rows"])
def test_denoise_loop_releases_every_step_buffer(monkeypatch, use_edit):
    recorder = _Recorder()
    solvers = [_FakeSolver(recorder, "video"), _FakeSolver(recorder, "audio")]
    mod, model, positive, video_edit, audio_edit = _install_stubs(
        monkeypatch, recorder, solvers, use_edit=use_edit
    )

    out_video, out_audio = _run(mod, model, positive, video_edit, audio_edit)

    assert out_video.shape == (_ROWS, _VIDEO_WIDTH)
    assert out_audio.shape == (_ROWS, _AUDIO_WIDTH)
    assert torch.isfinite(out_video).all() and torch.isfinite(out_audio).all()
    assert positive.forward_kwargs_calls == len(_SIGMAS) - 1
    assert solvers[0].calls == len(_SIGMAS) - 1
    assert solvers[1].calls == len(_SIGMAS) - 1
    # Forward N+1 must see forward N's buffers gone; all four families are
    # registered through the stubs above.
    assert len(recorder.alive_at_forward) == len(_SIGMAS) - 1
    assert all(not entry for entry in recorder.alive_at_forward)
    watched = {name for name, _ in recorder.pending}
    assert {"v_video", "v_audio", "video.x0", "audio.x0", "video.new_target", "audio.new_target"} <= watched
    if use_edit:
        assert {"video_edit.velocity", "audio_edit.velocity"} <= watched
        assert video_edit.calls == len(_SIGMAS) - 1
        assert audio_edit.calls == len(_SIGMAS) - 1
    else:
        assert "mv" in watched
    # Nothing survives the loop either: the buffers are not returned/stashed.
    assert recorder.survivors() == []


def test_velocity_and_gather_are_gone_before_the_solver_runs(monkeypatch):
    """``solver.step`` only needs x0, so the gathers must be dropped before it."""
    recorder = _Recorder()
    solvers = [_FakeSolver(recorder, "video"), _FakeSolver(recorder, "audio")]

    for solver in solvers:
        inner = solver.step

        def step(state, denoised, step_index, _inner=inner, _solver=solver):
            alive = recorder.alive()
            recorder.alive_in_solver.append(alive)
            return _inner(state, denoised, step_index)

        solver.step = step  # type: ignore[method-assign]

    mod, model, positive, _, _ = _install_stubs(monkeypatch, recorder, solvers, use_edit=False)
    _run(mod, model, positive, None, None)

    assert recorder.alive_in_solver, "expected the solvers to run"
    for entry in recorder.alive_in_solver:
        assert not entry, f"solver ran while per-step buffers were still alive: {sorted(entry)}"
