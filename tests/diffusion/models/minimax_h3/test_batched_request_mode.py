# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the ``H3_BATCHED_REQUEST_MODE`` request-batch prototype.

The prototype packs several requests into one DiT forward per denoise step. These
tests cover the three things that can break silently: the env switch, the packed
attention fallback that keeps documents isolated on backends without a
multi-document varlen kernel, and the per-step split/broadcast loop in
``MiniMaxH3Pipeline._forward_batched_requests``.
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as MT
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as P

pytestmark = pytest.mark.cpu

HEAD_DIM = 8
NUM_HEADS = 3


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(MT._BATCHED_REQUEST_MODE_ENV, raising=False)
    yield


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        (" on ", True),
    ],
)
def test_env_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(MT._BATCHED_REQUEST_MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(MT._BATCHED_REQUEST_MODE_ENV, value)
    assert MT.batched_request_mode_enabled() is expected


class _Backend:
    """Minimal backend double: no multi-document varlen kernel."""

    def __init__(self, name: str = "fake_npu") -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def supports_multi_doc_packed_varlen(self) -> bool:
        return False

    def supports_packed_mask_free(self) -> bool:
        return True

    def supports_prefix_kv_slicing(self) -> bool:
        return True


def _packed_layer():
    """A ``MiniMaxH3Attention`` with only the fields the packed path touches."""
    layer = object.__new__(MT.MiniMaxH3Attention)
    layer.num_heads = NUM_HEADS
    layer.head_dim = HEAD_DIM
    layer.softmax_scale = HEAD_DIM**-0.5
    layer.vsa_sparsity = None
    layer.attention = types.SimpleNamespace(attn_backend=_Backend(), use_ring=False, skip_sequence_parallel=True)
    return layer


def _reference_sdpa(q, k, v, cu_seqlens):
    """Attend each document on its own, the way the island must."""
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        seg_q = q[start:stop].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:stop].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:stop].transpose(0, 1).unsqueeze(0)
        seg_out = F.scaled_dot_product_attention(seg_q, seg_k, seg_v, scale=HEAD_DIM**-0.5, is_causal=False)
        out[start:stop] = seg_out.squeeze(0).transpose(0, 1)
    return out


def test_packed_attention_rejects_batches_without_the_switch():
    layer = _packed_layer()
    q = torch.randn(6, NUM_HEADS, HEAD_DIM)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    with pytest.raises(ValueError, match="does not isolate multi-document packed cu_seqlens"):
        layer._run_packed_attention(q, q, q, cu_seqlens=cu, max_seqlen=3, packed_total=6, num_requests=2)


def test_packed_attention_segments_documents_with_the_switch(monkeypatch):
    monkeypatch.setenv(MT._BATCHED_REQUEST_MODE_ENV, "1")
    layer = _packed_layer()
    torch.manual_seed(0)
    q = torch.randn(7, NUM_HEADS, HEAD_DIM)
    k = torch.randn(7, NUM_HEADS, HEAD_DIM)
    v = torch.randn(7, NUM_HEADS, HEAD_DIM)
    # Two documents of different length: a padding-mask rebuild would attend
    # across the boundary, the segment island must not.
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    out = layer._run_packed_attention(q, k, v, cu_seqlens=cu, max_seqlen=4, packed_total=7, num_requests=2)
    assert out.shape == (7, NUM_HEADS * HEAD_DIM)
    expected = _reference_sdpa(q, k, v, cu)
    assert torch.allclose(out, expected.reshape(7, NUM_HEADS * HEAD_DIM), atol=1e-6)

    # Documents must not leak into each other. Perturb the values (not the keys:
    # a uniform key shift only adds a constant to every logit, which softmax
    # cannot see) and check the untouched document stays bit-identical.
    v2 = v.clone()
    v2[3:] += 1.0
    out2 = layer._run_packed_attention(q, k, v2, cu_seqlens=cu, max_seqlen=4, packed_total=7, num_requests=2)
    assert torch.allclose(out2[:3], out[:3], atol=1e-6)
    assert not torch.allclose(out2[3:], out[3:], atol=1e-6)

    v3 = v.clone()
    v3[:3] += 1.0
    out3 = layer._run_packed_attention(q, k, v3, cu_seqlens=cu, max_seqlen=4, packed_total=7, num_requests=2)
    assert torch.allclose(out3[3:], out[3:], atol=1e-6)
    assert not torch.allclose(out3[:3], out[:3], atol=1e-6)


def _states(rows=(4, 3), steps=2):
    from vllm_omni.diffusion.worker.utils import StepRequestState

    states = []
    for index, row_count in enumerate(rows):
        state = StepRequestState(request_id=f"batch-{index}", sampling=object())
        state.latents = torch.zeros(row_count, 96)
        state.timesteps = torch.linspace(1.0, 0.0, steps + 1)[:-1]
        state.extra = {
            P._STEP_TRANSFORMER: _TRANSFORMER,
            P._STEP_SIGMAS_VIDEO: [1.0 - step / steps for step in range(steps + 1)],
            P._STEP_SHAPE: {"latent_t": 1, "latent_h": 2, "latent_w": 3, "audio_t": 4},
        }
        states.append(state)
    return states


_TRANSFORMER = object()


class _FakePipeline:
    """Carries only what ``_forward_batched_requests`` calls on ``self``."""

    # The homogeneity check and the batch view are pure helpers; drive the real
    # implementations so the test cannot drift from the production contract.
    _validate_batched_states = staticmethod(P.MiniMaxH3Pipeline._validate_batched_states)
    _batched_input_batch = staticmethod(P.MiniMaxH3Pipeline._batched_input_batch)

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.states: list[object] = []
        self.transformer = _TRANSFORMER

    def _resident_dit_layers_on_device(self, *, enabled: bool = True):
        self.calls.append(("resident", enabled))
        return contextlib.nullcontext()

    def prepare_encode(self, state):
        """Stand in for the real ``prepare_encode``: same shapes every request."""
        self.calls.append(("prepare_encode", state.request_id))
        rows = 4 if not self.states else 3
        state.latents = torch.zeros(rows, 96)
        state.timesteps = torch.tensor([1.0, 0.5])
        state.extra = {
            P._STEP_TRANSFORMER: self.transformer,
            P._STEP_SIGMAS_VIDEO: [1.0, 0.5, 0.0],
            P._STEP_SHAPE: {"latent_t": 1, "latent_h": 2, "latent_w": 3, "audio_t": 4},
        }
        self.states.append(state)
        return state

    def denoise_step(self, input_batch, *, states=None):
        assert states is not None and input_batch.num_reqs == len(states)
        self.calls.append(("denoise_step", tuple(int(state.latents.shape[0]) for state in states)))
        return torch.cat(
            [torch.full((int(state.latents.shape[0]), 2), float(state.step_index)) for state in states]
        )

    def step_scheduler(self, state, prediction):
        self.calls.append(("step_scheduler", state.request_id, int(prediction.shape[0])))
        state.step_index += 1

    def post_decode(self, state):
        self.calls.append(("post_decode", state.request_id))
        return {"decoded": state.request_id}


def test_forward_batched_requests_runs_one_step_loop():
    pipeline = _FakePipeline()
    request = types.SimpleNamespace(
        prompts=["a red fox", "a blue fox"],
        sampling_params_list=[object(), object()],
    )
    outputs = P.MiniMaxH3Pipeline._forward_batched_requests(pipeline, request)

    assert [output["decoded"] for output in outputs] == ["batch-0", "batch-1"]
    names = [call[0] for call in pipeline.calls]
    # Encode both requests first, then hold one resident DiT window around the
    # whole step loop (a DLO run must not re-materialize the DiT per step).
    assert names[:2] == ["prepare_encode", "prepare_encode"]
    assert names.count("resident") == 1 and names.index("resident") == 2
    assert ("resident", True) in pipeline.calls
    # Two denoise steps (len(sigmas_video) - 1); one scheduler call per request
    # per step, each on its own row slice (4 then 3 rows).
    assert names.count("denoise_step") == 2
    assert [call[2] for call in pipeline.calls if call[0] == "step_scheduler"] == [4, 3, 4, 3]
    assert names.count("post_decode") == 2


def test_batched_input_batch_carries_the_step_bookkeeping():
    states = _states()
    batch = P.MiniMaxH3Pipeline._batched_input_batch(states)
    assert list(batch.request_ids) == ["batch-0", "batch-1"]
    assert batch.num_reqs == batch.num_reqs_after_padding == 2
    assert batch.latents.shape[0] == 7
    assert batch.timesteps.tolist() == [1.0, 1.0]


def test_validate_batched_states_accepts_homogeneous_requests():
    P.MiniMaxH3Pipeline._validate_batched_states(_states())


def test_validate_batched_states_rejects_mixed_geometry():
    states = _states()
    states[1].extra[P._STEP_SHAPE] = {"latent_t": 1, "latent_h": 2, "latent_w": 9, "audio_t": 4}
    with pytest.raises(Exception, match="identical latent geometry"):
        P.MiniMaxH3Pipeline._validate_batched_states(states)


def test_validate_batched_states_rejects_mixed_dit():
    states = _states()
    states[1].extra[P._STEP_TRANSFORMER] = object()
    with pytest.raises(Exception, match="same task-specific DiT"):
        P.MiniMaxH3Pipeline._validate_batched_states(states)


def test_validate_batched_states_rejects_mixed_steps():
    states = _states()
    states[1].extra[P._STEP_SIGMAS_VIDEO] = [1.0, 0.5, 0.25, 0.0]
    with pytest.raises(Exception, match="same number of denoise steps"):
        P.MiniMaxH3Pipeline._validate_batched_states(states)


def test_validate_batched_states_rejects_edit_rows():
    states = _states()
    states[1].extra[P._STEP_VIDEO_EDIT] = object()
    with pytest.raises(Exception, match="video/audio edit rows"):
        P.MiniMaxH3Pipeline._validate_batched_states(states)


def test_forward_without_the_switch_keeps_the_single_request_error(monkeypatch):
    monkeypatch.delenv(MT._BATCHED_REQUEST_MODE_ENV, raising=False)
    request = types.SimpleNamespace(prompts=["a", "b"], sampling_params_list=[object(), object()])
    with pytest.raises(Exception, match="one request at a time"):
        P.MiniMaxH3Pipeline.forward(object.__new__(P.MiniMaxH3Pipeline), request)


def test_forward_dispatches_to_the_batch_path_when_enabled(monkeypatch):
    monkeypatch.setenv(MT._BATCHED_REQUEST_MODE_ENV, "1")
    seen = {}

    def fake_batched(self, request):
        seen["prompts"] = list(request.prompts)
        return ["batched"]

    monkeypatch.setattr(P.MiniMaxH3Pipeline, "_forward_batched_requests", fake_batched)
    request = types.SimpleNamespace(prompts=["a", "b"], sampling_params_list=[object(), object()])
    assert P.MiniMaxH3Pipeline.forward(object.__new__(P.MiniMaxH3Pipeline), request) == ["batched"]
    assert seen["prompts"] == ["a", "b"]


def test_class_attribute_follows_the_import_time_env():
    # The engine reads this class attribute once at construction time, so the
    # module-level default must stay False without the env variable.
    assert P.MiniMaxH3Pipeline.supports_request_batch is False
