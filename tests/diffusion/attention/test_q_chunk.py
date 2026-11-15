# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for exact Q-row chunking (H3_ATTN_Q_CHUNKS)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.q_chunk import (
    ALLOWED_Q_CHUNKS,
    ENV_Q_CHUNKS,
    build_chunk_metadata,
    chunk_bounds,
    plan_q_chunking,
    q_chunk_count,
    reset_q_chunk_cache,
    run_chunked_attention,
)

pytestmark = pytest.mark.cpu

USED_Q = 65
USED_K = 129
PAD = 48
TOTAL_Q = USED_Q + PAD
TOTAL_K = USED_K + PAD
HEADS = 7
DIM = 128
REL_L2_LIMIT = 5e-3


def _cu(used: int, total: int) -> torch.Tensor:
    return torch.tensor([0, used, total], dtype=torch.int32)


def canonical_metadata(used_q: int = USED_Q, used_k: int = USED_K) -> AttentionMetadata:
    """Mirror MiniMax-H3's packed [real, pad] metadata (transformer:600-690)."""
    cu_q = _cu(used_q, TOTAL_Q)
    cu_k = _cu(used_k, TOTAL_K)
    return AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=used_q,
            kv_length=used_k,
            cu_seqlens_q=cu_q[:2],
            cu_seqlens_k=cu_k[:2],
        ),
        extra={
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": used_q,
            "max_seqlen_k": used_k,
            "valid_kv_length": used_k,
            "npu_attn_varlen": True,
            "laser_input_scale": 8.0,
        },
    )


def sdpa_kernel(query, key, value, metadata):
    """Stand-in for the NPU kernel: exact softmax over the full valid KV."""
    del metadata  # the kernel reads the packed metadata only to slice KV

    def bhsd(t):
        return t.transpose(1, 2).float()

    out = F.scaled_dot_product_attention(
        bhsd(query), bhsd(key), bhsd(value), is_causal=False
    )
    return out.transpose(1, 2).to(query.dtype)


def softmax_reference(query, key, value, used_q: int, used_k: int) -> torch.Tensor:
    """Straightforward float32 reference: every query row over all valid KV."""
    q = query[:, :used_q].transpose(1, 2).float()  # [1, H, used_q, D]
    k = key[:, :used_k].transpose(1, 2).float()
    v = value[:, :used_k].transpose(1, 2).float()
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DIM)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2)


def make_qkv(used_q: int = USED_Q, seed: int = 0, dtype=torch.bfloat16):
    """Packed Q/K/V with the [real, pad] layout MiniMax-H3 produces.

    ``used_q`` is the length of the real query document; the structural padding
    document therefore starts at row ``used_q`` (its "q start" is non-zero
    whenever ``used_q < TOTAL_Q``). K/V likewise carry a valid prefix of length
    ``USED_K`` followed by padding rows.
    """
    gen = torch.Generator().manual_seed(seed)
    scale = 1.0 / math.sqrt(DIM)
    query = torch.randn(1, TOTAL_Q, HEADS, DIM, generator=gen)
    key = torch.randn(1, TOTAL_K, HEADS, DIM, generator=gen)
    value = torch.randn(1, TOTAL_K, HEADS, DIM, generator=gen)
    return (
        (query * scale).to(dtype),
        (key * scale).to(dtype),
        (value * scale).to(dtype),
    )


def plan_for(chunks: int, used_q: int = USED_Q, metadata: AttentionMetadata | None = None):
    metadata = metadata if metadata is not None else canonical_metadata(used_q=used_q)
    query, key, _ = make_qkv(used_q)
    return plan_q_chunking(
        chunks=chunks,
        attn_metadata=metadata,
        query=query,
        key=key,
        backend_name="FLASH_ATTN",
        backend_supports_mask_free=True,
    )


# --------------------------------------------------------------------------- env


def test_env_default_is_one(monkeypatch):
    monkeypatch.delenv(ENV_Q_CHUNKS, raising=False)
    reset_q_chunk_cache()
    assert q_chunk_count() == 1


@pytest.mark.parametrize("value", ALLOWED_Q_CHUNKS)
def test_env_accepts_documented_values(monkeypatch, value):
    monkeypatch.setenv(ENV_Q_CHUNKS, str(value))
    reset_q_chunk_cache()
    assert q_chunk_count() == value


@pytest.mark.parametrize("raw", ["3", "8", "0", "-2", "two"])
def test_env_rejects_unsupported_values(monkeypatch, raw):
    monkeypatch.setenv(ENV_Q_CHUNKS, raw)
    reset_q_chunk_cache()
    with pytest.raises(ValueError, match=ENV_Q_CHUNKS):
        q_chunk_count()


def test_env_reads_changes_between_requests(monkeypatch):
    monkeypatch.setenv(ENV_Q_CHUNKS, "2")
    reset_q_chunk_cache()
    assert q_chunk_count() == 2
    monkeypatch.setenv(ENV_Q_CHUNKS, "6")
    assert q_chunk_count() == 6  # no restart needed


# ------------------------------------------------------------------------ bounds


@pytest.mark.parametrize(
    ("rows", "chunks", "expected"),
    [
        (65, 1, ((0, 65),)),
        (65, 2, ((0, 33), (33, 65))),
        (65, 4, ((0, 17), (17, 34), (34, 51), (51, 65))),
        (65, 6, ((0, 11), (11, 22), (22, 33), (33, 44), (44, 55), (55, 65))),
        (12, 6, ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12))),
        (4, 6, ((0, 1), (1, 2), (2, 3), (3, 4))),
        (1, 4, ((0, 1),)),
        (0, 4, ()),
    ],
)
def test_chunk_bounds(rows, chunks, expected):
    bounds = chunk_bounds(rows, chunks)
    assert bounds == expected
    assert all(end > start for start, end in bounds)
    assert sum(end - start for start, end in bounds) == rows


# ---------------------------------------------------------------------- planning


def test_plan_accepts_the_canonical_contract():
    plan = plan_for(4)
    assert (plan.used_q, plan.used_k, plan.total_q) == (USED_Q, USED_K, TOTAL_Q)
    assert plan.chunk_rows == 17
    assert [end - start for start, end in plan.bounds] == [17, 17, 17, 14]
    assert plan.describe()["rows_per_chunk"] == [17, 17, 17, 14]


def test_plan_accepts_a_short_real_document():
    """used_q=17 => the padding document starts at a non-zero row offset."""
    plan = plan_for(4, used_q=17)
    assert plan.used_q == 17
    assert plan.bounds == ((0, 5), (5, 10), (10, 15), (15, 17))


def test_plan_accepts_the_unpadded_single_document():
    plan = plan_for(6, used_q=TOTAL_Q)
    assert plan.used_q == TOTAL_Q


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: setattr(m, "attn_mask", torch.ones(1, TOTAL_Q, dtype=torch.bool)), "full_qk"),
        (lambda m: setattr(m, "full_attn_spans", [[(0, 4)]]), "full_attn_spans"),
        (lambda m: setattr(m, "joint_query", torch.zeros(1)), "joint"),
        (lambda m: m.extra.__setitem__("kv_cache_dtype", "fp8"), "quantized"),
        (lambda m: m.extra.__setitem__("npu_attn_varlen", False), "npu_attn_varlen"),
        (lambda m: m.extra.pop("cu_seqlens_q"), "missing packed keys"),
        (lambda m: m.extra.__setitem__("max_seqlen_q", 0), "out of range"),
        (lambda m: m.extra.__setitem__("max_seqlen_q", float(USED_Q)), "host ints"),
        (
            lambda m: m.extra.__setitem__(
                "cu_seqlens_q", torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
            ),
            "cu_seqlens must be canonical",
        ),
    ],
)
def test_plan_rejects_non_exact_contracts(mutate, match):
    metadata = canonical_metadata()
    mutate(metadata)
    with pytest.raises(NotImplementedError, match=match):
        plan_for(2, metadata=metadata)


@pytest.mark.parametrize(
    ("backend_name", "supports_mask_free", "match"),
    [("SDPA", True, "FLASH_ATTN"), ("FLASH_ATTN", False, "supports_packed_mask_free")],
)
def test_plan_rejects_other_backends(backend_name, supports_mask_free, match):
    query, key, _ = make_qkv()
    with pytest.raises(NotImplementedError, match=match):
        plan_q_chunking(
            chunks=2,
            attn_metadata=canonical_metadata(),
            query=query,
            key=key,
            backend_name=backend_name,
            backend_supports_mask_free=supports_mask_free,
        )


def test_plan_requires_metadata():
    query, key, _ = make_qkv()
    with pytest.raises(NotImplementedError, match="no attention metadata"):
        plan_q_chunking(
            chunks=2,
            attn_metadata=None,
            query=query,
            key=key,
            backend_name="FLASH_ATTN",
            backend_supports_mask_free=True,
        )


def test_plan_requires_a_single_packed_batch():
    query, key, _ = make_qkv()
    query = query.repeat(2, 1, 1, 1)
    with pytest.raises(NotImplementedError, match="single packed batch"):
        plan_q_chunking(
            chunks=2,
            attn_metadata=canonical_metadata(),
            query=query,
            key=key,
            backend_name="FLASH_ATTN",
            backend_supports_mask_free=True,
        )


# ------------------------------------------------------------------ equivalence


@pytest.mark.parametrize("chunks", [1, 2, 4, 6])
@pytest.mark.parametrize("used_q", [TOTAL_Q, USED_Q, 17])
def test_chunked_attention_matches_the_softmax_reference(chunks, used_q):
    query, key, value = make_qkv(used_q)
    metadata = canonical_metadata(used_q=used_q)
    plan = plan_for(chunks, used_q=used_q)
    out = run_chunked_attention(
        plan=plan,
        query=query,
        key=key,
        value=value,
        attn_metadata=metadata,
        kernel=sdpa_kernel,
    )
    got = out[:, :used_q].float()
    reference = softmax_reference(query, key, value, used_q, USED_K)
    assert torch.isfinite(got).all()
    rel_l2 = float((got - reference).norm() / reference.norm())
    assert rel_l2 <= REL_L2_LIMIT, f"rel-L2 {rel_l2:.3e} for chunks={chunks}, used_q={used_q}"
    assert torch.allclose(got, reference, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("chunks", [1, 2, 4, 6])
def test_chunked_attention_tracks_a_single_launch(chunks):
    """Row blocks must reproduce one launch to bf16 noise (not bit-for-bit)."""
    query, key, value = make_qkv()
    metadata = canonical_metadata()
    full = sdpa_kernel(query[:, :USED_Q].contiguous(), key[:, :USED_K], value[:, :USED_K], metadata)
    out = run_chunked_attention(
        plan=plan_for(chunks),
        query=query,
        key=key,
        value=value,
        attn_metadata=metadata,
        kernel=sdpa_kernel,
    )
    got = out[:, :USED_Q].float()
    delta = (got - full.float()).abs()
    rel_l2 = float((got - full.float()).norm() / full.float().norm())
    assert rel_l2 <= REL_L2_LIMIT
    assert float(delta.max()) <= 1e-2
    if chunks == 1:
        # One chunk is literally one launch, so it must be exact.
        assert torch.equal(out[:, :USED_Q], full)


def test_kv_perturbation_changes_the_result():
    query, key, value = make_qkv(seed=3)
    plan = plan_for(4)
    base = run_chunked_attention(
        plan=plan, query=query, key=key, value=value,
        attn_metadata=canonical_metadata(), kernel=sdpa_kernel,
    )
    perturbed = key.clone()
    perturbed[:, USED_K // 2] += 0.5
    other = run_chunked_attention(
        plan=plan, query=query, key=perturbed, value=value,
        attn_metadata=canonical_metadata(), kernel=sdpa_kernel,
    )
    assert not torch.equal(base[:, :USED_Q], other[:, :USED_Q])


@pytest.mark.parametrize("chunks", [1, 2, 4, 6])
def test_padding_rows_are_exactly_zero(chunks):
    query, key, value = make_qkv()
    out = run_chunked_attention(
        plan=plan_for(chunks), query=query, key=key, value=value,
        attn_metadata=canonical_metadata(), kernel=sdpa_kernel,
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.equal(out[:, USED_Q:], torch.zeros_like(out[:, USED_Q:]))


def test_chunk_metadata_is_a_single_document():
    metadata = canonical_metadata()
    plan = plan_for(4, metadata=metadata)
    chunk = build_chunk_metadata(metadata, plan, start=34, rows=17, device=torch.device("cpu"))
    assert chunk.attn_mask is None
    assert chunk.extra["cu_seqlens_q"].tolist() == [0, 17]
    assert chunk.extra["cu_seqlens_k"].tolist() == [0, USED_K]
    assert chunk.extra["max_seqlen_q"] == 17
    assert chunk.extra["max_seqlen_k"] == USED_K
    assert chunk.extra["valid_kv_length"] == USED_K
    assert chunk.packed_padding.q_length == 17
    assert chunk.packed_padding.kv_length == USED_K
    assert chunk.extra["q_chunk_start"] == 34
    # The parent metadata object must not be mutated.
    assert metadata.extra["cu_seqlens_q"].tolist() == [0, USED_Q, TOTAL_Q]


def test_driver_rejects_a_kernel_that_returns_the_wrong_row_count():
    query, key, value = make_qkv()
    plan = plan_for(4)

    def bad_kernel(q, k, v, metadata):
        del metadata
        return sdpa_kernel(q, k, v, None)[:, :1]

    with pytest.raises(RuntimeError, match="returned 1 rows"):
        run_chunked_attention(
            plan=plan, query=query, key=key, value=value,
            attn_metadata=canonical_metadata(), kernel=bad_kernel,
        )


# -------------------------------------------------- backend resolver contract


def _resolver():
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl.causal = False
    return impl


def test_full_metadata_resolves_to_two_documents_and_chunks_to_one():
    impl = _resolver()
    query, key, _ = make_qkv()
    full = impl._resolve_packed_seq_npu(query, key, canonical_metadata().extra)
    assert full == ([USED_Q, TOTAL_Q], [USED_K, TOTAL_K])

    plan = plan_for(4)
    rows = [end - start for start, end in plan.bounds]
    for (start, end), expected_rows in zip(plan.bounds, rows):
        chunk = build_chunk_metadata(
            canonical_metadata(), plan, start=start, rows=end - start, device=torch.device("cpu")
        )
        resolved = impl._resolve_packed_seq_npu(
            query[:, start:end].contiguous(), key[:, :USED_K], chunk.extra
        )
        assert resolved == ([expected_rows], [USED_K])
