#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence harness for exact Q-row chunking (H3_ATTN_Q_CHUNKS).

Compares, for every requested Q offset and chunk count, the chunked driver
against a straightforward float32 softmax-over-full-KV reference, and checks
the backend resolver contract that makes the chunking exact:

  * full metadata resolves to a ``[real, pad]`` two-document packing;
  * every chunk resolves to a *single* document over the full valid KV;
  * a ``full_qk`` mask is rejected instead of silently falling back.

This never touches an NPU: it runs on CPU tensors with a stand-in kernel.

Usage::

    python3 scripts/h3_attention_equivalence.py --q-offsets 0 17 65 --full-kv --reject-full-qk
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.q_chunk import (
    ALLOWED_Q_CHUNKS,
    build_chunk_metadata,
    plan_q_chunking,
    run_chunked_attention,
)

USED_Q = 65
USED_K = 129
PAD = 48
TOTAL_Q = USED_Q + PAD
TOTAL_K = USED_K + PAD
HEADS = 7
DIM = 128
DTYPE = torch.bfloat16
REL_L2_LIMIT = 5e-3


@dataclass
class Case:
    chunks: int
    q_offset: int
    used_q: int
    rel_l2: float
    max_abs: float
    finite: bool
    bitwise_stable: bool
    tail_rows: int


@dataclass
class Report:
    cases: list[Case] = field(default_factory=list)
    resolver_full: list[int] | None = None
    resolver_full_kv: list[int] | None = None
    resolver_chunks: list[list[int]] = field(default_factory=list)
    full_qk_rejected: bool | None = None
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "used_q": USED_Q,
            "used_k": USED_K,
            "pad": PAD,
            "heads": HEADS,
            "dim": DIM,
            "dtype": str(DTYPE).replace("torch.", ""),
            "rel_l2_limit": REL_L2_LIMIT,
            "cases": [c.__dict__ for c in self.cases],
            "resolver_full": self.resolver_full,
            "resolver_full_kv": self.resolver_full_kv,
            "resolver_chunks": self.resolver_chunks,
            "full_qk_rejected": self.full_qk_rejected,
            "failures": self.failures,
            "ok": not self.failures,
        }


def canonical_metadata(q_used: int = USED_Q, kv_used: int = USED_K) -> AttentionMetadata:
    cu_q = torch.tensor([0, q_used, TOTAL_Q], dtype=torch.int32)
    cu_k = torch.tensor([0, kv_used, TOTAL_K], dtype=torch.int32)
    return AttentionMetadata(
        packed_padding=PackedPaddingMetadata(
            q_length=q_used, kv_length=kv_used, cu_seqlens_q=cu_q[:2], cu_seqlens_k=cu_k[:2]
        ),
        extra={
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": q_used,
            "max_seqlen_k": kv_used,
            "valid_kv_length": kv_used,
            "npu_attn_varlen": True,
            "laser_input_scale": 8.0,
        },
    )


def sdpa_kernel(query, key, value, metadata):
    del metadata

    def bhsd(t):
        return t.transpose(1, 2).float()

    out = F.scaled_dot_product_attention(bhsd(query), bhsd(key), bhsd(value), is_causal=False)
    return out.transpose(1, 2).to(query.dtype)


def softmax_reference(query, key, value, used_q: int = USED_Q) -> torch.Tensor:
    q = query[:, :used_q].transpose(1, 2).float()
    k = key[:, :USED_K].transpose(1, 2).float()
    v = value[:, :USED_K].transpose(1, 2).float()
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DIM)
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2)


def make_qkv(used_q: int = USED_Q, seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    scale = 1.0 / math.sqrt(DIM)
    query = (torch.randn(1, TOTAL_Q, HEADS, DIM, generator=gen) * scale).to(DTYPE)
    key = (torch.randn(1, TOTAL_K, HEADS, DIM, generator=gen) * scale).to(DTYPE)
    value = (torch.randn(1, TOTAL_K, HEADS, DIM, generator=gen) * scale).to(DTYPE)
    return query, key, value


def run_case(report: Report, chunks: int, q_offset: int) -> None:
    """``q_offset`` is the number of real Q rows before the padding document.

    ``0`` means "no padding document" (used_q == TOTAL_Q); any other value is
    the real document length, so the padding document starts at that row - the
    non-zero q_start case the acceptance asks for.
    """
    used_q = TOTAL_Q if q_offset == 0 else q_offset
    query, key, value = make_qkv(used_q)
    metadata = canonical_metadata(used_q)
    plan = plan_q_chunking(
        chunks=chunks,
        attn_metadata=metadata,
        query=query,
        key=key,
        backend_name="FLASH_ATTN",
        backend_supports_mask_free=True,
    )
    out = run_chunked_attention(
        plan=plan, query=query, key=key, value=value, attn_metadata=metadata, kernel=sdpa_kernel
    )
    got = out[:, :used_q].float()
    reference = softmax_reference(query, key, value, used_q)
    rel_l2 = float((got - reference).norm() / reference.norm())
    max_abs = float((got - reference).abs().max())
    finite = bool(torch.isfinite(out).all())

    full = sdpa_kernel(query[:, :used_q].contiguous(), key[:, :USED_K], value[:, :USED_K], metadata)
    bitwise_stable = bool(torch.equal(out[:, :used_q], full))
    tail_rows = plan.bounds[-1][1] - plan.bounds[-1][0]
    padding_zero = bool(torch.equal(out[:, used_q:], torch.zeros_like(out[:, used_q:])))

    report.cases.append(
        Case(
            chunks=chunks,
            q_offset=q_offset,
            used_q=used_q,
            rel_l2=rel_l2,
            max_abs=max_abs,
            finite=finite,
            bitwise_stable=bitwise_stable,
            tail_rows=tail_rows,
        )
    )
    if rel_l2 > REL_L2_LIMIT:
        report.failures.append(f"chunks={chunks} q_offset={q_offset}: rel-L2 {rel_l2:.3e} > {REL_L2_LIMIT}")
    if not finite:
        report.failures.append(f"chunks={chunks} q_offset={q_offset}: non-finite output")
    if not padding_zero:
        report.failures.append(f"chunks={chunks} q_offset={q_offset}: padding rows are not zero")


def check_resolver(report: Report, full_kv: bool) -> None:
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl.causal = False
    query, key, _ = make_qkv(USED_Q)
    metadata = canonical_metadata(USED_Q)
    resolved = impl._resolve_packed_seq_npu(query, key, metadata.extra)
    report.resolver_full = resolved[0] if resolved else None
    report.resolver_full_kv = resolved[1] if resolved else None
    if resolved != ([USED_Q, TOTAL_Q], [USED_K, TOTAL_K]):
        report.failures.append(f"full metadata resolved to {resolved}, expected a two-document packing")
    if full_kv and report.resolver_full_kv != [USED_K, TOTAL_K]:
        report.failures.append(f"full-KV contract broken: KV documents {report.resolver_full_kv}")

    # Without a padding document the resolver may report either one document
    # or a zero-length second one; both describe the same attention.
    unpadded = impl._resolve_packed_seq_npu(query, key, canonical_metadata(TOTAL_Q).extra)
    if (
        unpadded is None
        or unpadded[0][0] != TOTAL_Q
        or unpadded[0][-1] != TOTAL_Q
        or unpadded[1][0] != USED_K
    ):
        report.failures.append(
            f"unpadded metadata resolved to {unpadded}, expected the identity Q range over the valid KV prefix"
        )

    plan = plan_q_chunking(
        chunks=4,
        attn_metadata=metadata,
        query=query,
        key=key,
        backend_name="FLASH_ATTN",
        backend_supports_mask_free=True,
    )
    for start, end in plan.bounds:
        chunk = build_chunk_metadata(metadata, plan, start=start, rows=end - start, device=query.device)
        chunk_resolved = impl._resolve_packed_seq_npu(
            query[:, start:end].contiguous(), key[:, :USED_K], chunk.extra
        )
        report.resolver_chunks.append([chunk_resolved[0][0], chunk_resolved[1][0]] if chunk_resolved else [])
        if chunk_resolved != ([end - start], [USED_K]):
            report.failures.append(
                f"chunk [{start},{end}) resolved to {chunk_resolved}, expected a single document over full KV"
            )


def check_full_qk_rejected(report: Report) -> None:
    query, key, _ = make_qkv(USED_Q)
    metadata = canonical_metadata(USED_Q)
    metadata.attn_mask = torch.arange(TOTAL_Q)[None] < USED_Q
    try:
        plan_q_chunking(
            chunks=4,
            attn_metadata=metadata,
            query=query,
            key=key,
            backend_name="FLASH_ATTN",
            backend_supports_mask_free=True,
        )
    except NotImplementedError as exc:
        report.full_qk_rejected = "full_qk" in str(exc)
        if not report.full_qk_rejected:
            report.failures.append(f"mask rejection message lost the full_qk wording: {exc}")
    else:
        report.full_qk_rejected = False
        report.failures.append("a masked (full_qk) forward was accepted by the chunk planner")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=int, default=list(ALLOWED_Q_CHUNKS))
    parser.add_argument("--q-offsets", nargs="+", type=int, default=[0, 17, 65])
    parser.add_argument("--full-kv", action="store_true", help="assert KV is never split across chunks")
    parser.add_argument("--reject-full-qk", action="store_true", help="assert masked forwards are rejected")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    report = Report()
    for chunks in args.chunks:
        for q_offset in args.q_offsets:
            run_case(report, chunks, q_offset)
    check_resolver(report, args.full_kv)
    if args.reject_full_qk:
        check_full_qk_rejected(report)

    payload = report.to_json()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
