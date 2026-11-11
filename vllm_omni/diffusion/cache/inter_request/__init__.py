# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-request (inter-request) diffusion cache.

Ported from cta-11/vllm-omni PR #2 (``vllm_omni/diffusion/cache/inter_request``)
to the vLLM-Omni 0.32.0 ``CacheBackend`` lifecycle and adapted to MiniMax-H3
T2VA (video + audio dual latent, USP/ring parallel, DLO).
"""

from .backend import (
    HIT_EXACT,
    HIT_MISS,
    HIT_PARTIAL,
    InterRequestCacheBackend,
    InterRequestDecision,
    InterRequestSession,
    compute_skip_steps,
)
from .config import InterRequestCacheConfig
from .keys import H3CacheKey, build_h3_cache_key
from .store import H3CacheEntry, H3InterRequestCacheStore, H3StepCheckpoint

__all__ = [
    "HIT_EXACT",
    "HIT_MISS",
    "HIT_PARTIAL",
    "H3CacheEntry",
    "H3CacheKey",
    "H3InterRequestCacheStore",
    "H3StepCheckpoint",
    "InterRequestCacheBackend",
    "InterRequestCacheConfig",
    "InterRequestDecision",
    "InterRequestSession",
    "build_h3_cache_key",
    "compute_skip_steps",
]
