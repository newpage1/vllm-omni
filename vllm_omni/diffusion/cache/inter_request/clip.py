# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CLIP prompt embeddings for the semantic cross-request cache path.

Only the text tower is loaded (``CLIPTextModel``), so the 1.7 GB CLIP
checkpoint costs roughly 0.5 GB of host memory and no NPU memory.  Similarity
is cosine over the pooled text embedding, matching the PR #2 semantics.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class CLIPPromptEmbedder:
    """Lazily loaded CLIP text encoder returning L2-normalised embeddings."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        from transformers import CLIPTextModel, CLIPTokenizer

        self.model_path = model_path
        self.device = torch.device(device)
        logger.info("[inter-request] loading CLIP text encoder from %s on %s", model_path, self.device)
        self._tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = CLIPTextModel.from_pretrained(model_path, local_files_only=True)
        self._model.to(self.device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        tokens = self._tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        pooled = self._model(**tokens).pooler_output.to(dtype=torch.float32)
        return pooled[0].detach().to("cpu")

    @torch.inference_mode()
    def similarity(self, left: str | torch.Tensor, right: str | torch.Tensor) -> float:
        """Cosine similarity; arguments may be prompts or pre-encoded embeddings."""
        a = left if isinstance(left, torch.Tensor) else self.encode(left)
        b = right if isinstance(right, torch.Tensor) else self.encode(right)
        a = a.reshape(-1).to(dtype=torch.float32)
        b = b.reshape(-1).to(dtype=torch.float32)
        return float(torch.dot(a / a.norm().clamp_min(1e-12), b / b.norm().clamp_min(1e-12)))

    def describe(self) -> dict[str, Any]:
        return {"model_path": self.model_path, "device": str(self.device)}


__all__ = ["CLIPPromptEmbedder"]
