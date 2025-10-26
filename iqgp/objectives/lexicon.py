"""Lexicon projection layers for aligning vocabulary logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class LexiconProjector(nn.Module):
    """Project vocabulary logits into a shared bilingual space."""

    def __init__(self, vocab_size: int, projection_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab_size, projection_dim))
        self.bias = nn.Parameter(torch.zeros(projection_dim)) if bias else None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(-1) != self.embedding.size(0):
            raise ValueError("Logit dimension does not match projector vocabulary size")
        projected = logits @ self.embedding
        if self.bias is not None:
            projected = projected + self.bias
        return projected


__all__ = ["LexiconProjector"]
