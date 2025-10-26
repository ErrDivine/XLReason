"""Lexicon projector for bilingual alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LexiconProjector(nn.Module):
    """Projects token logits into a shared lexical embedding space."""

    def __init__(self, vocab_size: int, projection_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab_size, projection_dim) * 0.02)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return probs @ self.embedding


__all__ = ["LexiconProjector"]
