"""Decoders for verbalizing from the interlingua graph."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DecoderOutput:
    logits: torch.Tensor


class CoTDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> DecoderOutput:
        return DecoderOutput(logits=self.proj(hidden))


class AnswerDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: torch.Tensor) -> DecoderOutput:
        return DecoderOutput(logits=self.proj(hidden))


__all__ = ["DecoderOutput", "CoTDecoder", "AnswerDecoder"]
