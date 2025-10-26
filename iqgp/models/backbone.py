"""Backbone encoder producing bilingual hidden states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    en_hidden: torch.Tensor
    zh_hidden: torch.Tensor


class BilingualBackbone(nn.Module):
    """Lightweight bilingual encoder with shared parameters."""

    def __init__(self, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, en_input: torch.Tensor, zh_input: torch.Tensor) -> BackboneOutput:
        if en_input.shape != zh_input.shape:
            raise ValueError("English and Chinese inputs must have the same shape")
        en_norm = self.layer_norm(en_input)
        zh_norm = self.layer_norm(zh_input)
        en_hidden, _ = self.encoder(en_norm)
        zh_hidden, _ = self.encoder(zh_norm)
        return BackboneOutput(en_hidden=en_hidden, zh_hidden=zh_hidden)


__all__ = ["BackboneOutput", "BilingualBackbone"]
