"""Shared backbone and adapters for the bilingual model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..planner.iqgp import PlannerIQGP
from ..planner.graph import PlannerGraph


class LanguageAdapter(nn.Module):
    """A lightweight bottleneck adapter for language-specific modulation."""

    def __init__(self, hidden_size: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden = self.down(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.up(hidden)
        return residual + hidden


class SharedBackbone(nn.Module):
    """Transformer encoder shared across languages."""

    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask
        return self.encoder(inputs, src_key_padding_mask=key_padding_mask)


class LanguageDecoder(nn.Module):
    """Language-specific decoder that consumes planner graphs."""

    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.graph_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True, dropout=dropout)
        self.generator = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoded: torch.Tensor, graph: PlannerGraph, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        node_states = graph.masked_node_states()
        logits, _ = self.attention(encoded, node_states, node_states, key_padding_mask=None if mask is None else ~mask)
        logits = logits + encoded
        logits = self.graph_projection(logits)
        return self.generator(logits)


@dataclass
class ReasonerOutput:
    en_logits: torch.Tensor
    zh_logits: torch.Tensor
    graph: PlannerGraph
    vq_loss: torch.Tensor


class BilingualReasoner(nn.Module):
    """Main entry point that wires backbone, planner, and decoders."""

    def __init__(
        self,
        hidden_size: int,
        vocab_en: int,
        vocab_zh: int,
        max_nodes: int,
        codebook_size: int,
        num_entities: int,
        num_units: int,
    ) -> None:
        super().__init__()
        self.backbone = SharedBackbone(hidden_size)
        self.adapter_en = LanguageAdapter(hidden_size)
        self.adapter_zh = LanguageAdapter(hidden_size)
        self.planner = PlannerIQGP(hidden_size, max_nodes, codebook_size, num_entities, num_units)
        self.decoder_en = LanguageDecoder(hidden_size, vocab_en)
        self.decoder_zh = LanguageDecoder(hidden_size, vocab_zh)

    def encode(self, inputs: torch.Tensor, mask: Optional[torch.Tensor], adapter: LanguageAdapter) -> torch.Tensor:
        hidden = self.backbone(inputs, mask)
        return adapter(hidden)

    def forward(
        self,
        en_inputs: torch.Tensor,
        zh_inputs: torch.Tensor,
        entity_candidates: Tuple[torch.Tensor, Optional[torch.Tensor]],
        unit_candidates: Tuple[torch.Tensor, Optional[torch.Tensor]],
        en_mask: Optional[torch.Tensor] = None,
        zh_mask: Optional[torch.Tensor] = None,
    ) -> ReasonerOutput:
        en_hidden = self.encode(en_inputs, en_mask, self.adapter_en)
        zh_hidden = self.encode(zh_inputs, zh_mask, self.adapter_zh)
        graph, vq_loss = self.planner(en_hidden, zh_hidden, entity_candidates, unit_candidates, token_mask=en_mask)
        en_logits = self.decoder_en(en_hidden, graph, en_mask)
        zh_logits = self.decoder_zh(zh_hidden, graph, zh_mask)
        return ReasonerOutput(en_logits=en_logits, zh_logits=zh_logits, graph=graph, vq_loss=vq_loss)


__all__ = [
    "LanguageAdapter",
    "SharedBackbone",
    "LanguageDecoder",
    "BilingualReasoner",
    "ReasonerOutput",
]
