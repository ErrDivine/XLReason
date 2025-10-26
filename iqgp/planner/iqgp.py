"""Implementation of the Interlingua QID-Graph planner."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import PlannerGraph
from ..utils.vq import vector_quantize, commitment_loss


class PlannerIQGP(nn.Module):
    """Build a discrete plan graph shared by English and Chinese views."""

    def __init__(
        self,
        hidden_size: int,
        max_nodes: int,
        codebook_size: int,
        num_entities: int,
        num_units: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes
        self.num_entities = num_entities
        self.num_units = num_units

        self.cross_attn_en = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_zh = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.node_queries = nn.Parameter(torch.randn(max_nodes, hidden_size))
        self.node_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.node_norm = nn.LayerNorm(hidden_size)

        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_size))

        self.entity_pointer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.unit_pointer = nn.Linear(hidden_size, hidden_size, bias=False)

        self.edge_scorer = nn.Bilinear(hidden_size, hidden_size, 1)
        self.edge_bias = nn.Parameter(torch.zeros(max_nodes, max_nodes))

        self.length_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def fuse(self, h_en: torch.Tensor, h_zh: torch.Tensor) -> torch.Tensor:
        """Fuse English and Chinese hidden states via cross attention."""
        attn_en, _ = self.cross_attn_en(h_en, h_zh, h_zh)
        attn_zh, _ = self.cross_attn_zh(h_zh, h_en, h_en)
        fused_en = self.fuse_gate(torch.cat([h_en, attn_en], dim=-1))
        fused_zh = self.fuse_gate(torch.cat([h_zh, attn_zh], dim=-1))
        return 0.5 * (fused_en + fused_zh)

    def propose_nodes(self, fused: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = fused.size(0)
        queries = self.node_queries.unsqueeze(0).expand(batch_size, -1, -1)
        nodes, _ = self.node_attn(queries, fused, fused, key_padding_mask=None if mask is None else ~mask)
        nodes = self.node_norm(nodes)
        return nodes

    def pointer_scores(
        self,
        node_states: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
        pointer: nn.Linear,
    ) -> torch.Tensor:
        projected_nodes = pointer(node_states)
        scores = torch.matmul(projected_nodes, candidates.transpose(-1, -2))
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.unsqueeze(1), float("-inf"))
        return scores

    def forward(
        self,
        h_en: torch.Tensor,
        h_zh: torch.Tensor,
        entity_candidates: Tuple[torch.Tensor, Optional[torch.Tensor]],
        unit_candidates: Tuple[torch.Tensor, Optional[torch.Tensor]],
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[PlannerGraph, torch.Tensor]:
        fused = self.fuse(h_en, h_zh)
        node_states = self.propose_nodes(fused, token_mask)
        quantized, codes = vector_quantize(node_states, self.codebook)
        vq_loss = commitment_loss(node_states, quantized)

        ent_embeds, ent_mask = entity_candidates
        unit_embeds, unit_mask = unit_candidates

        qid_logits = self.pointer_scores(quantized, ent_embeds, ent_mask, self.entity_pointer)
        unit_logits = self.pointer_scores(quantized, unit_embeds, unit_mask, self.unit_pointer)

        edge_logits = self.compute_edges(quantized)
        node_mask = self.compute_node_mask(node_states)

        graph = PlannerGraph(
            node_states=quantized,
            codes=codes,
            qid_logits=qid_logits,
            unit_logits=unit_logits,
            edge_logits=edge_logits,
            node_mask=node_mask,
        )
        return graph, vq_loss

    def compute_edges(self, node_states: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = node_states.shape
        flat_src = node_states.unsqueeze(2).expand(-1, -1, nodes, -1)
        flat_dst = node_states.unsqueeze(1).expand(-1, nodes, -1, -1)
        scores = self.edge_scorer(flat_src, flat_dst).squeeze(-1)
        scores = scores + self.edge_bias.unsqueeze(0)
        return scores

    def compute_node_mask(self, node_states: torch.Tensor) -> torch.Tensor:
        length_logits = self.length_controller(node_states).squeeze(-1)
        length_scores = torch.sigmoid(length_logits)
        if length_scores.dim() != 2:
            raise ValueError("Unexpected shape for length scores")
        threshold = 0.5
        return length_scores > threshold


__all__ = ["PlannerIQGP"]
