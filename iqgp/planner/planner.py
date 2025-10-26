"""Planner implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import InterlinguaGraph
from .vq import VectorQuantizerEMA, VQOutput


def _expand_queries(batch_size: int, queries: torch.Tensor) -> torch.Tensor:
    """Expand query parameters across the batch dimension."""

    return queries.unsqueeze(0).expand(batch_size, *queries.shape)


class CrossAttentionFusion(nn.Module):
    """Fuse bilingual representations via cross-attention + gating."""

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn_en = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attn_zh = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, en: torch.Tensor, zh: torch.Tensor) -> torch.Tensor:
        en_ctx, _ = self.attn_en(en, zh, zh)
        zh_ctx, _ = self.attn_zh(zh, en, en)
        fused = torch.cat([en_ctx, zh_ctx, 0.5 * (en + zh)], dim=-1)
        fused = torch.tanh(self.proj(fused))
        gate = torch.sigmoid(self.gate(fused))
        return gate * fused + (1 - gate) * (0.5 * (en + zh))


class BiaffineRelationHead(nn.Module):
    """Predict relation scores between node pairs."""

    def __init__(self, input_dim: int, num_relations: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.query = nn.Linear(input_dim, num_relations * input_dim, bias=False)
        self.key = nn.Linear(input_dim, num_relations * input_dim, bias=False)
        self.scale = input_dim ** -0.5

    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, hidden = node_states.shape
        query = self.query(node_states).view(batch, num_nodes, self.num_relations, hidden)
        key = self.key(node_states).view(batch, num_nodes, self.num_relations, hidden)
        logits = torch.einsum("binh,bjnh->bijn", query, key) * self.scale
        return logits


@dataclass
class PlannerOutput:
    graph: InterlinguaGraph
    vq: VQOutput


class InterlinguaPlanner(nn.Module):
    """Generates a discrete, entity-grounded reasoning graph."""

    def __init__(
        self,
        hidden_size: int,
        num_nodes: int,
        codebook_size: int,
        embedding_dim: int,
        num_entities: int,
        num_units: int,
        num_edge_types: int = 4,
        commitment_cost: float = 0.25,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if embedding_dim != hidden_size:
            self.proj_to_embed = nn.Linear(hidden_size, embedding_dim)
        else:
            self.proj_to_embed = nn.Identity()
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.fusion = CrossAttentionFusion(hidden_size, num_heads=attn_heads)
        self.node_queries = nn.Parameter(torch.randn(num_nodes, hidden_size))
        self.node_attention = nn.MultiheadAttention(hidden_size, attn_heads, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.vq = VectorQuantizerEMA(codebook_size, embedding_dim, commitment_cost)
        self.arg_query = nn.Linear(embedding_dim, embedding_dim)
        self.arg_key = nn.Linear(embedding_dim, embedding_dim)
        self.edge_head = BiaffineRelationHead(embedding_dim, num_edge_types)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.unit_embeddings = nn.Embedding(num_units, embedding_dim)
        self.evidence_proj = nn.Linear(hidden_size, embedding_dim)

    def forward(
        self,
        en_hidden: torch.Tensor,
        zh_hidden: torch.Tensor,
        evidence_hidden: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        if en_hidden.shape != zh_hidden.shape:
            raise ValueError("English and Chinese hidden states must share shape")
        fused = self.fusion(en_hidden, zh_hidden)
        fused = self.node_norm(fused)

        batch_size, seq_len, _ = fused.shape
        queries = _expand_queries(batch_size, self.node_queries)
        node_states, _ = self.node_attention(queries, fused, fused)
        node_states = self.node_norm(node_states)

        node_embed = self.proj_to_embed(node_states)
        vq_out = self.vq(node_embed)
        quantized_nodes = vq_out.quantized

        arg_q = self.arg_query(quantized_nodes)
        arg_k = self.arg_key(quantized_nodes)
        arg_logits = torch.einsum("bik,bjk->bij", arg_q, arg_k) * (self.embedding_dim ** -0.5)

        edge_logits = self.edge_head(quantized_nodes)

        entity_scores = torch.matmul(quantized_nodes, self.entity_embeddings.weight.t())
        unit_scores = torch.matmul(quantized_nodes, self.unit_embeddings.weight.t())

        if evidence_hidden is None:
            evidence_logits = None
        else:
            evidence_proj = self.evidence_proj(evidence_hidden)
            evidence_logits = torch.einsum("bik,bjk->bij", quantized_nodes, evidence_proj) * (self.embedding_dim ** -0.5)

        graph = InterlinguaGraph(
            codes=vq_out.codes,
            node_states=quantized_nodes,
            arg_logits=arg_logits,
            edge_logits=edge_logits,
            qid_logits=entity_scores,
            unit_logits=unit_scores,
            evidence_logits=evidence_logits,
        )
        return PlannerOutput(graph=graph, vq=vq_out)


__all__ = ["InterlinguaPlanner", "PlannerOutput"]
