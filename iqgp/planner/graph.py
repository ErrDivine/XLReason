"""Graph data structures for the interlingua planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class GraphNode:
    """Lightweight container for a single planner node."""

    op_code: int
    args: Sequence[int]
    qids: Sequence[int]
    units: Sequence[int]
    evidence_ptrs: Sequence[int]


@dataclass
class GraphEdge:
    """Typed, directed dependency edge."""

    src: int
    dst: int
    label: int


@dataclass
class InterlinguaGraph:
    """Tensor representation of planner outputs.

    Attributes:
        codes: LongTensor [B, K] of vector-quantized code indices.
        node_states: Tensor [B, K, D] quantized node embeddings.
        arg_logits: Tensor [B, K, K] pointer scores for node arguments.
        edge_logits: Tensor [B, K, K, R] scores over edge relation labels.
        qid_logits: Tensor [B, K, E] entity pointer scores.
        unit_logits: Tensor [B, K, U] unit pointer scores.
        evidence_logits: Optional Tensor [B, K, S] evidence span scores.
    """

    codes: torch.Tensor
    node_states: torch.Tensor
    arg_logits: torch.Tensor
    edge_logits: torch.Tensor
    qid_logits: torch.Tensor
    unit_logits: torch.Tensor
    evidence_logits: Optional[torch.Tensor] = None

    def num_nodes(self) -> int:
        return int(self.codes.shape[-1])

    def detach(self) -> "InterlinguaGraph":
        """Return a copy with all tensors detached."""

        kwargs = {
            "codes": self.codes.detach(),
            "node_states": self.node_states.detach(),
            "arg_logits": self.arg_logits.detach(),
            "edge_logits": self.edge_logits.detach(),
            "qid_logits": self.qid_logits.detach(),
            "unit_logits": self.unit_logits.detach(),
            "evidence_logits": self.evidence_logits.detach() if self.evidence_logits is not None else None,
        }
        return InterlinguaGraph(**kwargs)

    def to_dict(self) -> dict:
        """Convert the graph into a JSON-serializable dictionary."""

        payload = {
            "codes": self.codes.detach().cpu().tolist(),
            "edges": self.edge_logits.detach().cpu().tolist(),
            "args": self.arg_logits.detach().cpu().tolist(),
            "qids": self.qid_logits.detach().cpu().tolist(),
            "units": self.unit_logits.detach().cpu().tolist(),
        }
        if self.evidence_logits is not None:
            payload["evidence"] = self.evidence_logits.detach().cpu().tolist()
        return payload


__all__ = ["GraphNode", "GraphEdge", "InterlinguaGraph"]
