"""Data structures for planner outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PlannerGraph:
    """Container for planner outputs.

    Attributes:
        node_states: Tensor of shape (batch, num_nodes, hidden_size).
        codes: Tensor of shape (batch, num_nodes) with VQ code indices.
        qid_logits: Tensor of shape (batch, num_nodes, num_entities).
        unit_logits: Tensor of shape (batch, num_nodes, num_units).
        edge_logits: Tensor of shape (batch, num_nodes, num_nodes).
        node_mask: Bool tensor indicating valid nodes (batch, num_nodes).
    """

    node_states: torch.Tensor
    codes: torch.Tensor
    qid_logits: torch.Tensor
    unit_logits: torch.Tensor
    edge_logits: torch.Tensor
    node_mask: torch.Tensor

    def masked_node_states(self) -> torch.Tensor:
        """Return node states with padded positions zeroed out."""
        if self.node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be a boolean tensor")
        return self.node_states * self.node_mask.unsqueeze(-1)

    def adjacency(self, threshold: float = 0.0) -> torch.Tensor:
        """Return a binary adjacency matrix using the provided threshold."""
        return (self.edge_logits > threshold).to(self.edge_logits.dtype)

    def split_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split states into active and padded tensors."""
        mask = self.node_mask.unsqueeze(-1).expand(-1, -1, self.node_states.size(-1))
        if mask.any():
            active = self.node_states[mask].reshape(-1, self.node_states.size(-1))
        else:
            active = self.node_states.new_zeros((0, self.node_states.size(-1)))
        padded_mask = ~mask
        if padded_mask.any():
            padded = self.node_states[padded_mask].reshape(-1, self.node_states.size(-1))
        else:
            padded = self.node_states.new_zeros((0, self.node_states.size(-1)))
        return active, padded

    def detach(self) -> "PlannerGraph":
        """Detach all tensors from the graph for logging or evaluation."""
        return PlannerGraph(
            node_states=self.node_states.detach(),
            codes=self.codes.detach(),
            qid_logits=self.qid_logits.detach(),
            unit_logits=self.unit_logits.detach(),
            edge_logits=self.edge_logits.detach(),
            node_mask=self.node_mask.detach(),
        )

    def to(self, device: Optional[torch.device] = None) -> "PlannerGraph":
        """Move tensors to the given device."""
        if device is None:
            return self
        return PlannerGraph(
            node_states=self.node_states.to(device),
            codes=self.codes.to(device),
            qid_logits=self.qid_logits.to(device),
            unit_logits=self.unit_logits.to(device),
            edge_logits=self.edge_logits.to(device),
            node_mask=self.node_mask.to(device),
        )
