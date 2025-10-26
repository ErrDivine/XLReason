"""Planner-specific unit tests."""

import pytest


torch = pytest.importorskip("torch")

from iqgp.planner.iqgp import PlannerIQGP


def test_planner_shapes():
    torch.manual_seed(0)
    batch, seq_len, hidden = 2, 5, 16
    max_nodes = 4
    codebook_size = 8
    num_entities = 6
    num_units = 5

    planner = PlannerIQGP(hidden, max_nodes, codebook_size, num_entities, num_units)
    h_en = torch.randn(batch, seq_len, hidden)
    h_zh = torch.randn(batch, seq_len, hidden)
    entity_candidates = (torch.randn(batch, num_entities, hidden), torch.ones(batch, num_entities, dtype=torch.bool))
    unit_candidates = (torch.randn(batch, num_units, hidden), torch.ones(batch, num_units, dtype=torch.bool))

    graph, vq_loss = planner(h_en, h_zh, entity_candidates, unit_candidates)

    assert graph.node_states.shape == (batch, max_nodes, hidden)
    assert graph.codes.shape == (batch, max_nodes)
    assert graph.qid_logits.shape == (batch, max_nodes, num_entities)
    assert graph.unit_logits.shape == (batch, max_nodes, num_units)
    assert graph.edge_logits.shape == (batch, max_nodes, max_nodes)
    assert graph.node_mask.shape == (batch, max_nodes)
    assert vq_loss.dim() == 0
    assert graph.codes.max().item() < codebook_size
    assert graph.codes.min().item() >= 0
