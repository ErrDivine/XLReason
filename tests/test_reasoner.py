"""Integration tests for the bilingual reasoner stack."""

import pytest


torch = pytest.importorskip("torch")

from iqgp.models.backbone import BilingualReasoner


def test_reasoner_forward():
    torch.manual_seed(42)
    batch, seq_len, hidden = 2, 5, 16
    vocab = 12
    max_nodes = 4
    codebook = 8
    entities = 6
    units = 5

    model = BilingualReasoner(hidden, vocab, vocab, max_nodes, codebook, entities, units)
    en_inputs = torch.randn(batch, seq_len, hidden)
    zh_inputs = torch.randn(batch, seq_len, hidden)
    en_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    zh_mask = torch.ones(batch, seq_len, dtype=torch.bool)

    entity_candidates = (torch.randn(batch, entities, hidden), torch.ones(batch, entities, dtype=torch.bool))
    unit_candidates = (torch.randn(batch, units, hidden), torch.ones(batch, units, dtype=torch.bool))

    output = model(en_inputs, zh_inputs, entity_candidates, unit_candidates, en_mask=en_mask, zh_mask=zh_mask)

    assert output.en_logits.shape == (batch, seq_len, vocab)
    assert output.zh_logits.shape == (batch, seq_len, vocab)
    assert output.graph.node_states.shape[-1] == hidden
    assert output.vq_loss.dim() == 0
