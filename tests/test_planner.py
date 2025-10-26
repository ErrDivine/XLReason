import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for planner tests")
def test_planner_outputs_have_expected_shapes():
    from iqgp.planner import InterlinguaPlanner

    torch.manual_seed(0)
    batch = 2
    length = 10
    hidden = 32
    num_nodes = 6
    embedding_dim = 32
    codebook_size = 16
    en_hidden = torch.randn(batch, length, hidden)
    zh_hidden = torch.randn(batch, length, hidden)

    planner = InterlinguaPlanner(
        hidden_size=hidden,
        num_nodes=num_nodes,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_entities=20,
        num_units=12,
    )
    output = planner(en_hidden, zh_hidden)

    graph = output.graph
    assert graph.node_states.shape == (batch, num_nodes, embedding_dim)
    assert graph.arg_logits.shape == (batch, num_nodes, num_nodes)
    assert graph.edge_logits.shape == (batch, num_nodes, num_nodes, 4)
    assert graph.qid_logits.shape[-1] == 20
    assert graph.unit_logits.shape[-1] == 12
    assert output.vq.codes.shape == (batch, num_nodes)
    assert output.vq.vq_loss.ndim == 0
