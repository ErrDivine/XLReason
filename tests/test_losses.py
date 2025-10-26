import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for loss tests")
def test_loss_components_backpropagate():
    from iqgp.objectives import LossWeights, compute_total_loss
    from iqgp.planner import InterlinguaPlanner

    torch.manual_seed(1)
    batch = 2
    length = 5
    hidden = 16
    vocab = 11
    num_nodes = 4

    planner = InterlinguaPlanner(
        hidden_size=hidden,
        num_nodes=num_nodes,
        codebook_size=8,
        embedding_dim=hidden,
        num_entities=9,
        num_units=7,
    )
    en_hidden = torch.randn(batch, length, hidden, requires_grad=True)
    zh_hidden = torch.randn(batch, length, hidden, requires_grad=True)
    planner_output = planner(en_hidden, zh_hidden)

    answer_logits_en = torch.randn(batch, vocab, requires_grad=True)
    answer_logits_zh = torch.randn(batch, vocab, requires_grad=True)
    answers_en = torch.randint(vocab, (batch,))
    answers_zh = torch.randint(vocab, (batch,))

    cot_logits_en = torch.randn(batch, length, vocab, requires_grad=True)
    cot_logits_zh = torch.randn(batch, length, vocab, requires_grad=True)
    cot_en = torch.randint(vocab, (batch, length))
    cot_zh = torch.randint(vocab, (batch, length))

    projected_en = torch.randn(batch, length, hidden, requires_grad=True)
    projected_zh = torch.randn(batch, length, hidden, requires_grad=True)

    plan_states_en = torch.randn(batch, hidden, requires_grad=True)
    plan_states_zh = torch.randn(batch, hidden, requires_grad=True)

    entity_targets = torch.randint(9, (batch, num_nodes))
    unit_targets = torch.randint(7, (batch, num_nodes))

    adversary_logits = torch.randn(batch * 2, 2, requires_grad=True)
    language_targets = torch.cat([
        torch.zeros(batch, dtype=torch.long),
        torch.ones(batch, dtype=torch.long),
    ])

    total, bundle = compute_total_loss(
        planner_output=planner_output,
        answer_logits_en=answer_logits_en,
        answer_logits_zh=answer_logits_zh,
        answer_targets_en=answers_en,
        answer_targets_zh=answers_zh,
        projected_en=projected_en,
        projected_zh=projected_zh,
        plan_states_en=plan_states_en,
        plan_states_zh=plan_states_zh,
        entity_targets=entity_targets,
        unit_targets=unit_targets,
        loss_weights=LossWeights(),
        cot_logits_en=cot_logits_en,
        cot_logits_zh=cot_logits_zh,
        cot_targets_en=cot_en,
        cot_targets_zh=cot_zh,
        switched_graph=planner_output.graph,
        adversary_logits=adversary_logits,
        language_targets=language_targets,
    )

    total.backward()
    assert total.item() > 0
    assert bundle.task.item() > 0
    assert planner_output.graph.node_states.grad is None
    assert answer_logits_en.grad is not None
    assert adversary_logits.grad is not None
