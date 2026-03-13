import pytest

try:
    import torch
except ImportError:
    torch = None


def test_bilingual_agreement_identical():
    from iqgp.utils.metrics import bilingual_agreement

    assert bilingual_agreement(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_bilingual_agreement_none_match():
    from iqgp.utils.metrics import bilingual_agreement

    assert bilingual_agreement(["a", "b"], ["x", "y"]) == 0.0


def test_bilingual_agreement_partial():
    from iqgp.utils.metrics import bilingual_agreement

    assert bilingual_agreement(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)


def test_bilingual_agreement_empty():
    from iqgp.utils.metrics import bilingual_agreement

    assert bilingual_agreement([], []) == 0.0


def test_bilingual_agreement_different_lengths():
    from iqgp.utils.metrics import bilingual_agreement

    result = bilingual_agreement(["a", "b", "c"], ["a", "b"])
    assert result == 1.0  # 2 matches out of min(3,2)=2


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_graph_f1_perfect():
    from iqgp.utils.metrics import graph_f1

    edges = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert graph_f1(edges, edges) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_graph_f1_no_overlap():
    from iqgp.utils.metrics import graph_f1

    pred = torch.tensor([1.0, 0.0])
    gold = torch.tensor([0.0, 1.0])
    assert graph_f1(pred, gold) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_graph_f1_all_zero():
    from iqgp.utils.metrics import graph_f1

    z = torch.zeros(4)
    assert graph_f1(z, z) == 0.0


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_plan_stability_identical():
    from iqgp.utils.metrics import plan_stability

    t = torch.randn(3, 4)
    assert plan_stability(t, t) == pytest.approx(1.0)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_plan_stability_different():
    from iqgp.utils.metrics import plan_stability

    a = torch.zeros(3)
    b = torch.ones(3) * 10
    result = plan_stability(a, b)
    assert 0.0 < result < 1.0
