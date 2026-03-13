import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_vq_output_shapes():
    from iqgp.planner.vq import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=16, embedding_dim=8)
    inputs = torch.randn(2, 5, 8)
    out = vq(inputs)
    assert out.quantized.shape == (2, 5, 8)
    assert out.codes.shape == (2, 5)
    assert out.commitment_loss.ndim == 0
    assert out.codebook_loss.ndim == 0
    assert out.vq_loss.ndim == 0


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_vq_straight_through_gradients():
    from iqgp.planner.vq import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=16, embedding_dim=8)
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    out = vq(inputs)
    loss = out.quantized.sum() + out.vq_loss
    loss.backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_vq_ema_updates_during_training():
    from iqgp.planner.vq import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=8, embedding_dim=4)
    vq.train()
    embedding_before = vq.embedding.clone()
    inputs = torch.randn(3, 6, 4)
    vq(inputs)
    assert not torch.allclose(vq.embedding, embedding_before)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_vq_no_updates_during_eval():
    from iqgp.planner.vq import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=8, embedding_dim=4)
    vq.eval()
    embedding_before = vq.embedding.clone()
    inputs = torch.randn(3, 6, 4)
    vq(inputs)
    assert torch.allclose(vq.embedding, embedding_before)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_vq_multi_step_no_graph_error():
    """Ensure consecutive forward+backward calls don't cause graph errors."""
    from iqgp.planner.vq import VectorQuantizerEMA

    torch.manual_seed(42)
    vq = VectorQuantizerEMA(num_embeddings=8, embedding_dim=4)
    vq.train()
    for _ in range(3):
        inputs = torch.randn(2, 4, 4, requires_grad=True)
        out = vq(inputs)
        loss = out.quantized.sum() + out.vq_loss
        loss.backward()


def test_vq_invalid_config_raises():
    from iqgp.planner.vq import VectorQuantizerEMA

    with pytest.raises(ValueError, match="positive"):
        VectorQuantizerEMA(num_embeddings=0, embedding_dim=4)

    with pytest.raises(ValueError, match="positive"):
        VectorQuantizerEMA(num_embeddings=4, embedding_dim=-1)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_l2_distance_non_negative():
    from iqgp.planner.vq import _compute_l2_distance

    x = torch.randn(10, 8)
    y = torch.randn(5, 8)
    dist = _compute_l2_distance(x, y)
    assert (dist >= 0).all()
    assert dist.shape == (10, 5)


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_l2_distance_self_is_zero():
    from iqgp.planner.vq import _compute_l2_distance

    x = torch.randn(4, 6)
    dist = _compute_l2_distance(x, x)
    diagonal = dist.diag()
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5)
