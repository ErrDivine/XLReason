import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is required for training loop test")
def test_single_epoch_training_runs_without_error():
    from iqgp.data import SyntheticDatasetConfig, SyntheticReasoningDataset
    from iqgp.objectives import LossWeights
    from iqgp.system import IQGPSystem, ModelConfig
    from iqgp.training import train_epoch
    from iqgp.utils import ProgressLogger

    torch.manual_seed(3)
    model_cfg = ModelConfig(
        hidden_size=32,
        vocab_size=40,
        num_entities=12,
        num_units=9,
        num_nodes=5,
        codebook_size=16,
        embedding_dim=32,
        num_edge_types=3,
        projection_dim=24,
    )
    system = IQGPSystem(model_cfg)
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-3)
    dataset_cfg = SyntheticDatasetConfig(
        vocab_size=40,
        hidden_size=32,
        num_entities=12,
        num_units=9,
        seq_len=6,
        batch_size=2,
        num_batches=3,
        num_nodes=5,
    )
    dataset = SyntheticReasoningDataset(dataset_cfg)
    metrics = train_epoch(
        model=system,
        optimizer=optimizer,
        dataset=dataset,
        loss_weights=LossWeights(),
        device=torch.device("cpu"),
        logger=ProgressLogger(log_every=100),
        code_switch_prob=0.0,
    )
    assert "loss" in metrics
    for key, value in metrics.items():
        assert isinstance(value, float)
        assert value == pytest.approx(value)
