import pytest

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_batch_size_from_dense_input():
    from iqgp.data import BilingualBatch

    batch = BilingualBatch(
        en_input=torch.randn(4, 6, 16),
        zh_input=torch.randn(4, 6, 16),
        answers_en=torch.zeros(4, dtype=torch.long),
        answers_zh=torch.zeros(4, dtype=torch.long),
        cot_en=torch.zeros(4, 6, dtype=torch.long),
        cot_zh=torch.zeros(4, 6, dtype=torch.long),
        entities=torch.zeros(4, 6, dtype=torch.long),
        units=torch.zeros(4, 6, dtype=torch.long),
        plan_entities=torch.zeros(4, 4, dtype=torch.long),
        plan_units=torch.zeros(4, 4, dtype=torch.long),
    )
    assert batch.batch_size == 4


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_batch_size_from_token_ids():
    from iqgp.data import BilingualBatch

    batch = BilingualBatch(
        en_input=None,
        zh_input=None,
        answers_en=torch.zeros(3, dtype=torch.long),
        answers_zh=torch.zeros(3, dtype=torch.long),
        cot_en=torch.zeros(3, 5, dtype=torch.long),
        cot_zh=torch.zeros(3, 5, dtype=torch.long),
        entities=torch.zeros(3, 5, dtype=torch.long),
        units=torch.zeros(3, 5, dtype=torch.long),
        plan_entities=torch.zeros(3, 4, dtype=torch.long),
        plan_units=torch.zeros(3, 4, dtype=torch.long),
        en_input_ids=torch.zeros(3, 5, dtype=torch.long),
        zh_input_ids=torch.zeros(3, 5, dtype=torch.long),
    )
    assert batch.batch_size == 3


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_batch_size_missing_raises():
    from iqgp.data import BilingualBatch

    batch = BilingualBatch(
        en_input=None,
        zh_input=None,
        answers_en=torch.zeros(2, dtype=torch.long),
        answers_zh=torch.zeros(2, dtype=torch.long),
        cot_en=torch.zeros(2, 5, dtype=torch.long),
        cot_zh=torch.zeros(2, 5, dtype=torch.long),
        entities=torch.zeros(2, 5, dtype=torch.long),
        units=torch.zeros(2, 5, dtype=torch.long),
        plan_entities=torch.zeros(2, 4, dtype=torch.long),
        plan_units=torch.zeros(2, 4, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="missing both"):
        _ = batch.batch_size


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_batch_to_device():
    from iqgp.data import BilingualBatch

    batch = BilingualBatch(
        en_input=torch.randn(2, 4, 8),
        zh_input=torch.randn(2, 4, 8),
        answers_en=torch.zeros(2, dtype=torch.long),
        answers_zh=torch.zeros(2, dtype=torch.long),
        cot_en=torch.zeros(2, 4, dtype=torch.long),
        cot_zh=torch.zeros(2, 4, dtype=torch.long),
        entities=torch.zeros(2, 4, dtype=torch.long),
        units=torch.zeros(2, 4, dtype=torch.long),
        plan_entities=torch.zeros(2, 3, dtype=torch.long),
        plan_units=torch.zeros(2, 3, dtype=torch.long),
    )
    moved = batch.to("cpu")
    assert moved.batch_size == 2
    assert moved.en_input is not None
    assert moved.en_input.device.type == "cpu"


@pytest.mark.skipif(torch is None, reason="PyTorch is required")
def test_synthetic_dataset_yields_correct_batches():
    from iqgp.data import SyntheticDatasetConfig, SyntheticReasoningDataset

    cfg = SyntheticDatasetConfig(
        vocab_size=32,
        hidden_size=16,
        num_entities=8,
        num_units=4,
        seq_len=5,
        batch_size=3,
        num_batches=2,
        num_nodes=4,
    )
    ds = SyntheticReasoningDataset(cfg)
    batches = list(ds)
    assert len(batches) == 2
    for batch in batches:
        assert batch.batch_size == 3
        assert batch.en_input.shape == (3, 5, 16)
        assert batch.answers_en.shape == (3,)
        assert batch.plan_entities.shape == (3, 4)
