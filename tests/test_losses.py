"""Unit tests for the bilingual loss helpers."""

import pytest


torch = pytest.importorskip("torch")

from iqgp.objectives.lexicon import LexiconProjector
from iqgp.objectives.losses import (
    SinkhornDistance,
    code_switch_consistency,
    emd_bilingual,
    entity_unit_agreement,
    info_nce_loss,
    language_eraser_loss,
)


def test_emd_loss_runs():
    torch.manual_seed(0)
    batch, seq_len, vocab, proj_dim = 2, 4, 10, 6
    logits_en = torch.randn(batch, seq_len, vocab)
    logits_zh = torch.randn(batch, seq_len, vocab)
    projector = LexiconProjector(vocab, proj_dim)
    loss = emd_bilingual(logits_en, logits_zh, projector)
    assert loss.item() >= 0


def test_contrastive_and_agreement_losses():
    torch.manual_seed(1)
    batch, hidden, classes = 3, 12, 7
    anchor = torch.randn(batch, hidden)
    positive = torch.randn(batch, hidden)
    contrast = info_nce_loss(anchor, positive)
    assert torch.isfinite(contrast)

    qid_en = torch.randn(batch, classes)
    qid_zh = torch.randn(batch, classes)
    unit_en = torch.randn(batch, classes)
    unit_zh = torch.randn(batch, classes)
    agreement = entity_unit_agreement(qid_en, qid_zh, unit_en, unit_zh)
    assert torch.isfinite(agreement)

    qid_en_3d = torch.randn(batch, 2, classes)
    qid_zh_3d = torch.randn(batch, 2, classes)
    unit_en_3d = torch.randn(batch, 2, classes)
    unit_zh_3d = torch.randn(batch, 2, classes)
    agreement_3d = entity_unit_agreement(qid_en_3d, qid_zh_3d, unit_en_3d, unit_zh_3d)
    assert torch.isfinite(agreement_3d)


def test_language_eraser_and_csd():
    torch.manual_seed(2)
    num_samples, classes = 5, 8
    preds = torch.randn(num_samples, classes)
    targets = torch.randint(0, classes, (num_samples,))
    csd = code_switch_consistency(preds, targets)
    assert torch.isfinite(csd)

    logits = torch.randn(num_samples, 2)
    labels = torch.randint(0, 2, (num_samples,))
    erase = language_eraser_loss(logits, labels)
    assert torch.isfinite(erase)

    mask = torch.tensor([True, False, True, False, True])
    csd_masked = code_switch_consistency(preds, targets, mask=mask)
    assert torch.isfinite(csd_masked)

    empty_mask = torch.zeros(num_samples, dtype=torch.bool)
    csd_empty = code_switch_consistency(preds, targets, mask=empty_mask)
    assert csd_empty.item() == pytest.approx(0.0)


def test_sinkhorn_distance_symmetry():
    torch.manual_seed(3)
    sinkhorn = SinkhornDistance()
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    dist_xy = sinkhorn(x, y)
    dist_yx = sinkhorn(y, x)
    assert torch.isfinite(dist_xy)
    assert torch.isfinite(dist_yx)
