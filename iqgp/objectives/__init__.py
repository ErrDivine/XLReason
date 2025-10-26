"""Objective exports."""
from .adversary import LanguageAdversary, grad_reverse
from .lexicon import LexiconProjector
from .losses import (
    SinkhornDistance,
    code_switch_consistency,
    emd_bilingual,
    entity_unit_agreement,
    info_nce_loss,
    language_eraser_loss,
)

__all__ = [
    "LanguageAdversary",
    "LexiconProjector",
    "SinkhornDistance",
    "code_switch_consistency",
    "emd_bilingual",
    "entity_unit_agreement",
    "grad_reverse",
    "info_nce_loss",
    "language_eraser_loss",
]
