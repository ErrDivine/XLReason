"""Training objectives for I-QGP."""

from .losses import LossBundle, LossWeights, LanguageAdversary, compute_total_loss

__all__ = [
    "LossBundle",
    "LossWeights",
    "LanguageAdversary",
    "compute_total_loss",
]
