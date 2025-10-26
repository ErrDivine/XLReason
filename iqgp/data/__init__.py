"""Data utilities for I-QGP."""

from .batch import BilingualBatch
from .synthetic import SyntheticDatasetConfig, SyntheticReasoningDataset

__all__ = [
    "BilingualBatch",
    "SyntheticDatasetConfig",
    "SyntheticReasoningDataset",
]

try:  # optional dependency on datasets
    from .mgsm import MGSMConfig, MGSMReasoningDataset
except ImportError:  # pragma: no cover
    MGSMConfig = None  # type: ignore
    MGSMReasoningDataset = None  # type: ignore
else:
    __all__.extend(["MGSMConfig", "MGSMReasoningDataset"])
