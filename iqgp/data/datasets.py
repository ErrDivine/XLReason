"""Synthetic bilingual dataset utilities used for smoke testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class BilingualExample:
    """Container representing a single bilingual training instance."""

    en_tokens: torch.Tensor
    zh_tokens: torch.Tensor
    en_mask: torch.Tensor
    zh_mask: torch.Tensor
    answer_en: torch.Tensor
    answer_zh: torch.Tensor


class SyntheticBilingualDataset(Dataset[BilingualExample]):
    """Create tiny bilingual batches for unit testing."""

    def __init__(self, length: int, seq_len: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(0)
        self.examples: List[BilingualExample] = []
        for _ in range(length):
            en_tokens = torch.randn(seq_len, hidden_size, generator=rng)
            zh_tokens = torch.randn(seq_len, hidden_size, generator=rng)
            en_mask = torch.ones(seq_len, dtype=torch.bool)
            zh_mask = torch.ones(seq_len, dtype=torch.bool)
            answer_en = torch.randint(0, vocab_size, (seq_len,), generator=rng)
            answer_zh = torch.randint(0, vocab_size, (seq_len,), generator=rng)
            self.examples.append(
                BilingualExample(
                    en_tokens=en_tokens,
                    zh_tokens=zh_tokens,
                    en_mask=en_mask,
                    zh_mask=zh_mask,
                    answer_en=answer_en,
                    answer_zh=answer_zh,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> BilingualExample:
        return self.examples[idx]


def collate_examples(examples: List[BilingualExample]) -> Tuple[torch.Tensor, ...]:
    en_tokens = torch.stack([ex.en_tokens for ex in examples], dim=0)
    zh_tokens = torch.stack([ex.zh_tokens for ex in examples], dim=0)
    en_mask = torch.stack([ex.en_mask for ex in examples], dim=0)
    zh_mask = torch.stack([ex.zh_mask for ex in examples], dim=0)
    answer_en = torch.stack([ex.answer_en for ex in examples], dim=0)
    answer_zh = torch.stack([ex.answer_zh for ex in examples], dim=0)
    return en_tokens, zh_tokens, en_mask, zh_mask, answer_en, answer_zh


__all__ = ["SyntheticBilingualDataset", "BilingualExample", "collate_examples"]
