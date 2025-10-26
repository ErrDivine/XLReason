"""Synthetic dataset for quick experimentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import torch

from .batch import BilingualBatch


@dataclass
class SyntheticDatasetConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    num_entities: int = 32
    num_units: int = 16
    seq_len: int = 12
    batch_size: int = 4
    num_batches: int = 50
    num_nodes: int = 8


class SyntheticReasoningDataset:
    def __init__(self, config: SyntheticDatasetConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)

    def __iter__(self) -> Iterator[BilingualBatch]:
        cfg = self.config
        for _ in range(cfg.num_batches):
            en = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, device=self.device)
            zh = torch.randn_like(en)
            cot_en = torch.randint(cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=self.device)
            cot_zh = torch.randint(cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=self.device)
            ans_en = torch.randint(cfg.vocab_size, (cfg.batch_size,), device=self.device)
            ans_zh = torch.randint(cfg.vocab_size, (cfg.batch_size,), device=self.device)
            entities = torch.randint(cfg.num_entities, (cfg.batch_size, cfg.seq_len), device=self.device)
            units = torch.randint(cfg.num_units, (cfg.batch_size, cfg.seq_len), device=self.device)
            plan_entities = torch.randint(cfg.num_entities, (cfg.batch_size, cfg.num_nodes), device=self.device)
            plan_units = torch.randint(cfg.num_units, (cfg.batch_size, cfg.num_nodes), device=self.device)
            yield BilingualBatch(
                en_input=en,
                zh_input=zh,
                answers_en=ans_en,
                answers_zh=ans_zh,
                cot_en=cot_en,
                cot_zh=cot_zh,
                entities=entities,
                units=units,
                plan_entities=plan_entities,
                plan_units=plan_units,
            )


__all__ = ["SyntheticDatasetConfig", "SyntheticReasoningDataset"]
