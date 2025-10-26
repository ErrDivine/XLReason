"""Batch utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BilingualBatch:
    en_input: torch.Tensor
    zh_input: torch.Tensor
    answers_en: torch.Tensor
    answers_zh: torch.Tensor
    cot_en: torch.Tensor
    cot_zh: torch.Tensor
    entities: torch.Tensor
    units: torch.Tensor
    plan_entities: torch.Tensor
    plan_units: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.en_input.shape[0])

    def to(self, device: torch.device | str) -> "BilingualBatch":
        kwargs: Dict[str, torch.Tensor] = {
            "en_input": self.en_input.to(device),
            "zh_input": self.zh_input.to(device),
            "answers_en": self.answers_en.to(device),
            "answers_zh": self.answers_zh.to(device),
            "cot_en": self.cot_en.to(device),
            "cot_zh": self.cot_zh.to(device),
            "entities": self.entities.to(device),
            "units": self.units.to(device),
            "plan_entities": self.plan_entities.to(device),
            "plan_units": self.plan_units.to(device),
        }
        return BilingualBatch(**kwargs)


__all__ = ["BilingualBatch"]
