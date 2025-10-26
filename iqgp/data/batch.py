"""Batch utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class BilingualBatch:
    en_input: Optional[torch.Tensor]
    zh_input: Optional[torch.Tensor]
    answers_en: torch.Tensor
    answers_zh: torch.Tensor
    cot_en: torch.Tensor
    cot_zh: torch.Tensor
    entities: torch.Tensor
    units: torch.Tensor
    plan_entities: torch.Tensor
    plan_units: torch.Tensor
    en_input_ids: Optional[torch.Tensor] = None
    zh_input_ids: Optional[torch.Tensor] = None
    en_attention_mask: Optional[torch.Tensor] = None
    zh_attention_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        if self.en_input is not None:
            return int(self.en_input.shape[0])
        if self.en_input_ids is not None:
            return int(self.en_input_ids.shape[0])
        raise ValueError("Batch is missing both dense inputs and token IDs.")

    def to(self, device: torch.device | str) -> "BilingualBatch":
        def maybe(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.to(device)

        kwargs: Dict[str, Optional[torch.Tensor] | torch.Tensor] = {
            "en_input": maybe(self.en_input),
            "zh_input": maybe(self.zh_input),
            "answers_en": self.answers_en.to(device),
            "answers_zh": self.answers_zh.to(device),
            "cot_en": self.cot_en.to(device),
            "cot_zh": self.cot_zh.to(device),
            "entities": self.entities.to(device),
            "units": self.units.to(device),
            "plan_entities": self.plan_entities.to(device),
            "plan_units": self.plan_units.to(device),
            "en_input_ids": maybe(self.en_input_ids),
            "zh_input_ids": maybe(self.zh_input_ids),
            "en_attention_mask": maybe(self.en_attention_mask),
            "zh_attention_mask": maybe(self.zh_attention_mask),
        }
        return BilingualBatch(**kwargs)


__all__ = ["BilingualBatch"]
