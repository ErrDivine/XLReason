"""MGSM dataset loader producing bilingual batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("datasets package is required for MGSM loading") from exc

from iqgp.data.batch import BilingualBatch


@dataclass
class MGSMConfig:
    dataset_name: str = "juletxara/mgsm"
    languages: Sequence[str] = ("en", "zh")
    split: str = "train"
    max_length: int = 128
    batch_size: int = 2
    max_samples: int | None = 128
    num_entities: int = 32
    num_units: int = 16
    num_nodes: int = 8


def _align_datasets(ds_a, ds_b, max_samples: int | None) -> list[tuple[dict, dict]]:
    if len(ds_a) != len(ds_b):
        limit = min(len(ds_a), len(ds_b))
    else:
        limit = len(ds_a)
    if max_samples is not None:
        limit = min(limit, max_samples)
    paired = []
    for idx in range(limit):
        paired.append((ds_a[idx], ds_b[idx]))
    return paired


class MGSMReasoningDataset:
    """Iterable over MGSM bilingual samples tokenized by a HuggingFace tokenizer."""

    def __init__(
        self,
        config: MGSMConfig,
        tokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        if len(config.languages) != 2:
            raise ValueError("MGSMConfig.languages must contain exactly two language codes")
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        lang_a, lang_b = config.languages
        self.dataset_a = load_dataset(config.dataset_name, lang_a, split=config.split)
        self.dataset_b = load_dataset(config.dataset_name, lang_b, split=config.split)
        self.paired = _align_datasets(self.dataset_a, self.dataset_b, config.max_samples)
        if len(self.paired) == 0:
            raise ValueError("No samples available after alignment; check dataset configuration")

    def __iter__(self) -> Iterator[BilingualBatch]:
        cfg = self.config
        step = 0
        while step < len(self.paired):
            chunk = self.paired[step : step + cfg.batch_size]
            step += cfg.batch_size
            if len(chunk) == 0:
                break

            en_texts = [sample[0]["question"] for sample in chunk]
            zh_texts = [sample[1]["question"] for sample in chunk]
            en_answers = [sample[0].get("answer", "") for sample in chunk]
            zh_answers = [sample[1].get("answer", "") for sample in chunk]
            en_cots = [sample[0].get("solution", "") for sample in chunk]
            zh_cots = [sample[1].get("solution", "") for sample in chunk]

            en_inputs = self.tokenizer(
                en_texts,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            zh_inputs = self.tokenizer(
                zh_texts,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            en_cot_tokens = self.tokenizer(
                en_cots,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            zh_cot_tokens = self.tokenizer(
                zh_cots,
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )

            def answer_ids(texts: list[str]) -> torch.Tensor:
                ids = []
                for text in texts:
                    tokens = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=1,
                        return_tensors="pt",
                    )
                    if tokens.input_ids.numel() == 0:
                        ids.append(torch.tensor([0]))
                    else:
                        ids.append(tokens.input_ids[0, 0].unsqueeze(0))
                return torch.cat(ids, dim=0)

            answers_en = answer_ids(en_answers)
            answers_zh = answer_ids(zh_answers)

            zeros_entities = torch.zeros(len(chunk), cfg.max_length, dtype=torch.long)
            zeros_units = torch.zeros(len(chunk), cfg.max_length, dtype=torch.long)
            plan_entities = torch.zeros(len(chunk), cfg.num_nodes, dtype=torch.long)
            plan_units = torch.zeros(len(chunk), cfg.num_nodes, dtype=torch.long)

            batch = BilingualBatch(
                en_input=None,
                zh_input=None,
                answers_en=answers_en,
                answers_zh=answers_zh,
                cot_en=en_cot_tokens.input_ids,
                cot_zh=zh_cot_tokens.input_ids,
                entities=zeros_entities,
                units=zeros_units,
                plan_entities=plan_entities,
                plan_units=plan_units,
                en_input_ids=en_inputs.input_ids,
                zh_input_ids=zh_inputs.input_ids,
                en_attention_mask=en_inputs.attention_mask,
                zh_attention_mask=zh_inputs.attention_mask,
            )
            yield batch.to(self.device)


__all__ = ["MGSMConfig", "MGSMReasoningDataset"]
