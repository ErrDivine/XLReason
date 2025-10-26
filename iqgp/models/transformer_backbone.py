"""Wrapper around HuggingFace transformers backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - handled in tests
    raise ImportError("transformers package is required for TransformerBackbone") from exc


@dataclass
class TransformerBackbone:
    model: torch.nn.Module
    tokenizer: "AutoTokenizer"  # type: ignore[name-defined]
    hidden_size: int

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        **model_kwargs,
    ) -> "TransformerBackbone":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            output_hidden_states=False,
            **model_kwargs,
        )
        hidden_size = int(getattr(model.config, "hidden_size"))
        return cls(model=model, tokenizer=tokenizer, hidden_size=hidden_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


__all__ = ["TransformerBackbone"]
