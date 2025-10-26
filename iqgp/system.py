"""End-to-end I-QGP system assembly leveraging transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from iqgp.models import (
    AnswerDecoder,
    BilingualBackbone,
    CoTDecoder,
    LexiconProjector,
    TransformerBackbone,
)
from iqgp.objectives import LanguageAdversary
from iqgp.planner import InterlinguaPlanner, PlannerOutput


@dataclass
class ModelConfig:
    vocab_size: Optional[int] = None
    num_entities: int
    num_units: int
    num_nodes: int
    codebook_size: int
    embedding_dim: int
    num_edge_types: int = 4
    commitment_cost: float = 0.25
    attn_heads: int = 4
    projection_dim: int = 128
    hidden_size: Optional[int] = None  # only used when fallback backbone is enabled
    hf_model_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    hf_trust_remote_code: bool = False
    hf_use_fast_tokenizer: bool = True
    use_synthetic_backbone: bool = False


@dataclass
class SystemOutput:
    planner: PlannerOutput
    answer_logits_en: torch.Tensor
    answer_logits_zh: torch.Tensor
    cot_logits_en: torch.Tensor
    cot_logits_zh: torch.Tensor
    projected_en: torch.Tensor
    projected_zh: torch.Tensor
    plan_states_en: torch.Tensor
    plan_states_zh: torch.Tensor
    adversary_logits: torch.Tensor
    switched_planner: Optional[PlannerOutput]


class IQGPSystem(nn.Module):
    """Main model tying together backbone, planner, decoders, and projector."""

    def __init__(self, config: ModelConfig, encoder: Optional[TransformerBackbone] = None) -> None:
        super().__init__()
        self.config = config
        self.use_transformer = not config.use_synthetic_backbone
        self.transformer_backbone: Optional[TransformerBackbone] = None
        self.simple_backbone: Optional[BilingualBackbone] = None
        hidden_size: int

        if self.use_transformer:
            if encoder is not None:
                self.transformer_backbone = encoder
            else:
                if config.hf_model_name is None:
                    raise ValueError("hf_model_name must be provided when use_synthetic_backbone is False")
                self.transformer_backbone = TransformerBackbone.from_pretrained(
                    model_name=config.hf_model_name,
                    tokenizer_name=config.hf_tokenizer_name,
                    revision=config.hf_revision,
                    cache_dir=config.hf_cache_dir,
                    trust_remote_code=config.hf_trust_remote_code,
                    use_fast_tokenizer=config.hf_use_fast_tokenizer,
                )
            hidden_size = self.transformer_backbone.hidden_size  # type: ignore[assignment]
        else:
            if config.hidden_size is None:
                raise ValueError("hidden_size must be set when use_synthetic_backbone is True")
            hidden_size = config.hidden_size
            self.simple_backbone = BilingualBackbone(hidden_size)

        if self.use_transformer:
            tokenizer_vocab = int(self.transformer_backbone.tokenizer.vocab_size)  # type: ignore[union-attr]
            vocab_size = tokenizer_vocab if config.vocab_size is None else config.vocab_size
        else:
            if config.vocab_size is None:
                raise ValueError("vocab_size must be provided when using the synthetic backbone")
            vocab_size = config.vocab_size

        self.vocab_size = vocab_size

        self.planner = InterlinguaPlanner(
            hidden_size=hidden_size,
            num_nodes=config.num_nodes,
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            num_entities=config.num_entities,
            num_units=config.num_units,
            num_edge_types=config.num_edge_types,
            commitment_cost=config.commitment_cost,
            attn_heads=config.attn_heads,
        )
        self.en_answer = AnswerDecoder(config.embedding_dim, self.vocab_size)
        self.zh_answer = AnswerDecoder(config.embedding_dim, self.vocab_size)
        self.en_cot = CoTDecoder(hidden_size, self.vocab_size)
        self.zh_cot = CoTDecoder(hidden_size, self.vocab_size)
        self.projector = LexiconProjector(self.vocab_size, config.projection_dim)
        self.language_adversary = LanguageAdversary(config.embedding_dim)

    @property
    def tokenizer(self):
        if not self.use_transformer:
            raise AttributeError("Tokenizer is only available when using the transformer backbone")
        if self.transformer_backbone is None:
            raise RuntimeError("Transformer backbone was not initialized")
        return self.transformer_backbone.tokenizer

    def _decode_answers(self, planner_output: PlannerOutput) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = planner_output.graph.node_states.mean(dim=1)
        return self.en_answer(pooled).logits, self.zh_answer(pooled).logits

    def _encode(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_transformer:
            if self.transformer_backbone is None:
                raise RuntimeError("Transformer backbone is missing")
            if batch.en_input_ids is None or batch.zh_input_ids is None:
                raise ValueError("Tokenized inputs required when using transformer backbone")
            en_hidden = self.transformer_backbone.encode(batch.en_input_ids, batch.en_attention_mask)
            zh_hidden = self.transformer_backbone.encode(batch.zh_input_ids, batch.zh_attention_mask)
            return en_hidden, zh_hidden
        if self.simple_backbone is None:
            raise RuntimeError("Simple backbone is missing")
        if batch.en_input is None or batch.zh_input is None:
            raise ValueError("Dense inputs required when using synthetic backbone")
        outputs = self.simple_backbone(batch.en_input, batch.zh_input)
        return outputs.en_hidden, outputs.zh_hidden

    def _make_code_switched_hidden(self, en_hidden: torch.Tensor, zh_hidden: torch.Tensor, prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        if prob <= 0:
            return en_hidden, zh_hidden
        mask = torch.rand_like(en_hidden[..., 0:1]) < prob
        switched_en = torch.where(mask, zh_hidden, en_hidden)
        switched_zh = torch.where(mask, en_hidden, zh_hidden)
        return switched_en, switched_zh

    def forward(
        self,
        batch,
        code_switch_prob: float = 0.0,
    ) -> SystemOutput:
        en_hidden, zh_hidden = self._encode(batch)
        planner_output = self.planner(en_hidden, zh_hidden)
        answer_en, answer_zh = self._decode_answers(planner_output)
        cot_en = self.en_cot(en_hidden).logits
        cot_zh = self.zh_cot(zh_hidden).logits
        projected_en = self.projector(cot_en)
        projected_zh = self.projector(cot_zh)
        plan_states_en = en_hidden.mean(dim=1)
        plan_states_zh = zh_hidden.mean(dim=1)

        adv_states = torch.cat([en_hidden, zh_hidden], dim=0)
        adversary_logits = self.language_adversary(adv_states)

        switched_planner: Optional[PlannerOutput] = None
        if code_switch_prob > 0:
            switched_en, switched_zh = self._make_code_switched_hidden(en_hidden, zh_hidden, code_switch_prob)
            switched_planner = self.planner(switched_en, switched_zh)

        return SystemOutput(
            planner=planner_output,
            answer_logits_en=answer_en,
            answer_logits_zh=answer_zh,
            cot_logits_en=cot_en,
            cot_logits_zh=cot_zh,
            projected_en=projected_en,
            projected_zh=projected_zh,
            plan_states_en=plan_states_en,
            plan_states_zh=plan_states_zh,
            adversary_logits=adversary_logits,
            switched_planner=switched_planner,
        )


__all__ = ["ModelConfig", "IQGPSystem", "SystemOutput"]
