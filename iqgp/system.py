"""End-to-end I-QGP system assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from iqgp.models import AnswerDecoder, BilingualBackbone, CoTDecoder, LexiconProjector
from iqgp.objectives import LanguageAdversary
from iqgp.planner import InterlinguaPlanner, PlannerOutput


@dataclass
class ModelConfig:
    hidden_size: int
    vocab_size: int
    num_entities: int
    num_units: int
    num_nodes: int
    codebook_size: int
    embedding_dim: int
    num_edge_types: int = 4
    commitment_cost: float = 0.25
    attn_heads: int = 4
    projection_dim: int = 128


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = BilingualBackbone(config.hidden_size)
        self.planner = InterlinguaPlanner(
            hidden_size=config.hidden_size,
            num_nodes=config.num_nodes,
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            num_entities=config.num_entities,
            num_units=config.num_units,
            num_edge_types=config.num_edge_types,
            commitment_cost=config.commitment_cost,
            attn_heads=config.attn_heads,
        )
        self.en_answer = AnswerDecoder(config.embedding_dim, config.vocab_size)
        self.zh_answer = AnswerDecoder(config.embedding_dim, config.vocab_size)
        self.en_cot = CoTDecoder(config.hidden_size, config.vocab_size)
        self.zh_cot = CoTDecoder(config.hidden_size, config.vocab_size)
        self.projector = LexiconProjector(config.vocab_size, config.projection_dim)
        self.language_adversary = LanguageAdversary(config.embedding_dim)

    def _decode_answers(self, planner_output: PlannerOutput) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = planner_output.graph.node_states.mean(dim=1)
        return self.en_answer(pooled).logits, self.zh_answer(pooled).logits

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
        en_hidden, zh_hidden = self.backbone(batch.en_input, batch.zh_input)
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
