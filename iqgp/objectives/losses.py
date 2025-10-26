"""Loss functions for the I-QGP experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from iqgp.planner import InterlinguaGraph, PlannerOutput


@dataclass
class LossWeights:
    emd: float = 0.5
    plan: float = 0.5
    entity: float = 0.5
    csd: float = 0.2
    erase: float = 0.2


@dataclass
class LossBundle:
    task: torch.Tensor
    emd: torch.Tensor
    plan_contrast: torch.Tensor
    entity_alignment: torch.Tensor
    code_switch: torch.Tensor
    eraser: torch.Tensor
    vq: torch.Tensor

    def total(self, weights: LossWeights) -> torch.Tensor:
        return (
            self.task
            + weights.emd * self.emd
            + weights.plan * self.plan_contrast
            + weights.entity * self.entity_alignment
            + weights.csd * self.code_switch
            + weights.erase * self.eraser
            + self.vq
        )


class GradientReversalFn(torch.autograd.Function):
    """Simple gradient reversal layer."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[type-arg]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[type-arg]
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.lambd)


class LanguageAdversary(nn.Module):
    """Adversary that predicts language ID from planner states."""

    def __init__(self, hidden_dim: int, num_languages: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_languages),
        )
        self.grad_reverse = GradientReversal()

    def forward(self, planner_states: torch.Tensor) -> torch.Tensor:
        pooled = planner_states.mean(dim=1)
        return self.classifier(self.grad_reverse(pooled))


def _task_loss(
    answer_logits_en: torch.Tensor,
    answer_logits_zh: torch.Tensor,
    answer_targets_en: torch.Tensor,
    answer_targets_zh: torch.Tensor,
    cot_logits_en: Optional[torch.Tensor] = None,
    cot_logits_zh: Optional[torch.Tensor] = None,
    cot_targets_en: Optional[torch.Tensor] = None,
    cot_targets_zh: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.cross_entropy(answer_logits_en, answer_targets_en)
    loss = loss + F.cross_entropy(answer_logits_zh, answer_targets_zh)
    if cot_logits_en is not None and cot_targets_en is not None:
        loss = loss + F.cross_entropy(
            cot_logits_en.reshape(-1, cot_logits_en.size(-1)),
            cot_targets_en.reshape(-1),
        )
    if cot_logits_zh is not None and cot_targets_zh is not None:
        loss = loss + F.cross_entropy(
            cot_logits_zh.reshape(-1, cot_logits_zh.size(-1)),
            cot_targets_zh.reshape(-1),
        )
    return loss


def _sinkhorn(
    p: torch.Tensor,
    q: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.1,
    iterations: int = 20,
) -> torch.Tensor:
    K = torch.exp(-cost / epsilon)
    u = torch.ones_like(p)
    v = torch.ones_like(q)
    for _ in range(iterations):
        u = p / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        v = q / (K.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
    transport = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return (transport * cost).sum(dim=(-2, -1))


def _emd_alignment(projected_en: torch.Tensor, projected_zh: torch.Tensor) -> torch.Tensor:
    bsz, len_en, _ = projected_en.shape
    _, len_zh, _ = projected_zh.shape
    cost = torch.cdist(projected_en, projected_zh, p=2)
    p = torch.full((bsz, len_en), 1.0 / len_en, device=projected_en.device)
    q = torch.full((bsz, len_zh), 1.0 / len_zh, device=projected_en.device)
    return _sinkhorn(p, q, cost)


def _plan_contrast(en_states: torch.Tensor, zh_states: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    en = F.normalize(en_states, p=2, dim=-1)
    zh = F.normalize(zh_states, p=2, dim=-1)
    logits = en @ zh.t() / temperature
    targets = torch.arange(en.size(0), device=en.device)
    loss = F.cross_entropy(logits, targets)
    loss = loss + F.cross_entropy(logits.t(), targets)
    return loss * 0.5


def _entity_unit_loss(
    qid_logits: torch.Tensor,
    unit_logits: torch.Tensor,
    entity_targets: torch.Tensor,
    unit_targets: torch.Tensor,
) -> torch.Tensor:
    ent_loss = F.cross_entropy(
        qid_logits.reshape(-1, qid_logits.size(-1)),
        entity_targets.reshape(-1),
    )
    unit_loss = F.cross_entropy(
        unit_logits.reshape(-1, unit_logits.size(-1)),
        unit_targets.reshape(-1),
    )
    return ent_loss + unit_loss


def _code_switch_loss(base_graph: InterlinguaGraph, switched_graph: Optional[InterlinguaGraph]) -> torch.Tensor:
    if switched_graph is None:
        return torch.tensor(0.0, device=base_graph.codes.device)
    node_loss = F.mse_loss(switched_graph.node_states, base_graph.node_states)
    edge_loss = F.mse_loss(switched_graph.edge_logits, base_graph.edge_logits)
    arg_loss = F.mse_loss(switched_graph.arg_logits, base_graph.arg_logits)
    return node_loss + edge_loss + arg_loss


def _eraser_loss(adversary_logits: Optional[torch.Tensor], language_targets: Optional[torch.Tensor]) -> torch.Tensor:
    if adversary_logits is None or language_targets is None:
        return torch.tensor(0.0, device=(adversary_logits.device if adversary_logits is not None else "cpu"))
    return F.cross_entropy(adversary_logits, language_targets)


def compute_total_loss(
    planner_output: PlannerOutput,
    answer_logits_en: torch.Tensor,
    answer_logits_zh: torch.Tensor,
    answer_targets_en: torch.Tensor,
    answer_targets_zh: torch.Tensor,
    projected_en: torch.Tensor,
    projected_zh: torch.Tensor,
    plan_states_en: torch.Tensor,
    plan_states_zh: torch.Tensor,
    entity_targets: torch.Tensor,
    unit_targets: torch.Tensor,
    loss_weights: LossWeights,
    cot_logits_en: Optional[torch.Tensor] = None,
    cot_logits_zh: Optional[torch.Tensor] = None,
    cot_targets_en: Optional[torch.Tensor] = None,
    cot_targets_zh: Optional[torch.Tensor] = None,
    switched_graph: Optional[InterlinguaGraph] = None,
    adversary_logits: Optional[torch.Tensor] = None,
    language_targets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, LossBundle]:
    graph = planner_output.graph
    task = _task_loss(
        answer_logits_en,
        answer_logits_zh,
        answer_targets_en,
        answer_targets_zh,
        cot_logits_en,
        cot_logits_zh,
        cot_targets_en,
        cot_targets_zh,
    )
    emd = _emd_alignment(projected_en, projected_zh).mean()
    plan_contrast = _plan_contrast(plan_states_en, plan_states_zh)
    entity_alignment = _entity_unit_loss(graph.qid_logits, graph.unit_logits, entity_targets, unit_targets)
    code_switch = _code_switch_loss(graph, switched_graph)
    eraser = _eraser_loss(adversary_logits, language_targets)
    vq = planner_output.vq.vq_loss
    bundle = LossBundle(
        task=task,
        emd=emd,
        plan_contrast=plan_contrast,
        entity_alignment=entity_alignment,
        code_switch=code_switch,
        eraser=eraser,
        vq=vq,
    )
    total = bundle.total(loss_weights)
    return total, bundle


__all__ = [
    "LossBundle",
    "LossWeights",
    "LanguageAdversary",
    "GradientReversal",
    "compute_total_loss",
]
