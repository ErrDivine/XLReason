"""Evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import torch


def bilingual_agreement(en_answers: Sequence[str], zh_answers: Sequence[str]) -> float:
    total = max(len(en_answers), 1)
    matches = sum(a == b for a, b in zip(en_answers, zh_answers))
    return matches / total


def graph_f1(pred_edges: torch.Tensor, gold_edges: torch.Tensor) -> float:
    pred = (pred_edges > 0).float()
    gold = (gold_edges > 0).float()
    tp = (pred * gold).sum().item()
    fp = ((pred == 1) * (gold == 0)).sum().item()
    fn = ((pred == 0) * (gold == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def plan_stability(graph_a: torch.Tensor, graph_b: torch.Tensor) -> float:
    diff = torch.abs(graph_a - graph_b).mean().item()
    return float(1.0 / (1.0 + diff))


__all__ = ["bilingual_agreement", "graph_f1", "plan_stability"]
