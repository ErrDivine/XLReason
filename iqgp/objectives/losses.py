"""Losses for bilingual reasoning objectives."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDistance(nn.Module):
    """Approximate earth mover distance via Sinkhorn iterations."""

    def __init__(self, epsilon: float = 0.1, max_iter: int = 50, reduction: str = "mean") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("Sinkhorn inputs must be of shape (B, L, D)")
        cost = torch.cdist(x, y, p=2)
        batch, len_x, len_y = cost.shape
        mu = torch.full((batch, len_x), 1.0 / len_x, device=x.device, dtype=x.dtype)
        nu = torch.full((batch, len_y), 1.0 / len_y, device=y.device, dtype=y.dtype)

        log_u = torch.zeros_like(mu)
        log_v = torch.zeros_like(nu)
        log_mu = mu.log()
        log_nu = nu.log()
        log_k = -cost / self.epsilon

        for _ in range(self.max_iter):
            log_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
            log_v = log_nu - torch.logsumexp(log_k.transpose(1, 2) + log_u.unsqueeze(1), dim=2)

        u = log_u.exp()
        v = log_v.exp()
        transport = u.unsqueeze(-1) * (log_k.exp()) * v.unsqueeze(-2)
        distance = (transport * cost).sum(dim=(1, 2))
        if self.reduction == "mean":
            return distance.mean()
        if self.reduction == "sum":
            return distance.sum()
        return distance


def emd_bilingual(en_logits: torch.Tensor, zh_logits: torch.Tensor, projector: nn.Module) -> torch.Tensor:
    """Compute lexicon-projected EMD alignment between English and Chinese logits."""
    projected_en = projector(en_logits)
    projected_zh = projector(zh_logits)
    sinkhorn = SinkhornDistance()
    return sinkhorn(projected_en, projected_zh) + sinkhorn(projected_zh, projected_en)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if anchor.shape != positive.shape:
        raise ValueError("Anchor and positive must share the same shape")
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    batch = anchor.shape[0]
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(batch, device=anchor.device)
    return F.cross_entropy(logits, labels)


def entity_unit_agreement(qid_en: torch.Tensor, qid_zh: torch.Tensor, unit_en: torch.Tensor, unit_zh: torch.Tensor) -> torch.Tensor:
    qid_loss = F.cross_entropy(qid_en, qid_zh.argmax(dim=-1))
    unit_loss = F.cross_entropy(unit_en, unit_zh.argmax(dim=-1))
    return qid_loss + unit_loss


def code_switch_consistency(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]
    return F.cross_entropy(predictions, targets)


def language_eraser_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


__all__ = [
    "SinkhornDistance",
    "emd_bilingual",
    "info_nce_loss",
    "entity_unit_agreement",
    "code_switch_consistency",
    "language_eraser_loss",
]
