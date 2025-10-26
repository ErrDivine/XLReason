"""Training loop for the synthetic I-QGP experiment."""

from __future__ import annotations

from typing import Dict

import torch

from iqgp.data import BilingualBatch
from iqgp.objectives import LossBundle, LossWeights, compute_total_loss
from iqgp.system import IQGPSystem
from iqgp.utils import ProgressLogger


def _make_language_targets(batch_size: int, device: torch.device) -> torch.Tensor:
    zeros = torch.zeros(batch_size, dtype=torch.long, device=device)
    ones = torch.ones(batch_size, dtype=torch.long, device=device)
    return torch.cat([zeros, ones], dim=0)


def train_epoch(
    model: IQGPSystem,
    optimizer: torch.optim.Optimizer,
    dataset,
    loss_weights: LossWeights,
    device: torch.device,
    logger: ProgressLogger,
    code_switch_prob: float = 0.15,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    aggregates = {
        "task": 0.0,
        "emd": 0.0,
        "plan": 0.0,
        "entity": 0.0,
        "csd": 0.0,
        "erase": 0.0,
        "vq": 0.0,
    }

    for step, batch in enumerate(dataset, start=1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, code_switch_prob=code_switch_prob)
        language_targets = _make_language_targets(batch.batch_size, device)
        switched_graph = output.switched_planner.graph if output.switched_planner is not None else None
        total, bundle = compute_total_loss(
            planner_output=output.planner,
            answer_logits_en=output.answer_logits_en,
            answer_logits_zh=output.answer_logits_zh,
            answer_targets_en=batch.answers_en,
            answer_targets_zh=batch.answers_zh,
            projected_en=output.projected_en,
            projected_zh=output.projected_zh,
            plan_states_en=output.plan_states_en,
            plan_states_zh=output.plan_states_zh,
            entity_targets=batch.plan_entities,
            unit_targets=batch.plan_units,
            loss_weights=loss_weights,
            cot_logits_en=output.cot_logits_en,
            cot_logits_zh=output.cot_logits_zh,
            cot_targets_en=batch.cot_en,
            cot_targets_zh=batch.cot_zh,
            switched_graph=switched_graph,
            adversary_logits=output.adversary_logits,
            language_targets=language_targets,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += total.item()
        total_batches += 1
        aggregates["task"] += bundle.task.item()
        aggregates["emd"] += bundle.emd.item()
        aggregates["plan"] += bundle.plan_contrast.item()
        aggregates["entity"] += bundle.entity_alignment.item()
        aggregates["csd"] += bundle.code_switch.item()
        aggregates["erase"] += bundle.eraser.item()
        aggregates["vq"] += bundle.vq.item()

        logger.log(step, "train", {
            "loss": total.item(),
            "task": bundle.task.item(),
            "emd": bundle.emd.item(),
        })

    if total_batches == 0:
        return {key: 0.0 for key in ["loss", *aggregates.keys()]}

    metrics = {key: value / total_batches for key, value in aggregates.items()}
    metrics["loss"] = total_loss / total_batches
    return metrics


__all__ = ["train_epoch"]
