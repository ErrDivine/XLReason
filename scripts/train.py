"""Training script for the XLReason experiment (transformer-powered)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from iqgp.data import (
    MGSMConfig,
    MGSMReasoningDataset,
    SyntheticDatasetConfig,
    SyntheticReasoningDataset,
)
from iqgp.objectives import LossWeights
from iqgp.system import IQGPSystem, ModelConfig
from iqgp.training import train_epoch
from iqgp.utils import ProgressLogger


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(dataset_cfg: Dict[str, Any], system: IQGPSystem, device: torch.device):
    dtype = dataset_cfg.get("type", "synthetic")
    if dtype == "mgsm":
        if MGSMConfig is None or MGSMReasoningDataset is None:
            raise RuntimeError("datasets package is required for MGSM loading")
        mgsm_cfg = MGSMConfig(**dataset_cfg["mgsm"])
        try:
            tokenizer = system.tokenizer
        except AttributeError as exc:
            raise RuntimeError("Transformer backbone required for MGSM dataset") from exc
        return MGSMReasoningDataset(mgsm_cfg, tokenizer, device=device)
    if dtype == "synthetic":
        syn_cfg = SyntheticDatasetConfig(**dataset_cfg.get("synthetic", {}))
        return SyntheticReasoningDataset(syn_cfg, device=device)
    raise ValueError(f"Unknown dataset type: {dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XLReason experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="YAML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    seed = cfg.get("seed", 13)
    _set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = ModelConfig(**cfg["model"])
    loss_weights = LossWeights(**cfg["loss_weights"])
    training_cfg = cfg["training"]
    epochs = args.epochs or training_cfg.get("epochs", 1)

    model = IQGPSystem(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("lr", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )
    logger = ProgressLogger(log_every=training_cfg.get("log_every", 5))
    code_switch_prob = training_cfg.get("code_switch_prob", 0.15)

    dataset = _build_dataset(cfg["dataset"], system=model, device=device)

    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            loss_weights=loss_weights,
            device=device,
            logger=logger,
            code_switch_prob=code_switch_prob,
        )
        print({"epoch": epoch, **metrics})


if __name__ == "__main__":
    main()
