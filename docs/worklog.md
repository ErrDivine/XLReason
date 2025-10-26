# XLReason Development Log

## Overview
This log documents the initial implementation of the **I-QGP synthetic experiment**. The repository now contains a runnable PyTorch stack that mirrors the plan in `docs/plan.md` while remaining lightweight enough for rapid iteration and testing.

## Repository Structure
- `iqgp/`
  - `models/`: backbone encoder, bilingual decoders, and the lexicon projector.
  - `planner/`: cross-lingual fusion, vector-quantized planner, and graph structures.
  - `objectives/`: all training losses (task CE, EMD alignment, InfoNCE, entity/unit agreement, code-switch consistency, language eraser) plus an adversarial head with gradient reversal.
  - `data/`: synthetic dataset generator and batch container used for fast prototyping.
  - `training/`: single-epoch training loop with logging and gradient clipping.
  - `system.py`: high-level assembly that wires the backbone, planner, decoders, projector, and adversary into one module.
- `scripts/train.py`: CLI that reads YAML configs, instantiates the system, and trains on synthetic data.
- `configs/default.yaml`: canonical hyper-parameters for the experiment (model, dataset, training, and loss weights).
- `tests/`: pytest suite covering planner outputs, loss back-propagation, and the training loop.

## Key Components
### Planner (`iqgp/planner`)
- **Cross-attention fusion**: `CrossAttentionFusion` joins EN/zh hidden states using bidirectional attention and gating.
- **Node queries**: a learned query bank attended over fused features to propose K planner nodes.
- **Vector quantizer**: `VectorQuantizerEMA` implements EMA-updated VQ-VAE with commitment and codebook losses.
- **Graph heads**: biaffine edge scorer, argument pointers, and entity/unit pointer heads produce tensorized graph outputs (codes, edges, arguments, QIDs, units, optional evidence scores).

### Objectives (`iqgp/objectives/losses.py`)
- **Task CE** for answers and CoT.
- **EMD alignment** via a batched Sinkhorn solver on lexicon-projected representations.
- **Plan contrast** (InfoNCE) over pooled EN/zh states.
- **Entity/unit agreement** enforcing consistent pointer predictions.
- **Code-switch dropout** comparing original vs perturbed graphs.
- **Language eraser** using a gradient-reversal adversary on bilingual hidden states.
- `LossBundle` aggregates all components with configurable weights.

### System (`iqgp/system.py`)
- `IQGPSystem` now binds a HuggingFace `TransformerBackbone` (auto-loaded via `transformers`) to the planner.
- Automatically matches decoder vocabulary to the tokenizer’s vocabulary size; synthetic fallback remains available for tests via `use_synthetic_backbone`.
- Uses `LanguageAdversary` to discourage language leakage in the encoder.

### Data (`iqgp/data/`)
- `MGSMReasoningDataset` loads paired EN/ZH samples from `juletxara/mgsm`, aligns them by index, and tokenizes with the shared transformer tokenizer.
- Existing `SyntheticReasoningDataset` remains for deterministic unit tests.

### Training (`scripts/train.py` & `iqgp/training/loop.py`)
- YAML-driven configuration includes transformer + dataset parameters (MGSM vs. synthetic fallback).
- Training script instantiates the HuggingFace model/tokenizer and reuses them across epochs, then streams MGSM batches via the new loader.
- Training loop handles logging, gradient clipping, and per-component loss accounting.

## Testing
All tests live under `tests/` and use pytest. Each test skips gracefully when PyTorch is unavailable.

| Test | Purpose |
| --- | --- |
| `tests/test_planner.py` | Validates planner tensor shapes (codes, args, edges, pointers). |
| `tests/test_losses.py` | Ensures combined loss back-propagates without runtime errors. |
| `tests/test_training.py` | Runs a miniature training epoch end-to-end and reports metrics. |

Run the suite with:

```bash
python -m pytest -q
```

## Usage Notes
- Training relies on PyTorch; install `torch` and `pyyaml` before running `scripts/train.py`.
- The synthetic dataset is designed for smoke testing; replace it with real loaders when integrating actual corpora.
- Default config uses modest dimensions suitable for CPU debugging.

## Follow-Up Ideas
1. Integrate real entity linker/unit normalizer outputs in place of random supervision.
2. Add serialization utilities for `InterlinguaGraph` (export to JSON/GraphML).
3. Layer-wise freezing and adapter injection to mimic the full LLM setup.
4. Extend evaluation utilities for bilingual agreement and graph F1 on real data.

## Environment Caveat
The current sandbox lacks PyTorch, so automated tests are skipped during CI execution. Once PyTorch is available, the same tests will execute fully without modification.
