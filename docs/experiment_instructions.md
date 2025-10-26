# I-QGP Experiment Instructions

This guide walks you through preparing the environment, validating the installation, and running the transformer-powered I-QGP experiment. Follow the steps in order; each stage builds on the previous one.

---

## 1. Prerequisites
- **Python**: 3.10 or newer (3.11 recommended).
- **Operating system**: macOS, Linux, or Windows with WSL. The quickstart assumes shell access.
- **Dependencies**: PyTorch (CPU or CUDA build), `transformers`, `datasets`, PyYAML, NumPy, and pytest for development/test runs.

```bash
python --version  # Should print 3.10+
```

> If you do not already have Python 3.10+, install it before proceeding (e.g., via `pyenv`, `conda`, or system packages).

---

## 2. Clone & Enter the Project
```bash
cd /path/to/workspace
git clone <repo-url> XLReason
cd XLReason
```

If the repository already exists, ensure you are on the branch that contains the latest experiment code.

---

## 3. Create & Activate a Virtual Environment
Use your preferred environment manager; the example below uses `python -m venv`.

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# or
.venv\Scripts\activate       # Windows PowerShell
```

Keep the environment active for the remaining steps.

---

## 4. Install Project Dependencies
1. **Install PyTorch** – choose the wheel that matches your platform and CUDA version (CPU example shown).
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   For GPU builds, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

2. **Install transformer & datasets tooling**
   ```bash
   pip install transformers datasets
   ```

3. **Install the project (and dev extras)**
   ```bash
   pip install -e .[dev]
   ```

4. **Verify the installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   If this import fails, revisit Step 4.1.

---

## 5. Inspect Configuration (Optional)
Default settings live in `configs/default.yaml`. Adjust model size, dataset dimensions, or training hyperparameters as needed before running experiments.

Key sections:
- `model`: transformer model identifiers and planner hyperparameters. By default this references **Qwen/Qwen2.5-Math-7B**; adjust to a smaller model if GPU memory is limited.
- `dataset`: choose `type: mgsm` (default) for real bilingual data or `type: synthetic` when you want the lightweight mock pipeline. MGSM settings include languages, split, max sequence length, and sample budget.
- `training`: optimizer settings, epochs, logging cadence, and code-switch probability.
- `loss_weights`: coefficients for the auxiliary objectives.

---

## 6. Run the Test Suite
Ensure the system is healthy before training.

```bash
python -m pytest -q
```

- All tests should pass. If PyTorch is missing, the suite reports skips—rehydrate Step 4.
- Tests cover planner tensor shapes, loss backpropagation, and an end-to-end training smoke test.

---

## 7. Launch the Training Run (MGSM by default)
Execute the training script (uses the YAML config by default):

```bash
python scripts/train.py --epochs 1 --device cpu
```

Notes:
- Use `--device cuda` when running on a GPU-enabled machine.
- Adjust `--config` to point at a custom YAML file if you created one.
- When `dataset.type` is `mgsm`, the loader pulls paired English–Chinese items from `juletxara/mgsm` using the shared transformer tokenizer.
- To run the synthetic toy pipeline instead, set `dataset.type: synthetic` and `model.use_synthetic_backbone: true` in your config.
- Training logs (loss, task, emd) stream to stdout via `ProgressLogger` every `log_every` steps.

---

## 8. Inspect Outputs
- Console output shows per-epoch metrics (aggregate loss components).
- Modify `docs/worklog.md` or other reports as you iterate to track experiment changes.

---

## 9. Next Steps
- Swap the synthetic dataset with real bilingual data loaders once available.
- Extend tests for dataset loading and evaluation metrics in `iqgp/utils/`.
- Export `InterlinguaGraph` structures for downstream inspection or visualization.

---

## Troubleshooting Checklist
| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | Re-run Step 4.1 with the correct PyTorch wheel. |
| `ModuleNotFoundError: No module named 'transformers'` | Execute Step 4.2. |
| `datasets.builder.DatasetNotFoundError` | Ensure you spelled the dataset/config correctly (defaults to `juletxara/mgsm`). |
| Tests skipped because PyTorch missing | Same as above—install PyTorch, then rerun `pytest`. |
| CUDA runtime errors | Ensure the PyTorch install matches your CUDA toolkit or fall back to CPU build. |
| Slow CPU training | Reduce `num_nodes`, `embedding_dim`, or `batch_size` in the config. |

Keep this guide updated as the project evolves, especially once real datasets and larger backbones are integrated.
