# Repository Guidelines

## Project Structure & Module Organization
- `iqgp/`: Core Python package; submodules cover planners, models, objectives, data loaders, training, and utilities.
- `scripts/`: Entry points (e.g., `train.py`) for running experiments.
- `configs/`: YAML configuration files; `default.yaml` binds transformer + MGSM settings.
- `docs/`: Long-form documentation (`plan.md`, `worklog.md`, experiment instructions).
- `tests/`: Pytest suite validating planner shapes, loss wiring, and training loop behaviors.

## Build, Test, and Development Commands
- `pip install -e .[dev]`: Install project with development extras (pytest, coverage).
- `python scripts/train.py --config configs/default.yaml --device cuda`: Run MGSM training with configured transformer backbone.
- `python -m pytest -q`: Execute all tests (skips when PyTorch or HF stack missing).

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indentation.
- Prefer descriptive module names (e.g., `transformer_backbone.py`, `losses.py`).
- Inline comments only when clarifying non-obvious logic; docstrings for public classes/functions.
- No auto-format tool enforced; keep imports grouped (stdlib, third-party, local).

## Testing Guidelines
- Framework: `pytest`; synthetic datasets ensure deterministic smoke coverage.
- Name tests `test_<feature>.py`; individual tests start with `test_`.
- Run `python -m pytest -q` before pushing; ensure new modules include focused unit tests (planner heads, loss components, data loaders).

## Commit & Pull Request Guidelines
- Commit messages: `<scope>: <summary>` (e.g., `planner: add transformer backbone`). Use present tense and keep summaries under ~72 chars.
- Include descriptive PR titles, summary of changes, testing evidence (`python -m pytest -q`), and link relevant issues.
- Highlight any model/dataset changes impacting reproducibility; attach config diffs or CLI commands for reviewers.

## Security & Configuration Tips
- Store HuggingFace tokens in environment variables; avoid committing credentials.
- Configure `model.hf_model_name` and `dataset` blocks via YAML overrides rather than editing source files.
- For large checkpoints, rely on cache directories (`hf_cache_dir`) instead of vendoring weights into the repo.
