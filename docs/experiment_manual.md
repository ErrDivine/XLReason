# I-QGP Experiment Implementation Manual

This manual explains how to operationalize the Interlingua QID-Graph Planner (I-QGP) experiment that lives in this repository. It consolidates the repository structure, required tooling, and the end-to-end workflow for preparing data, training, and validating the bilingual reasoning system. The instructions assume familiarity with PyTorch and bilingual NLP workflows.

## 1. Repository Overview

The repository is organized to mirror the pipeline described in `docs/plan.md`:

- `iqgp/planner/`: graph container and the `PlannerIQGP` module that fuses bilingual encodings and emits discrete plans.
- `iqgp/models/`: shared transformer backbone, language adapters/decoders, and the `BilingualReasoner` wrapper that orchestrates planning and decoding.
- `iqgp/objectives/`: lexicon projector, Sinkhorn-based EMD alignment, InfoNCE, entity/unit agreement, code-switch consistency, and language-erasure utilities.
- `iqgp/data/`: synthetic bilingual dataset helpers for smoke tests; replace with real corpora for full experiments.
- `iqgp/utils/`: vector-quantization helpers used by the planner.
- `scripts/train.py`: illustrative training loop wiring all components with the loss suite.
- `tests/`: unit and integration tests that cover planner outputs, losses, and the full reasoning stack.

Refer to `docs/plan.md` for the conceptual blueprint and `docs/progress.md` for the historical implementation log before extending any module.

## 2. Environment Preparation

1. **Create a virtual environment** (conda or venv) and activate it.
2. **Install dependencies**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt  # if you curate a requirements file
   ```
   The core code only depends on PyTorch; optional packages (e.g., for real entity linking) should be added as needed.
3. **Verify PyTorch availability** by running `python -c "import torch; print(torch.__version__)"`.
4. **Optional GPU support**: install the CUDA-enabled wheel that matches your system if you plan to train on GPUs.

## 3. Data Requirements and Preparation

The repository ships with a synthetic dataset for tests. For real experiments, prepare the following assets:

- **Paired bilingual reasoning data**: English/Chinese questions with aligned answers and (if available) chain-of-thought supervision.
- **Entity linking & unit normalization**: pre-compute candidate QID embeddings and unit embeddings to feed into the planner. The placeholders in `scripts/train.py` demonstrate the expected shapes: `(batch, num_candidates, hidden)` with an accompanying boolean mask.
- **Vocabulary statistics**: vocabulary sizes for each language to instantiate decoders and the lexicon projector.

Recommended workflow:

1. **Tokenize and encode** both languages with a shared backbone (or language-specific tokenizers) and store tensors shaped `(batch, seq_len, hidden)`.
2. **Generate entity/unit embeddings** per instance. Align them with the planner’s `max_nodes` budget and provide boolean masks to denote valid candidates.
3. **Persist datasets** in a format consumable by a custom `torch.utils.data.Dataset`. You can model the interface on `SyntheticBilingualDataset` in `iqgp/data/datasets.py`.
4. **Augment with code-switch variants** if you plan to exercise the code-switch consistency loss.

## 4. Configuring the Experiment

Key hyperparameters are exposed via CLI arguments in `scripts/train.py`:

- `--steps`: number of training batches to iterate.
- `--seq-len`, `--hidden`, `--vocab`: shape of token embeddings and decoder vocabularies.
- `--max-nodes`, `--codebook`: planner graph budget and VQ codebook size.
- `--entities`, `--units`: number of candidate entity/unit embeddings supplied per example.
- `--lexicon`: projection dimensionality for the bilingual lexicon head.
- `--batch-size`, `--lr`, `--log-every`: optimizer configuration and logging cadence.

Adjust these values to match your dataset. For larger backbones, ensure the hidden size matches the model embeddings you feed into the reasoner.

## 5. Training Workflow

1. **Prepare candidate tensors**: adapt the `build_candidates` helper in `scripts/train.py` to load precomputed entity and unit embeddings instead of the synthetic random tensors.
2. **Instantiate the reasoner**:
   ```python
   model = BilingualReasoner(
       hidden_size=args.hidden,
       vocab_en=args.vocab,
       vocab_zh=args.vocab,
       max_nodes=args.max_nodes,
       codebook_size=args.codebook,
       num_entities=args.entities,
       num_units=args.units,
   )
   ```
3. **Construct objectives**: the training script combines task cross-entropy, EMD alignment, InfoNCE, entity/unit agreement, code-switch consistency, language-erasure, and the VQ commitment loss. Tune the weighting coefficients inside the loop to balance gradients.
4. **Run the training script**:
   ```bash
   python scripts/train.py --steps 1000 --seq-len 128 --vocab 32000 --hidden 1024 \
       --max-nodes 32 --codebook 512 --entities 64 --units 32 --lexicon 256 \
       --batch-size 8 --lr 3e-4 --log-every 10
   ```
   Replace the defaults with values appropriate for your setup. The script detects CUDA automatically.
5. **Monitor logs**: the loop prints total loss every `log_every` steps. For deeper insight, log individual loss terms (task, EMD, InfoNCE, etc.) to TensorBoard or Weights & Biases by instrumenting the script.
6. **Checkpointing**: wrap the optimizer step with `torch.save` calls to persist model, planner codebook, and optimizer state dicts.

## 6. Evaluation and Inference

1. **Plan-once verbalize-twice**: after training, run a forward pass with English and Chinese inputs to obtain a shared `PlannerGraph`. Use the decoder logits to sample or beam-search language-specific answers.
2. **Cross-critique**: implement the critique loop outlined in `docs/plan.md` by having each decoder evaluate the other language’s output conditioned on the same graph. Select the hypothesis with the highest joint likelihood.
3. **Metrics**: compute answer exact match, bilingual agreement (EN vs. translated ZH answers), graph fidelity (if supervision exists), and entity/unit consistency.

## 7. Testing and Validation

1. **Unit tests**: execute `pytest` to run planner, loss, and reasoner tests. They skip automatically if PyTorch is unavailable.
2. **Smoke tests with synthetic data**: run `python scripts/train.py` without modifications to validate that the environment is configured correctly.
3. **Integration tests**: once real datasets are wired, add dataset-specific regression tests to `tests/` to cover data loaders and end-to-end decoding.

## 8. Extending the System

- **Backbone replacement**: swap `SharedBackbone` with a pre-trained transformer (e.g., mBERT, XGLM). Ensure adapter dimensions match the hidden size.
- **Real entity linker**: plug in an external EL system and feed learned embeddings through the planner pointer heads.
- **Advanced decoding**: replace the simple linear generators with language-model heads that condition on the planner graph through cross-attention.
- **Curriculum and ablations**: script experiments that toggle losses (`lambda_i` weights) to reproduce the ablation suite described in `docs/plan.md`.

## 9. Troubleshooting Checklist

- **Exploding losses**: check that entity/unit candidate masks are boolean and correctly limit invalid entries to `-inf` in pointer logits.
- **Degenerate plans**: inspect `PlannerGraph.node_mask` to ensure the length controller is calibrated; adjust the threshold or add a coverage penalty.
- **EMD instability**: reduce the Sinkhorn iterations or add temperature damping in the Sinkhorn solver inside `iqgp/objectives/losses.py`.
- **Language leakage**: verify that the adversary receives only non-entity states and that gradient reversal is active in `LanguageAdversary`.

## 10. Documentation Expectations

- Record new experiments, parameter sweeps, and observations in `docs/progress.md` to maintain a chronological log.
- Update this manual when introducing new scripts, datasets, or evaluation protocols so on-boarding contributors can reproduce results end-to-end.

By following this workflow, you can extend the I-QGP experiment from synthetic smoke tests to full bilingual reasoning studies aligned with the design goals in `docs/plan.md`.
