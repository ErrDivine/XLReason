# I-QGP Experiment Implementation Manual

This manual explains how to operationalize the Interlingua QID-Graph Planner (I-QGP) experiment that lives in this repository. It consolidates the repository structure, required tooling, and the end-to-end workflow for preparing data, training, and validating the bilingual reasoning system. The instructions assume familiarity with PyTorch and bilingual NLP workflows.

## 1. Quick Implementation Checklist

Follow this high-level sequence when moving from the synthetic demo to a real experiment:

1. Stand up a fresh Python environment that can run PyTorch (CPU or GPU).
2. Integrate bilingual datasets and produce tensors that match the `BilingualReasoner` interface.
3. Replace the synthetic candidate builders in `scripts/train.py` with your entity/unit pipeline.
4. Configure training hyperparameters and enable code-switch dropout if desired.
5. Run `python scripts/train.py …` to exercise the full loss suite, then graduate to larger jobs.
6. Add evaluation scripts and integration tests once end-to-end decoding works on real data.

Keep this checklist handy as you work through the detailed steps below.

## 2. Repository Overview

The repository is organized to mirror the pipeline described in `docs/plan.md`:

- `iqgp/planner/`: graph container and the `PlannerIQGP` module that fuses bilingual encodings and emits discrete plans.
- `iqgp/models/`: shared transformer backbone, language adapters/decoders, and the `BilingualReasoner` wrapper that orchestrates planning and decoding.
- `iqgp/objectives/`: lexicon projector, Sinkhorn-based EMD alignment, InfoNCE, entity/unit agreement, code-switch consistency, and language-erasure utilities.
- `iqgp/data/`: synthetic bilingual dataset helpers for smoke tests; replace with real corpora for full experiments.
- `iqgp/utils/`: vector-quantization helpers used by the planner.
- `scripts/train.py`: illustrative training loop wiring all components with the loss suite.
- `tests/`: unit and integration tests that cover planner outputs, losses, and the full reasoning stack.

Refer to `docs/plan.md` for the conceptual blueprint and `docs/progress.md` for the historical implementation log before extending any module.

## 3. Environment Preparation

1. **Install prerequisites**  
   - Python ≥ 3.10  
   - (Optional) CUDA toolkit that matches the GPU build of PyTorch you intend to use.

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. **Install base dependencies**
   ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
   ```
   The requirements file pins the minimal PyTorch/pytest stack. Adjust the PyTorch wheel index to target CPU or CUDA builds, for example:
   ```bash
   pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
   ```

4. **Install optional tooling as needed**
   - `transformers` and `sentencepiece` for tokenizer/backbone inference.
   - `pandas` or `datasets` for data wrangling.
   - Entity linker libraries (e.g., `spacy`, `wikidata` clients) if you plan to integrate real QID retrieval.

5. **Smoke-test the environment**
   ```bash
   python -c "import torch; print('torch:', torch.__version__)"
   python -m pytest tests/test_losses.py -k sinkhorn --maxfail=1 --disable-warnings || true
   ```
   The pytest command will fail inside write-restricted sandboxes unless `TMPDIR` points to a writable folder; adjust as needed.

## 4. Data Requirements and Preparation

The repository ships with random tensors for smoke testing. To run meaningful experiments you must supply real bilingual inputs, entity candidates, and unit candidates. Use the following workflow:

1. **Collect bilingual problems**
   - Paired English/Chinese prompts with answers (e.g., GSM8K + MGSM, translated logic datasets).
   - Optional chain-of-thought annotations or rationales to supervise decoding.

2. **Encode text with a backbone**
   - Choose a multilingual model (e.g., `Llama-3-8B`, `Qwen2-7B`, or a smaller frozen encoder).
   - Use the Hugging Face `transformers` API to tokenize each language separately, then obtain hidden states:
     ```python
     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
     model = AutoModel.from_pretrained("Qwen/Qwen2-7B-Instruct").eval()
     with torch.no_grad():
         en_hidden = model(**tokenizer(en_text, return_tensors="pt")).last_hidden_state
         zh_hidden = model(**tokenizer(zh_text, return_tensors="pt")).last_hidden_state
     ```
   - Align shapes to `(seq_len, hidden_size)` per example; pad/truncate sequences to the same length within a batch.

3. **Generate entity and unit candidates**
   - Run an entity linker over both language inputs; aggregate the top-k QIDs per example.
   - Produce embeddings (using the same backbone or a separate entity encoder) sized `(num_candidates, hidden_size)`.
   - Create boolean masks indicating which candidate slots are valid.
   - Repeat for unit normalization (e.g., map “公里” and “kilometer” to canonical SI units with embeddings).

4. **Persist processed batches**
   - Store tensors (hidden states, masks, answers, candidate pools) in an efficient format (`.pt`, `.npz`, or a custom dataset).
   - Track metadata such as vocabulary sizes, max nodes, and candidate counts to drive CLI arguments.

5. **Implement a dataset loader**
   - Mirror the interface of `SyntheticBilingualDataset`, yielding tuples
     `(en_tokens, zh_tokens, en_mask, zh_mask, answer_en, answer_zh, entity_candidates, unit_candidates)`.
   - Example skeleton:
     ```python
     class RealBilingualDataset(Dataset):
         def __getitem__(self, idx):
             example = load(idx)
             return (
                 example.en_hidden, example.zh_hidden,
                 example.en_mask, example.zh_mask,
                 example.answer_en, example.answer_zh,
                 example.entity_candidates, example.unit_candidates,
             )
     ```

6. **Code-switch augmentation (optional)**
   - Pre-compute swapped-token variants or rely on the runtime augmentation (`--cs-prob`) to mix language contexts during training.

## 5. Configuring the Experiment

All hyperparameters are exposed via CLI arguments in `scripts/train.py`:

- `--steps`: number of training batches to iterate (use dataset length for full epochs).
- `--seq-len`, `--hidden`, `--vocab`: token sequence length, hidden size, and decoder vocabulary size.
- `--max-nodes`, `--codebook`: planner graph budget and VQ codebook size; match plan.md recommendations.
- `--entities`, `--units`: candidate counts per instance; set to the maximum you feed from your dataset.
- `--lexicon`: bilingual projection dimensionality for the lexicon projector.
- `--batch-size`, `--lr`, `--log-every`: optimizer configuration.
- `--cs-prob`: probability of swapping EN/ZH token states per position (0 disables code-switch dropout).

Adapt defaults via CLI or create a wrapper script that loads YAML/JSON configs and forwards values to the parser.

## 6. Wiring Your Data into `scripts/train.py`

The stock script uses random tensors. Update it as follows:

1. **Replace the dataset import**
   ```python
   from iqgp.data.real_dataset import RealBilingualDataset  # your implementation
   ```

2. **Instantiate your dataset**
   ```python
   dataset = RealBilingualDataset(config_path=args.data_config)
   loader = DataLoader(
       dataset,
       batch_size=args.batch_size,
       shuffle=True,
       collate_fn=collate_real_examples,
   )
   ```
   Create `collate_real_examples` to stack tensors and bundle candidate pools.

3. **Remove `build_candidates`**
   - Use the candidate tensors emitted by the dataset loader directly.
   - Ensure they are moved to the correct device before being passed to the model.

4. **Ensure dimension alignment**
   - Verify that hidden sizes and candidate embedding dimensions match the model configuration.
   - Resolve mismatches by adding projection layers or adjusting backbone hidden size.

5. **Enable logging**
   - Instrument the script to log individual loss components, gradient norms, and effective plan lengths.
   - Hook TensorBoard or W&B if you need richer tracking.

6. **Checkpoint models**
   ```python
   torch.save(
       {
           "model": model.state_dict(),
           "optimizer": optimizer.state_dict(),
           "step": step,
           "args": vars(args),
       },
       f"checkpoints/iqgp_step_{step}.pt",
   )
   ```

## 7. Running Training Jobs

1. **Local smoke test**
   ```bash
   python scripts/train.py --steps 20 --batch-size 1 --cs-prob 0.1 --log-every 1
   ```
   Validate that losses are finite and that the planner produces non-empty node masks.

2. **Full experiment**
   ```bash
   python scripts/train.py \
       --steps 5000 \
       --seq-len 256 \
       --vocab 32000 \
       --hidden 2048 \
       --max-nodes 32 \
       --codebook 512 \
       --entities 64 \
       --units 32 \
       --lexicon 256 \
       --batch-size 8 \
       --lr 3e-4 \
       --cs-prob 0.2 \
       --log-every 20
   ```
   Adjust `--steps` to span multiple epochs over your dataset (e.g., `steps = ceil(num_examples / batch_size) * num_epochs`).

3. **Distributed or large-scale runs**
   - Wrap the training loop with PyTorch Distributed Data Parallel (DDP).
   - Shard datasets and synchronize planner codebooks across workers.

## 8. Evaluation and Inference

1. **Forward pass with trained weights**
   ```python
   checkpoint = torch.load("checkpoints/iqgp_final.pt", map_location="cpu")
   model.load_state_dict(checkpoint["model"])
   model.eval()
   output = model(en_tokens, zh_tokens, entity_candidates, unit_candidates, en_mask=en_mask, zh_mask=zh_mask)
   ```

2. **Decode answers**
   - Apply greedy or beam decoding over `output.en_logits` and `output.zh_logits`.
   - Align decoded tokens back to natural language using the same tokenizer as the backbone.

3. **Inspect planner graphs**
   - Examine `output.graph.codes`, `qid_logits`, `unit_logits`, and `edge_logits` to validate plan structure.
   - Use thresholds/argmax to materialize a symbolic representation of the reasoning steps.

4. **Metric computation**
   - **Accuracy**: compare decoded answers to ground truth.
   - **Bilingual agreement**: translate ZH answers (via MT or dictionary) and compare to EN outputs.
   - **Graph metrics**: if plan supervision exists, compute node/edge F1 and QID/unit agreement rates.

5. **Cross-critique loop**
   - Feed the English decoder’s output back into the Chinese decoder (and vice versa) conditioned on the same graph, as outlined in `docs/plan.md`.

## 9. Testing and Validation

1. **Unit tests**
   - Run `python -m pytest -q` after pointing `TMPDIR` (or the platform equivalent) to a writable directory.
   - Add new tests to cover your dataset loader and any modifications to the training loop.

2. **Regression suite**
   - Create deterministic fixtures (small batches with known outputs) to ensure planner code, entity selection, and decoding remain stable across refactors.

3. **Sanity dashboards**
   - Track average plan length, code usage distribution, and entity agreement metrics per epoch to catch collapse early.

## 10. Extending the System

- **Backbone replacement**: integrate a pre-trained multilingual transformer via adapters or fine-tuning.
- **Entity linker integration**: connect to a Wikidata lookup service or a local knowledge base and feed the top-k candidate embeddings into the planner pointers.
- **Unit normalization**: map numeric expressions to canonical units and embed them consistently across languages.
- **Decoder upgrades**: replace linear decoders with auto-regressive language models conditioned on planner node states.
- **Curriculum and ablations**: script experiments toggling loss weights (`lambda_i`) to reproduce the ablations in `docs/plan.md`.

## 11. Troubleshooting Checklist

- **Exploding losses**: ensure candidate masks are boolean and mask invalid logits with `-inf`; normalize learning rate if mixed-precision is enabled.
- **Degenerate plans**: inspect `PlannerGraph.node_mask`; adjust the length controller threshold or regularize via coverage penalties.
- **EMD instability**: reduce Sinkhorn iterations/effects in `SinkhornDistance` or add temperature scaling.
- **Language leakage**: verify that the adversary receives only non-entity planner states (`PlannerGraph.split_states`) and that gradient reversal is active.
- **GPU OOM**: lower `--seq-len`, `--max-nodes`, or batch size; accumulate gradients over multiple steps.

## 12. Documentation Expectations

- Record new experiments, parameter sweeps, and observations in `docs/progress.md` to maintain a chronological log.
- Update this manual when introducing new scripts, datasets, or evaluation protocols so onboarding contributors can reproduce results end-to-end.
- Keep `README.md` and `AGENTS.md` in sync with new CLI flags or setup steps.

By following this workflow, you can extend the I-QGP experiment from synthetic smoke tests to full bilingual reasoning studies aligned with the design goals in `docs/plan.md`.
