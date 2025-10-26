# Project Plan: Bilingual Reasoning via a QID-Graph Interlingua (I-QGP)

## 0) Goal
Build a bilingual reasoning system where Chinese and English **mutually amplify** each other by sharing a **language-agnostic plan**. The plan is a **discrete, entity-anchored graph**; both languages decode from the same graph. Novel mechanisms: QID-anchored interlingua, **Wasserstein (EMD) bilingual alignment** in a lexicon-projected space, **step-wise code-switch dropout**, **entity-aware language erasure**, and **entity/unit agreement** losses.

---

## 1) System Overview

### 1.1 Components
- **Backbone LLM** (4–8B): shared transformer.
- **Planner (I-QGP)**: small module that emits a **step graph** (discrete codes + arguments) from cross-attended bilingual features.
- **EN/ZH Adapters**: LoRA/MoE blocks gated per language (encoder & decoder sides).
- **Lexicon Projector**: tiny bilingual mapping layer for token logits → shared lexical space.
- **Entity Linker & Unit Normalizer (EL/UN)**: retrieves **Wikidata QIDs** (or task-local IDs) and canonical units.
- **Language Adversary**: predicts language from **non-entity** planner states (used with gradient reversal).
- **Answer Heads**: language-specific decoders that verbalize from the shared plan.

### 1.2 Data Flow (training)
1. Input in EN and ZH for the **same problem** (paired), or single-language (unpaired).
2. Encode with backbone + language adapters.
3. Cross-attend EN↔ZH to form **planner inputs**.
4. **Planner** emits a QID-Graph $G$ (nodes = steps; edges = dependencies).
5. **Decoders** generate CoT + answers in EN/ZH from $G$.
6. Losses: task CE, **EMD bilingual alignment**, **plan contrast**, **entity agreement**, **code-switch dropout**, **language eraser**.

---

## 2) Interlingua QID-Graph (I-QGP)

### 2.1 Graph Schema
- **Node**: $\langle \text{op\_code}, \text{args}, \text{QIDs}, \text{units}, \text{evidence\_ptrs}\rangle$  
  - `op_code`: discrete from a **VQ-codebook** (size 256–1024).
  - `args`: indices into previous nodes/constants.
  - `QIDs`: entity IDs (or “NONE”).
  - `units`: normalized (SI or task-specific).
  - `evidence_ptrs`: optional spans to input text.
- **Edge**: dependency (DAG) with typed labels (use, compare, add, explain).

### 2.2 Planner Parameterization
- Input: fused EN/ZH hidden states $H_{en}, H_{zh}$.
- Steps:
  1. **Fusion**: cross-attention + gated residual → $H_f$.
  2. **Node proposal**: small transformer predicts $K$ nodes.
  3. **VQ**: vector-quantize node prototypes → code indices.
  4. **Arguments/QIDs/units**: pointer nets over $H_f$ + EL/UN suggestions.
  5. **Edges**: biaffine scorer over node states.

### 2.3 Discreteness & Length Control
- Fix $K_{max}$ (e.g., 16–64); allow `PAD` nodes.
- Auxiliary penalty for short, sufficient plans (length prior + coverage of evidence).

---

## 3) Training Objectives

### 3.1 Task (Answers & CoT)
$\mathcal{L}_{task} = \mathrm{CE}(y^{en}\!\mid\!x^{en},G) + \mathrm{CE}(y^{zh}\!\mid\!x^{zh},G)$

### 3.2 **Bilingual EMD Alignment (Lexicon-Projected)**
- Map language logits $L_{en}, L_{zh}\in\mathbb{R}^{B\times L \times V}$ via **lexicon projector** $\Pi$ into a shared space $\tilde{L}$.
- Compute token-wise **earth mover distance** between $\tilde{p}_{en}$ and translated $\tilde{p}_{zh}$ with a transport cost that handles many-to-many mappings (idioms/MWEs).
$\mathcal{L}_{\text{EMD-biling}} = \mathrm{EMD}\big(\tilde{p}_{en}, \mathcal{T}(\tilde{p}_{zh})\big) + \mathrm{EMD}\big(\tilde{p}_{zh}, \mathcal{T}^{-1}(\tilde{p}_{en})\big)$

### 3.3 Plan Contrast (EN↔ZH Views)
- Contrastive InfoNCE between **planner states** $Z_{en}$ and $Z_{zh}$ (pooled from node embeddings) for the same problem; negatives are other problems.
$\mathcal{L}_{plan}=\mathrm{InfoNCE}\big(Z_{en}, Z_{zh}\big)$

### 3.4 **Entity/Unit Agreement**
- Encourage identical **QID sets**, **alignment** of node→QID assignments, and **unit normalization** across languages:
$\mathcal{L}_{entity} = \mathrm{CE}(\text{QID}_{en}\!\Rightarrow\!\text{QID}_{zh}) + \mathrm{CE}(\text{unit}_{en}\!\Rightarrow\!\text{unit}_{zh})$

### 3.5 **Step-wise Code-Switch Dropout (CSD)**
- Randomly render $p\%$ of nodes/args in the **other language** during training; require **same graph** and **same final answer**:
$\mathcal{L}_{csd} = \mathrm{CE}(y^{ans}\!\mid\!\text{codeswitched}(x), G) + \lambda_{topo}\,\mathrm{BCE}(\hat{E}, E)$

### 3.6 **Language-Eraser (Entity-Aware)**
- Language classifier on **non-entity** planner states; gradient reversal renders procedures language-agnostic while leaving entity embeddings untouched:
$\mathcal{L}_{erase} = -\mathrm{CE}(d(Z_{\text{non-entity}}), \text{lang})$

### 3.7 Total Loss
$\mathcal{L}=\mathcal{L}_{task} + \lambda_1\mathcal{L}_{\text{EMD-biling}} + \lambda_2\mathcal{L}_{plan} + \lambda_3\mathcal{L}_{entity} + \lambda_4\mathcal{L}_{csd} + \lambda_5\mathcal{L}_{erase}$

---

## 4) Data & Preprocessing

### 4.1 Sources
- Paired EN↔ZH reasoning tasks (math/logic/QA); augment via **high-quality MT** + human spot checks.
- Unpaired corpora for single-language steps (use $\mathcal{L}_{task}+\mathcal{L}_{erase}$ only).

### 4.2 Entity & Unit Grounding
- Run EL to propose candidate QIDs for spans; keep top-k with confidence.
- Normalize numeric expressions/units (e.g., 日期/度量衡 conversions).
- Attach $\{\text{QID}, \text{unit}\}$ to steps when applicable.

### 4.3 CoT & Graph Supervision
- If CoT exists: parse into step sequences; align to graph nodes (weak supervision with editable templates).
- If no CoT: train planner with latent supervision (REINFORCE-style reward on correct answer + minimal graph length).

### 4.4 Code-Switch Construction
- Token-, phrase-, or step-level replacements using bilingual dictionaries & MT; keep **graph constant**.

---

## 5) Implementation Details

### 5.1 Shapes
- Hidden size $d=2048$ (example).
- Planner nodes $K_{max}\in\{16,32,64\}$.
- VQ codebook size $C\in[256,1024]$; commitment loss $\beta_{vq}$.

### 5.2 Key Modules
- **Lexicon Projector** $\Pi$: linear + optional nonlinearity to $d_\ell$ (e.g., 256). Trained jointly.
- **EMD Solver**: Sinkhorn iterations (stabilized) for fast approximate transport.
- **Entity-Aware Masking**: mask entity dimensions from adversary; only structure gets erased.
- **Chinese Microchannels**: glyph/pinyin embeddings injected **only** in ZH decoder (residual add).

### 5.3 Training Loop (high-level)
1. Sample batch of paired/unpaired items.
2. Encode EN & ZH; fuse for planner; sample/argmax VQ codes; build $G$.
3. Decode EN & ZH CoTs/answers from $G$.
4. Compute all losses; backprop (mixed precision).
5. Periodically **EL/UN refresh** with updated retriever.

### 5.4 Optimization
- AdamW; cosine schedule; warmup 2–5%.
- Loss weights init: $\lambda_1=0.5,\lambda_2=0.5,\lambda_3=0.5,\lambda_4=0.2,\lambda_5=0.2$; tune by grid.

---

## 6) Inference

### 6.1 Plan-Once, Verbalize-Twice
- Build $G$ once (optionally ensemble by sampling top-k VQ sequences).
- Decode EN and ZH independently from $G$; prefer shortest valid CoT.

### 6.2 Graph-Level Cross-Critique
- EN decoder critiques nodes/edges (not text); ZH revises; select graph with highest joint likelihood.
- Finalize answers in both languages; report agreement.

---

## 7) Evaluation

### 7.1 Main Metrics
- **Answer EM** (EN, ZH).
- **Bilingual Agreement**: EN answer ↔ translated ZH answer.
- **Graph Fidelity**: node/edge F1 vs. supervised/parsed CoT.
- **Entity Consistency**: QID Jaccard, unit-match rate.
- **Plan Stability**: variance of $G$ across code-switch/noise.

### 7.2 Stress Tests
- **Code-Switch Robustness**: 10–30% step switches.
- **Cross-Lingual Transfer**: train mostly in EN, test ZH (and reverse).
- **Culture-Specific**: idioms/成语, date/number formats, unit conversions.

---

## 8) Ablations (must-run)
1. −EMD → +KL only (check idiom/phrase transfer).
2. −VQ (continuous planner) vs. VQ (discrete).
3. −Entity-aware eraser (eraser on all states).
4. −CSD (no code-switch dropout).
5. −QID anchoring (planner without entities/units).
6. Move Chinese microchannels from decoder → planner (should hurt).


---

## 9) Risks & Mitigations
- **Entity linker noise** → top-k with confidence, late fusion, agreement loss is soft.
- **EMD instability** → Sinkhorn with temperature & entropic regularization; clip costs.
- **Over-discretization** (VQ collapse) → EMA codebook, code usage penalty, Gumbel-softmax warmup.
- **Language leakage** in planner → strengthen eraser; exclude entity dims; increase CSD rate.

---

## 10) Deliverables
- **Code**: modular PyTorch repo (planner, projector, losses, adapters).
- **Data tools**: EL/UN pipeline, code-switch generator, CoT→graph parser.
- **Checkpoints**: base + adapters + planner.
- **Eval suite**: answer/graph/entity metrics, stress tests, ablation scripts.
- **Report**: methodology, experiments, and failure analyses.

---

## 11) Minimal Interfaces (pseudo)

```python
class PlannerIQGP(nn.Module):
    def forward(self, H_en, H_zh, ents, units):  # -> Graph G
        Hf = fuse(H_en, H_zh)                # cross-attn + gates
        nodes = node_proposer(Hf)            # K x d
        z, codes = vector_quantize(nodes)    # VQ indices
        qid_logits = qid_pointer(Hf, ents)   # node->QIDs
        unit_logits = unit_norm(Hf, units)   # node->units
        edges = edge_scorer(nodes)           # KxK
        return Graph(codes, edges, qid_logits, unit_logits)

class LexiconProjector(nn.Module):
    def forward(self, logits):               # [B,L,V] -> [B,L,Dl]
        return proj_layer(logits @ E_vocab)  # E_vocab frozen or trainable

def loss_total(batch):
    G = planner(H_en, H_zh, ents, units)
    logit_en = dec_en(G, H_en); logit_zh = dec_zh(G, H_zh)
    L_task = CE_en(logit_en, y_en) + CE_zh(logit_zh, y_zh)
    L_emd  = emd_bilingual(project(logit_en), project(translate(logit_zh)))
    L_plan = info_nce(pool(G.en_states), pool(G.zh_states))
    L_ent  = entity_unit_agree(G.en_qids, G.zh_qids, G.en_units, G.zh_units)
    L_csd  = csd_consistency(x_codeswitched, G)
    L_erase= lang_eraser(G.non_entity_states)
    return L_task + l1*L_emd + l2*L_plan + l3*L_ent + l4*L_csd + l5*L_erase
```

---

## 12) Acceptance Criteria
- +X% accuracy over monolingual fine-tune on both EN & ZH.
- ≥Y% bilingual agreement without sacrificing accuracy.
- Stat-sig gains on **transfer** (EN→ZH and ZH→EN) and **code-switch** stress.
- Evidence that planner states are language-invariant (adversary AUC ≈ 50%) while **entity/unit** alignment improves.


## 13) Resources scheduled to use
- Basebone model: Qwen2.5-Math-7B
- dataset: juletxara/mgsm (from transformer)
```python
# Exmaple usage. We aim on Chinese and English.
from datasets import load_dataset
ds = load_dataset("juletxara/mgsm", "en")
```