# Project Plan — Multi-CNN Feature Fusion for HyperKvasir

> **HOW TO USE THIS DOCUMENT**
>
> This file pairs with `project_structure.md`. `project_structure.md` defines
> what to build; this file defines what to do, when to do it, and how to verify
> that coding agents did not hallucinate.
>
> **Scope update:** the classifier is **MLP only**. The instructor confirmed
> that the earlier "3 classifier" phrase was accidental. All comparisons are
> CNN-wise, fusion-wise, and transfer-learning-wise; not classifier-wise.

---

## 1. Project Context

**Course:** Deep Learning, Spring 2026  
**Deadline:** 8 June 2026, 23:59 Turkey time  
**Today:** 8 May 2026  
**Available time:** about 31 days  
**Primary hardware:** RTX 5080 laptop GPU, 16 GB VRAM, 64 GB RAM  
**Secondary hardware:** Google Colab Pro+ as fallback / parallel compute

**Reference paper supplied by instructor:**
Bilal, H., Tian, Y., Ullah, I., Garg, S., Choi, B. J., & Hassan, M. M. (2025).
*M-CNN-RF: A hybrid deep learning model for accurate pediatric skeletal age
estimation using hand bone radiographs.* Alexandria Engineering Journal, 129,
289-301.

**Primary dataset:** HyperKvasir 23-class labeled subset.  
**Secondary dataset:** DeepWeeds, stretch only.

### Why HyperKvasir 23-class

- It is public, multi-class, medical-imaging related, and suitable for image
  classification.
- It has recent comparison papers from 2022-2025.
- It is large enough for CNN fine-tuning but small enough to run locally.
- Its class imbalance makes macro-F1, weighted sampling, focal loss, and
  per-class analysis meaningful.
- It is less saturated than many generic natural-image benchmarks.

### Differentiation strategy, revised by priority

**MVP / must finish:**
1. Correct data pipeline and split manifests.
2. Three required backbones: ResNet50, MobileNetV2, EfficientNetB0.
3. Mandatory fusion: concatenation and weighted fusion.
4. MLP classifier only.
5. Single-CNN, two-CNN, and three-CNN ablations.
6. Frozen feature extraction and fine-tuning variants.
7. Accuracy, macro precision, macro recall, macro-F1, per-class metrics,
   confusion matrix.

**Strong version / target if MVP is stable:**
1. GMU as the single primary advanced fusion method.
2. 5-fold stratified CV on selected important configurations.
3. Bootstrap 95% CI for headline metrics.
4. Grad-CAM++ visualizations for the best model.

**Stretch / only if time remains:**
1. AFF and LMF advanced fusion variants.
2. Effimix and GastroViT approximate cross-protocol comparison.
3. DeepWeeds domain-generality check.
4. UMAP + silhouette score.
5. Full McNemar matrix.

---

## 2. Timeline

| Week | Date range | Phase | Key deliverables |
|---:|---|---|---|
| 1 | May 8 - May 14 | Foundation | uv environment; data pipeline; split manifests; backbone wrappers; feature-cache smoke test; one end-to-end single-CNN run. |
| 2 | May 15 - May 21 | Mandatory methods | Single/pair/triple configs for concat and weighted fusion; frozen and fine-tune runs on at least one fold; first ablation table. |
| 3 | May 22 - May 28 | Focused strengthening | Select best mandatory configs; run 5-fold CV on selected set; implement and evaluate GMU; bootstrap CI. |
| 4 | May 29 - June 4 | Analysis and report | Confusion matrix; per-class table; Grad-CAM++; report draft; results discussion. |
| 4.5 | June 5 - June 8 | Buffer and submission | Fix failed runs; record YouTube video; package `.rar`; upload. |

### Hard checkpoints

- **End of Week 1:** `uv run pytest tests/` passes and one single-CNN run produces non-NaN metrics.
- **End of Week 2:** mandatory concat/weighted experiments work on one fold.
- **End of Week 3:** selected 5-fold results are ready; GMU is either complete or dropped.
- **End of Week 4:** report draft v1 is complete with real tables and figures.

---

## 3. Experiment Matrix

The full matrix lives in `configs/experiment_matrix.yaml`, but experiments are
run in stages to avoid over-scoping.

### 3.1 Stage 0 — smoke tests

Run on a tiny subset or one fold only.

| Experiment | Purpose | Required? |
|---|---|---|
| Single ResNet50 frozen | pipeline sanity | yes |
| Single MobileNetV2 frozen | feature dimension sanity | yes |
| Single EfficientNetB0 frozen | feature dimension sanity | yes |
| Triple concat frozen | fusion shape and MLP sanity | yes |

### 3.2 Stage 1 — mandatory one-fold ablation

Run all mandatory configurations on fold 0 first.

| Method | Frozen | Fine-tune narrow | Fine-tune wide |
|---|:---:|:---:|:---:|
| Single ResNet50 | yes | optional | yes |
| Single MobileNetV2 | yes | optional | yes |
| Single EfficientNetB0 | yes | optional | yes |
| Pair R+M concat | yes | optional | yes |
| Pair R+E concat | yes | optional | yes |
| Pair M+E concat | yes | optional | yes |
| Triple concat | yes | optional | yes |
| Triple weighted | yes | optional | yes |

Output: a one-fold ablation table used to select promising configs.

### 3.3 Stage 2 — selected 5-fold CV

Run 5-fold CV only for selected configurations.

Minimum selected set:

| Method | Mode | Runs |
|---|---|---:|
| Best single CNN | best mode from Stage 1 | 5 |
| Best pair concat | best mode from Stage 1 | 5 |
| Triple concat | best mode from Stage 1 | 5 |
| Triple weighted | best mode from Stage 1 | 5 |
| Triple GMU | best fine-tune mode | 5 |

This reduces scope while still satisfying the assignment and producing strong
comparative evidence.

### 3.4 Stage 3 — stretch experiments

| Experiment | Runs | Drop condition |
|---|---:|---|
| AFF | 1-5 | drop if GMU is unstable or report is unfinished |
| LMF | 1-5 | drop if tensor implementation takes too long |
| Effimix protocol | 1-2 | drop if exact split cannot be reconstructed quickly |
| GastroViT protocol | 1-2 | drop if exact protocol is unclear |
| DeepWeeds | 3 | drop if HyperKvasir analysis is incomplete |

### 3.5 Compute budget, revised

Expected core compute:
- Frozen feature-cache runs: about 1-2 hours total after cache.
- Stage 1 one-fold mandatory runs: about 8-15 hours depending on fine-tuning.
- Stage 2 selected 5-fold CV: about 20-35 hours.
- Analysis and reruns: add 30-50% buffer.

Target: **finish a defensible project before adding stretch experiments**.

---

## 4. Anti-Hallucination Protocol

Coding agents hallucinate confidently. This section catches mistakes before
they become broken experiments.

### 4.1 Common hallucinations in this project

1. Wrong torchvision layer indexing.
2. Wrong feature dimensions.
3. Invented PyTorch/torchvision APIs.
4. Paraphrased fusion equations instead of exact implementation.
5. Hidden non-MLP classifiers added because a paper used them.
6. Exact competitor splits invented from papers that do not specify them.
7. Missing data leakage checks.

### 4.2 Required verification before declaring a module done

1. Match the function/class signature in `project_structure.md`.
2. Add or update the matching test.
3. Run the relevant tests with `uv run pytest`.
4. For each backbone, verify feature shape:

```python
x = torch.randn(2, 3, 224, 224)
assert build_backbone("resnet50")(x).shape == (2, 2048)
assert build_backbone("mobilenetv2")(x).shape == (2, 1280)
assert build_backbone("efficientnetb0")(x).shape == (2, 1280)
```

5. For each fusion module, verify output shape.
6. For GMU/AFF/LMF, put the paper equation in the docstring first.
7. Save a short note in `docs/experiment_log.md` after each successful run.

### 4.3 Stop-and-ask triggers

The agent must stop and ask if:
- a paper equation is ambiguous,
- installed library behavior differs from verified facts,
- a test fails after an attempted fix,
- a new dependency is needed,
- it wants to simplify the plan,
- it wants to add a non-MLP classifier,
- it wants to change dataset class count or merge rare classes,
- a competitor split cannot be reconstructed.

---

## 5. Locked Design Decisions

| ID | Decision | Reason |
|---|---|---|
| PLD-01 | Primary dataset: HyperKvasir 23-class labeled subset | Public, multi-class, recent literature, medical-imaging fit |
| PLD-02 | Secondary dataset: DeepWeeds is stretch | Good domain-generality test but not needed for MVP |
| PLD-03 | Image size: 224×224 | Controlled comparison across all backbones |
| PLD-04 | Per-branch projection dim: 512 | Handles 2048 vs 1280 feature mismatch before fusion |
| PLD-05 | Projection uses Linear + LayerNorm + GELU | Stable with small/medium batch sizes |
| PLD-06 | Classifier: MLP only | Assignment requirement; instructor confirmed no 3-classifier requirement |
| PLD-07 | Mandatory fusion: concat and weighted | Assignment requirement |
| PLD-08 | Primary advanced fusion: GMU | Creative but manageable scope |
| PLD-09 | AFF and LMF are stretch | Avoid over-scoping |
| PLD-10 | Optimizer: AdamW + optional LLRD | Transfer-learning best practice |
| PLD-11 | Report macro-F1 and per-class metrics | HyperKvasir is imbalanced |
| PLD-12 | Split manifests are saved | Prevent leakage and enable reproducibility |
| PLD-13 | Use uv for local environment management | Faster setup and lockfile-based reproducibility |
| PLD-14 | Keep `requirements-colab.txt` | Practical Colab fallback |
| PLD-15 | Do not use timm for required backbones | Avoid layer-index and feature-dim mismatch |
| PLD-16 | Feature cache for frozen runs | Reduces compute time |
| PLD-17 | Exact competitor split not found → approximate reproduction | Honest reporting |
| PLD-18 | Determinism is config-controlled | Allows reproducibility mode and faster exploration mode |

---

## 6. Methods to Implement

### 6.1 Mandatory

| Method | File to read first | Difficulty |
|---|---|---:|
| Concatenation fusion | none | low |
| Weighted fusion | none | low |
| MLP classifier | assignment spec | low |
| Frozen feature extraction | project_structure.md | low |
| Fine-tuning variants | project_structure.md | medium |
| Single/pair/triple ablations | project_plan.md | medium |

### 6.2 Advanced / strong version

| Method | File to read first | Difficulty |
|---|---|---:|
| GMU | `references/methodology_fusion/gmu_arevalo_2020_ncaa.md` | low-medium |
| Grad-CAM++ | `references/methodology_evaluation/gradcam_pp_chattopadhyay_2018_wacv.md` | medium |
| Bootstrap CI | `references/methodology_evaluation/mcnemar_dietterich_1998.md` for evaluation context | low |

### 6.3 Stretch

| Method | File to read first | Difficulty |
|---|---|---:|
| AFF | `references/methodology_fusion/aff_dai_2021_wacv.md` | medium |
| LMF | `references/methodology_fusion/lmf_liu_2018_acl.md` | medium |
| UMAP | `references/methodology_evaluation/umap_mcinnes_2018_joss.md` | low-medium |
| DeepWeeds | `references/secondary_dataset/deepweeds_olsen_2019_scirep.md` | medium |

---

## 7. uv Setup Checklist

Local setup:

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest tests/
```

Add to README:

```bash
# Prepare data
uv run python scripts/prepare_data.py --dataset hyperkvasir

# Create split manifests
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_own.yaml

# Extract frozen features
uv run python scripts/extract_features.py --config configs/dataset/hyperkvasir_23class_own.yaml

# Train one experiment
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 01_single_resnet50_frozen_own
```

Colab fallback:

```bash
pip install -r env/requirements-colab.txt
```

Final environment archive:

```bash
uv pip freeze > env/pip-freeze.txt
```

---

## 8. Validation Checkpoints

### End of Week 1

- [ ] `uv sync` works locally.
- [ ] CUDA availability is tested and logged.
- [ ] `pytest tests/` passes.
- [ ] HyperKvasir sample count and class count are logged.
- [ ] Split manifests are generated and checked for leakage.
- [ ] One single-CNN run completes and saves metrics.

### End of Week 2

- [ ] Mandatory fusion modules work.
- [ ] One-fold ablation table is generated.
- [ ] No NaN losses.
- [ ] No all-one-class predictions.
- [ ] Feature cache files match image counts and labels.

### End of Week 3

- [ ] Selected 5-fold experiments are complete.
- [ ] GMU result is complete or explicitly dropped with reason.
- [ ] Bootstrap CI is computed for headline results.
- [ ] Report result tables are draftable.

### End of Week 4

- [ ] Per-class results table is ready.
- [ ] Confusion matrix is ready.
- [ ] Grad-CAM++ figure is ready or dropped with reason.
- [ ] Report draft v1 is complete.
- [ ] YouTube video outline is ready.

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| RTX 5080 / CUDA / PyTorch mismatch | Medium | High | Test on day 1; log exact versions; fallback to Colab. |
| HyperKvasir download unavailable | Low | High | Mirror dataset to Drive after first download. |
| Class imbalance destabilizes training | Medium | Medium | Macro-F1, weighted sampler, focal loss ablation. |
| Ultra-rare classes break stratified folds | Medium | Medium | Check fold class counts; document any class-handling decision. |
| Agent adds non-MLP classifier | Medium | Medium | PLD-06 and forbidden pattern. |
| Advanced methods consume too much time | High | Medium | GMU core only; AFF/LMF stretch. |
| Report/video deadline missed | Medium | Total | Freeze core experiments by end of Week 3. |
| Competitor protocols cannot be exactly reconstructed | Medium | Low | Mark as approximate reproduction or move to discussion only. |

---

## 10. Report Deliverables

Report sections:

1. **Introduction**
   - Problem definition
   - Dataset motivation
   - CNN feature extraction and fusion overview
   - MLP classifier statement
2. **Method**
   - ResNet50, MobileNetV2, EfficientNetB0
   - Feature extraction
   - Projection layer
   - Concatenation fusion
   - Weighted fusion
   - GMU if completed
   - MLP classifier
   - Frozen vs. fine-tuning setup
3. **Experiments**
   - Dataset description
   - Split protocol
   - Hyperparameters
   - Metrics
4. **Results**
   - Single CNN results
   - Pair fusion results
   - Triple fusion results
   - Frozen vs. fine-tune comparison
   - Best model confusion matrix
5. **Discussion**
   - Which CNN learned better features and why
   - Whether fusion helped and why
   - Best combination
   - Strengths/weaknesses
   - Class imbalance effects
   - Training time and transfer-learning comparison
6. **Conclusion**
   - Summary of findings
   - Limitations
   - Future work

### Tables

- Table 1: Mandatory one-fold ablation.
- Table 2: Selected 5-fold results, mean ± std.
- Table 3: Per-class precision/recall/F1/support for the best model.
- Table 4: Training time comparison.
- Optional Table 5: Cross-protocol or DeepWeeds stretch results.

### Figures

- Full architecture diagram.
- Confusion matrix.
- Training curves.
- Grad-CAM++ examples if completed.
- UMAP if completed.

No classifier-wise comparison table is required because the classifier is fixed
as MLP.

---

## 11. Submission Checklist

- [ ] Code committed to clean repository.
- [ ] `pyproject.toml`, `uv.lock`, `.python-version` present.
- [ ] `env/requirements-colab.txt` and `env/pip-freeze.txt` present.
- [ ] `README.md` includes uv and Colab setup.
- [ ] Report PDF complete.
- [ ] All cited papers have reference markdown files.
- [ ] All headline numbers trace back to `metrics.json` or paper evidence.
- [ ] YouTube video uploaded.
- [ ] `.rar` includes report, code, README, and YouTube link.
- [ ] Uploaded to ekampüs before deadline.

