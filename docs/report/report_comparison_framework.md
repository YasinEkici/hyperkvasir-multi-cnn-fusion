# Report Comparison Framework — Multi-CNN Feature Fusion (HyperKvasir 23-class)

> Planning document for the project report. Grounded in: assignment spec,
> `project_plan.md` (PLDs + report plan), `docs/decisions.md` (D-06…D-09),
> `docs/results_progress.md` (Week 1–3 results + CI), `references/INDEX.md`,
> and the primary baseline papers (Borgli 2020, Ramamurthy/Effimix 2022,
> GastroViT 2025, Ahmed 2023).

## 1. Assignment Interpretation

**Explicit requirements (assignment §3–§5):**

| Requirement | Scope |
|---|---|
| 3 CNNs | ResNet50, MobileNetV2, EfficientNetB0 (fixed) |
| Feature extraction | Drop classification head, take feature vector |
| Fusion | **Concat + Weighted** (both mandatory) |
| Classifier | **MLP** (single) |
| Ablation (§4.1) | single-CNN / two-CNN / three-CNN comparison |
| Transfer learning (§4.2) | **frozen feature-extraction** vs **fine-tuning (last 2–3 blocks)** |
| Metrics (§4.2) | Accuracy, Precision, Recall, F1-score |
| Discussion (§5.5, most critical, 25%) | Which model/combination is best, why fusion helped/didn't, strengths/weaknesses, which TL approach + how long it took |
| Submission (§6) | Code + PDF report + **YouTube video** (8 June 23:59) |

**Errors / ambiguities (flagged explicitly):**
- ⚠️ **"3 classifier-based comparison" (§4.1, §5.1) is WRONG.** Instructor confirmed; recorded in `project_plan.md` (lines 9–11) and PLD-06. → **No classifier-wise comparison.** Justify in one sentence in the report (assignment §3.4 already states classifier = MLP).
- Ambiguous: evaluation protocol (split/folds) undefined in the assignment → we adopt the **HyperKvasir official 5-fold** protocol (decision 2026-05-23). This is a strength, not a weakness.
- Ambiguous: macro vs micro metric — assignment only says "F1-score". Literature + imbalance resolve this to **macro** (see §3).

**Assumption:** Report language (TR/EN) and video format (screen recording + narration) not yet confirmed. The outline below is language-agnostic.

## 2. Existing Project Decisions (docs summary)

**Locked design (PLD, `project_plan.md` §5):** HyperKvasir 23-class (PLD-01) · 224×224 (PLD-03) · branch projection 512, Linear+LayerNorm+GELU (PLD-04/05) · **MLP-only** (PLD-06) · concat+weighted mandatory, **GMU = advanced** (PLD-07/08) · AdamW+LLRD (PLD-10) · **macro-F1 + per-class headline** (PLD-11) · torchvision-only, no timm (PLD-15) · config-controlled determinism (PLD-18).

**Protocol (decision 2026-05-23):** Official 5-fold; fold *k* = test, *(k+1)%5* = val, rest = train. Manifests saved, leak-checked.

**Week 3.5 additions (D-06…D-09):** Focal loss as **additive ablation** (exp 16, single variable CE→focal, γ=2.0) · leakage-free ensembling policy · A100 batch/LR scaling · Colab runner-only + provenance gate.

**Verifiable results on hand (`results_progress.md`, all traceable to `metrics.json`):**

| Comparison | Best | Evidence |
|---|---|---|
| Frozen, single backbone | EfficientNetB0 (F1 0.5586) | W2 table |
| Frozen, best overall | M+E concat (F1 0.5758) | W2 |
| Fine-tune fold-0 best | triple concat (F1 0.5796) | W2 |
| **5-fold CV best** | **triple weighted (F1 0.5892 ± 0.0102)** | W3 |
| **Bootstrap CI (pooled)** | **triple weighted F1 0.6000, CI [0.5814, 0.6206]** | W3, `ci_11…json` |
| Focal (exp 16) | running — placeholder | — |

## 3. Literature-Based Validation

Each claim is cited; the "comparable" note indicates protocol alignment.

| Claim | Source + evidence location | Implication for us |
|---|---|---|
| On HyperKvasir 23-class, **macro + micro F1 + MCC** are the standard reported metrics | **Borgli 2020**, Table 3 (macro & micro P/R/F1, MCC) | Macro-F1 headline, micro/acc support, MCC optional-recommended |
| **Accuracy alone is misleading** (imbalance) | Borgli Table 3: micro-F1 0.90 **vs** macro-F1 ~0.62 (same model) | We show acc 0.87 vs macro-F1 0.60 → same gap; discuss explicitly |
| Official split / cross-validation **recommended for fair comparison** | Borgli, "Usage Notes": "three official splits … keeping splits consistent … fair comparison" | Our official 5-fold choice is literature-backed |
| MLP-on-fused-CNN-features is a **valid fusion pipeline** | Borgli Table 3: "ResNet-152 + DenseNet-161 + **MLP**" baseline | Direct precedent for our architecture |
| Modern reporting includes **mean±std / uncertainty** | **GastroViT 2025**, abstract: F1 "64±0.38%"; Borgli two-split average | Our 5-fold mean±std + **bootstrap CI** is stronger than all baselines |
| High "accuracy" results (Effimix 97.99%) use **augmentation + balancing + non-official split** | **Ramamurthy 2022**, §3.2 (augment to 23k), §4.1 (equal-class sampling), split unspecified | NOT head-to-head; **contextual** only (PLD-17, honest reporting) |
| GastroViT 23-class: acc 91.98% but macro-F1 ~64% | GastroViT abstract + Table 4 | High acc ≠ high macro-F1; protocol difference, contextual |
| Confusion matrix + per-class analysis **expected** | Borgli Fig 8; Effimix Table 3 (class-wise); GastroViT (Grad-CAM) | Per-class table + confusion matrix mandatory |
| ROC-AUC **used on HyperKvasir but secondary** | Effimix (AUC 0.977), Ahmed (AUC) — single number, multiclass macro-AUC ambiguous | **Optional/skip**; macro-F1 more informative |

**Important protocol caveat (not a conflict, a difference):** Our ~0.60 macro-F1 is **not directly comparable** to Effimix's 0.97 — they augment + class-balance before split and use a non-official split (likely optimistic bias). **The only protocol-aligned anchor is Borgli's official-split macro-F1 ≈ 0.62**, and our 0.60 sits in the same band. State this clearly in the report; do not hide it.

## 4. Recommended Comparison Values

| Metric | Measures | Why needed | When critical | Use? | Literature |
|---|---|---|---|---|---|
| **Accuracy (micro)** | Overall correct rate | Assignment §4.2 mandatory | Balanced data | ✅ support (not headline) | Borgli T3 |
| **Macro Precision** | Class-avg precision | Assignment mandatory | Under imbalance | ✅ | Borgli/Effimix/GastroViT |
| **Macro Recall** | Class-avg recall | Assignment mandatory | Rare-class misses | ✅ | same |
| **Macro F1** ⭐ | Harmonic mean of P/R, class-equal | **PLD-11 headline** | **Imbalanced multiclass (our case)** | ✅ **primary** | Borgli macro-F1 |
| **Per-class P/R/F1 + support** | Per-class performance | §5.5 discussion core | Which class collapses | ✅ (Table 3) | Borgli, Effimix T3 |
| **Confusion matrix** | Error structure | Which class confuses which | Always | ✅ (best model) | Borgli Fig8 |
| **Micro/Weighted F1** | Sample-weighted | Shows macro↔micro gap | Imbalance evidence | ◯ optional | Borgli (both) |
| **MCC** | Balanced correlation | Imbalance-robust single score | Imbalance | ◯ recommended | Borgli, Effimix |
| **Bootstrap 95% CI** | Uncertainty interval | Statistical distinguishability | Close results | ✅ (already have) | our methodological edge |
| **Training time / epochs** | TL cost | §5.5 "how long it took" | frozen vs fine-tune | ✅ (Table 4) | GastroViT (compute) |
| **ROC-AUC** | Threshold-free | — | Binary/few-class | ✗ skip | Effimix/Ahmed (secondary) |

**Computed without retraining (from saved `predictions.npz`, pooled n=10,662, 5-fold):**

| Exp | Method | Acc / micro-F1 | Macro-F1 | Weighted-F1 | MCC |
|---|---|---:|---:|---:|---:|
| **11** | **R+M+E weighted** | **0.8706** | **0.6000** | **0.8716** | **0.8599** |
| 10 | R+M+E concat | 0.8579 | 0.5708 | 0.8613 | 0.8464 |
| 14 | M+E concat | 0.8482 | 0.5759 | 0.8522 | 0.8359 |
| 13 | EfficientNetB0 | 0.8484 | 0.5730 | 0.8536 | 0.8367 |
| 15 | R+M+E GMU | 0.8481 | 0.5690 | 0.8507 | 0.8358 |

Saved to `results/tables/extra_metrics_*.json`. Note: micro-F1 == accuracy for single-label multiclass. exp 11 is champion across every metric; the macro↔weighted gap (~0.27) is the imbalance evidence.

## 5. Recommended Comparison Framework

**Internal comparisons (all same official protocol, leakage-free → head-to-head valid):**
1. **CNN-wise:** R vs M vs E (single backbone) → which backbone learns better features (§5.5 question)
2. **Fusion-wise:** single → pair → triple; **concat vs weighted** (mandatory) + **GMU** (advanced)
3. **Transfer-wise:** frozen vs fine-tune (per config)
4. **Loss-wise (Week 3.5, creative):** CE vs **focal** (exp 11 vs exp 16)
5. **Inference-wise (Week 3.5, if done):** base vs TTA

**External comparisons (contextual only, separate table, with caveats):**
- Borgli official-split baseline (macro-F1 ≈ 0.62) → **only protocol-aligned anchor**
- Effimix / GastroViT → different protocol → **discussion only**, never in the head-to-head table (PLD-17)

**Why this framework is correct:** Our internal comparisons on official 5-fold + bootstrap CI are **methodologically stronger** than the baselines' single-split/augmented results. Keeping external numbers contextual is both honest (assignment §8) and satisfies "no superficial commentary" (§9).

## 6. Proposed Report Outline

Merges assignment §5 + `project_plan.md` §10; slots in our actual tables/figures and Week 3.5 placeholders.

**1. Introduction** — Problem (23-class GI endoscopy classification); approach (3-CNN feature fusion + MLP); **MLP-only statement + note "3-classifier was an error, instructor-confirmed"**; three comparison axes (CNN / fusion / transfer).

**2. Method** — Backbones (R/M/E, feature dims 2048/1280/1280); feature extraction; projection (512, LN+GELU); concat & weighted fusion (equations); GMU (advanced, equation from docstring); MLP; frozen vs fine-tune (last 3 blocks, BN frozen); **Week 3.5: focal loss (Lin 2017 eq. 5)**.

**3. Experiments** — Dataset (HyperKvasir, 10,662 images, 23 classes, imbalanced; cite Borgli); **official 5-fold protocol**; hyperparameters (table from `finetune_wide.yaml`); metrics (§4 table); reproducibility (git SHA, provenance, A100).

**4. Results** — *Table 1*: one-fold ablation (W2). *Table 2*: **5-fold CV mean±std** (W3). *Table 2b*: **bootstrap 95% CI**. *Table 3*: best-model per-class P/R/F1/support. *Table 4*: training time/epochs. *Table 5*: **CE vs focal** [placeholder]. *Figures*: architecture diagram, confusion matrix, training curves, comparison bar chart. *Separate subsection*: cross-protocol contextual (Borgli/Effimix/GastroViT, with caveats).

**5. Discussion (MOST CRITICAL)** — Which backbone (E best frozen, weighted wins CV) and why; whether fusion helped/didn't (concat↔weighted CI separation; GMU did not win); best combination (triple weighted, F1 0.60); strengths/weaknesses (rare classes F1=0, support≤10 → imbalance); **frozen vs fine-tune + timing**; **focal verdict** [placeholder]; honest positioning vs literature.

**6. Conclusion** — Summary, limitations (no direct comparison to non-official baselines, tiny rare classes), future work (Grad-CAM++, seed ensemble).

## 7. Final Recommendation

The most defensible structure for the instructor:

1. **Make macro-F1 the headline**, put accuracy beside it, and **discuss the gap explicitly** — shows you understand imbalance (grounded in Borgli; the heart of the 25% discussion grade).
2. **Present internal comparisons on official 5-fold + bootstrap CI**; this methodology is stronger than most baselines — frame it as a **contribution**.
3. **Don't hide external numbers (Effimix 0.97 vs our 0.60); explain them via protocol difference.** Hiding is an academic-honesty risk (§8); explaining shows maturity.
4. **Close the "3-classifier error" in one sentence with reference to instructor confirmation** — leave no gap.
5. **Highlight focal + TTA + Colab/A100 + provenance gate as "creative/extra" (§9)** — not mandatory, earns bonus.
6. **Add MCC + micro-F1 as optional** (cheap, computed from predictions.npz) — robustness signal.

**Missing info (not fabricated):** (a) exp 16 focal results in progress; (b) TTA (Step 4) not yet done; (c) need to confirm epoch-time data is in `epoch_log.jsonl` for Table 4; (d) report language to confirm.
