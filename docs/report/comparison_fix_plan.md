# §4.9 Literature Comparison — Fix Plan (validated)

> Goal: turn the weak, qualitative contextual-comparison subsection into a
> concrete, data-driven one, and close the metric gap (AUC/Kappa). All numbers
> below are VALIDATED against the repo (results/tables/, references/**/paper.md)
> on 2026-06-08. macro-F1 stays the headline; no SOTA; protocol-different
> literature stays contextual.

## 1. Validation status (all confirmed)

**Our model (traceable to results/tables/):**
| Variant | acc / micro-F1 | macro-F1 (CI) | macro-P | macro-R | weighted-F1 | MCC | Kappa | macro AUC (OVR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp 11 base (CE) | 0.8706 | 0.6000 [0.5814, 0.6206] | — | — | 0.8716 | 0.8599 | — | — |
| **exp 11 + TTA (final)** | **0.8765** | **0.6075 [0.5860, 0.6296]** | 0.6162 | 0.6158 | 0.8761 | 0.8662 | **0.8661** | **0.967** |

- Kappa (0.8661) and macro ROC-AUC OVR (0.967) computed from the saved pooled TTA
  softmax (`results/runs/11_*/predictions_tta.npz`, key `probs`); inference-free.
- acc/macro-F1/MCC/CI match `extra_metrics_11{,_tta}.json` and `ci_11{,_tta}.json` exactly.

**Borgli 2020 (paper.md Table 3, SAME official-split family):**
| Method | macro-F1 | micro-F1/acc | MCC |
|---|---:|---:|---:|
| ResNet-50 | 0.530 | 0.839 | 0.826 |
| ResNet-152 | 0.606 | 0.906 | 0.898 |
| DenseNet-161 | 0.619 | 0.907 | 0.899 |
| ResNet-152 + DenseNet-161 + MLP | 0.605 | 0.909 | 0.902 |

Caveat: Borgli reports the **average of 2 official splits**; we use the official
**5-fold**. Both official-split-based → protocol-aligned (minor split-count difference).

**Contextual (different protocol — NOT head-to-head):**
| Source | Method | Protocol | macro-F1 | acc | AUC |
|---|---|---|---:|---:|---:|
| Effimix (Ramamurthy 2022) | EffNetB0+Effimix fusion | augment→23k + equal-class sampling, non-official split | 0.97 | 0.98 | 0.977 |
| GastroViT 2025 | MobileViT ensemble | 64:16:20 single split (23-class) | 0.64 | 0.92 | — |
| He et al. (via Effimix Table 5) | ResNet-152+MobileNetV3 | unspecified | 0.66 | — | — |
| Galdran (CNN–BiT, via Effimix T5) | BiT | unspecified | 0.92 | — | — |
| Galdran (MNV2+ResNeXt, via Effimix T5) | MNV2+ResNeXt | unspecified | 0.64 | 0.65 | — |

(GastroViT's 16-class results are a DIFFERENT task; never mix with our 23-class.)

## 2. Honest narrative (the story to tell — data-driven)

1. **Same-protocol (Borgli) = near-direct, the key comparison.** Our lighter triple
   fusion (ResNet50+MobileNetV2+EfficientNetB0, ≈14M backbone params) **matches
   Borgli's heavier RN152+DN161+MLP fusion on macro-F1** (0.6075 vs 0.605) and
   approaches their best single backbone DenseNet-161 (0.619); but it **trails by
   ~3–4 points on accuracy/micro-F1** (0.877 vs ~0.907) and **MCC** (0.866 vs ~0.90).
   Honest reading: we match on the imbalance-sensitive metric (macro-F1) with far
   lighter backbones, but do not surpass the heavier models on overall accuracy.
   No superiority/SOTA claim.
2. **Contextual high numbers come from non-official protocols.** Effimix's 0.97
   macro-F1 / 0.98 acc rely on augmentation-to-23k + equal-class sampling + a
   non-official split (optimistic). The 23-class macro-F1 landscape under *honest
   official-protocol* evaluation sits ~0.53–0.62 (Borgli) — our 0.61 is in that band.
3. **AUC nuance (illustrative, also cross-protocol):** our threshold-free macro AUC
   (0.967) is close to Effimix's reported AUC (0.977), even though our macro-F1
   (0.61) is far below theirs (0.97). This shows the large F1 gap is driven by the
   operating point / class balance / protocol, not by ranking quality — frame as
   illustrative, not head-to-head.

## 3. Fix tasks

### Task 1 — close the metric gap (script + tables) [ASSISTANT, inference-free]
- Extend `scripts/compute_extra_metrics.py` to also compute **Cohen's Kappa**
  (`cohen_kappa_score`) and **macro one-vs-rest ROC-AUC** (`roc_auc_score(..,
  multi_class='ovr', average='macro')`) from the saved per-fold `probs` (pooled).
  Add `cohen_kappa` and `auc_macro_ovr` to the output JSON.
- Re-run for exp 11 base and +TTA (and optionally 10/13/14/15). No training.
- These match the metric sets Effimix/GastroViT report → fair comparability.

### Task 2 — rewrite `04_results.tex` §4.9 (subsec:contextual) [ASSISTANT]
Replace the qualitative paragraph with:
- **Table A — "Aynı protokol (resmi bölme) karşılaştırması"**: the Borgli rows +
  our exp 11 + TTA row (macro-F1 / micro-F1 / MCC). Caption notes 2-split vs 5-fold.
- **Table B — "Bağlamsal karşılaştırma (farklı protokol)"**: Effimix / GastroViT /
  He / Galdran with a **Protokol** column + macro-F1 (+ acc/AUC where available);
  explicit caveat that these are contextual, not head-to-head.
- Prose: the honest narrative (§2 above) — match macro-F1 vs Borgli, trail on
  accuracy/MCC, contextual numbers non-comparable, AUC nuance. macro-F1 headline.
- Optionally add Kappa + macro-AUC to the supporting-metrics table (subsec:support-metrics).

## 4. Honesty rules (must hold)
- macro-F1 is the headline; accuracy/micro-F1/MCC/Kappa/AUC are supporting.
- No SOTA / no direct-superiority claim over any external work.
- Same-protocol (Borgli) = near-direct, with the 2-split/5-fold caveat stated.
- Different-protocol rows = contextual ONLY; cite source (paper.md or "via Effimix Table 5").
- Do not mix GastroViT 16-class with our 23-class.
- Every number traces to results/tables/ or a references/**/paper.md.

## 5. Acceptance criteria
- `compute_extra_metrics.py` emits `cohen_kappa` + `auc_macro_ovr`; exp 11 (+TTA)
  JSONs regenerated (Kappa 0.8661, AUC 0.967 for +TTA).
- §4.9 has Table A (same-protocol) + Table B (contextual) with the validated numbers
  + honest narrative + caveats.
- `uv run pytest tests/` passes; no training; no PLD change; no SOTA claim.
