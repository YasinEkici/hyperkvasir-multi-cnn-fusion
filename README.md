# Multi-CNN Feature Fusion for HyperKvasir 23-Class Classification

This repository contains a PyTorch implementation of a multi-CNN feature-fusion
classifier for the HyperKvasir 23-class gastrointestinal endoscopy benchmark.
It was built as a Deep Learning course term project with a focus on controlled
ablation, reproducibility, and honest reporting under class imbalance.

The core idea is simple: three pretrained CNN backbones learn complementary
visual representations, each branch is projected into a shared feature space,
the projected vectors are fused, and a fixed MLP classifier predicts one of the
23 endoscopy classes.

## What This Project Adds

The assignment required three CNN models, feature extraction, feature fusion,
and an MLP classifier. This project implements that baseline, then extends it
with a stronger experimental protocol:

- Three controlled torchvision backbones: ResNet50, MobileNetV2, EfficientNetB0.
- Branch-wise projection to a common 512-dimensional feature space before
  fusion.
- Mandatory fusion methods: concatenation and learnable weighted fusion.
- Advanced fusion ablation: GMU, with AFF and LMF implemented as stretch modules.
- Frozen feature extraction and end-to-end fine-tuning variants.
- Official HyperKvasir 5-fold protocol with leakage-checked fold manifests.
- Macro-F1 as the headline metric because the dataset is highly imbalanced.
- Bootstrap 95% confidence intervals on pooled 5-fold predictions.
- Inference-only Test-Time Augmentation (TTA) on the final frozen model.
- Leakage-free seed ensembling policy documented in `docs/decisions.md`.
- Report-ready figures and tables generated from saved predictions/logs.

The project does not claim state of the art. External papers that use different
splits, class counts, or augmentation protocols are treated as contextual
comparisons, not direct head-to-head baselines.

## Final Result Snapshot

Final frozen model: `11_triple_weighted_finetune_wide_official` + deterministic
4-view TTA.

| Metric | Value | Source |
|---|---:|---|
| Macro-F1 | 0.6075 | `results/tables/ci_11_triple_weighted_finetune_wide_official_tta.json` |
| 95% CI for macro-F1 | [0.5860, 0.6296] | same |
| Accuracy / micro-F1 | 0.8765 | `results/tables/extra_metrics_11_triple_weighted_finetune_wide_official_tta.json` |
| Weighted-F1 | 0.8761 | same |
| MCC | 0.8662 | same |
| Pooled test samples | 10,662 | official 5-fold pooled test predictions |

Important caveat: TTA improved the point estimate over the base exp 11 model
(0.6075 vs. 0.6000 macro-F1), but the 95% confidence intervals overlap. It is
reported as a useful zero-training-cost inference improvement, not as a
statistically conclusive gain.

## Architecture

```text
Image 224x224
   |
   +-- ResNet50        -> 2048-d feature -> Linear + LayerNorm + GELU -> 512-d
   +-- MobileNetV2     -> 1280-d feature -> Linear + LayerNorm + GELU -> 512-d
   +-- EfficientNetB0  -> 1280-d feature -> Linear + LayerNorm + GELU -> 512-d
                                                        |
                                                fusion module
                                      concat / weighted / GMU / AFF / LMF
                                                        |
                                                MLP classifier
                                                        |
                                                  23 classes
```

Design constraints:

- The required backbones are loaded through `torchvision`, not `timm`.
- The classifier is always an MLP. SVM, RandomForest, and XGBoost are not used
  as official baselines.
- Frozen backbones have `requires_grad=False`; BatchNorm statistics remain
  frozen during fine-tuning unless an explicit ablation changes that.
- Fine-tuning updates the last three backbone blocks in the main recipe.

## Dataset

Primary dataset: HyperKvasir labeled image subset.

| Property | Value |
|---|---|
| Classes | 23 |
| Images | 10,662 |
| Task | Multi-class gastrointestinal endoscopy image classification |
| License | CC BY 4.0 |
| Main protocol | Official HyperKvasir 5-fold split |
| Image size used here | 224x224 |
| Normalization | ImageNet mean/std |

The dataset is severely imbalanced. Some classes contain hundreds or more than
one thousand examples, while rare classes have only a few samples. For this
reason, macro-F1 is the headline metric and accuracy is only a supporting metric.

Expected local data layout:

```text
<DATA_ROOT>/
  hyperkvasir/
    labeled-images/
      class_a/
      class_b/
      ...
```

The dataset root is configured through `DATA_ROOT` and
`configs/dataset/hyperkvasir_23class_official.yaml`.

## Repository Layout

```text
configs/                 YAML configs for datasets, methods, training, matrix
data/splits/             Official and materialized fold manifests
docs/                    Decisions, experiment logs, final model record
env/                     Colab and environment export files
references/              Paper markdown and metadata used by the report
reports/final/           LaTeX report sources and final figures
results/figures/         Report-ready generated figures
results/runs/            Metrics, predictions, logs, and local checkpoints
results/tables/          Ablation tables, CV summaries, CIs, per-class tables
scripts/                 CLI entry points for data, training, eval, analysis
src/                     Core data/model/training/evaluation implementation
tests/                   Unit and smoke tests
```

Large local artifacts are intentionally gitignored:

- raw datasets under `data/raw/` or external `DATA_ROOT`
- feature caches under `results/feature_cache/`
- checkpoints such as `best.pt`, `last.pt`, `ema.pt`
- reference PDFs and extracted heavy assets

## Environment

The project uses `uv` and Python 3.11.

Validated local stack:

- Python 3.11.9
- PyTorch 2.12.0+cu132
- torchvision 0.27.0+cu132
- NVIDIA GeForce RTX 5080 Laptop GPU

Install dependencies:

```bash
uv sync
```

Check the environment:

```bash
uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Run the full test suite:

```bash
uv run pytest tests/ -q
```

Colab fallback:

```bash
pip install -r env/requirements-colab.txt
```

## Data Preparation

Set the dataset root:

```bash
# PowerShell
$env:DATA_ROOT="D:\datasets"
```

Create the image manifest:

```bash
uv run python scripts/prepare_data.py \
  --dataset hyperkvasir \
  --config configs/dataset/hyperkvasir_23class_official.yaml
```

Materialize official 5-fold train/validation/test manifests:

```bash
uv run python scripts/make_splits.py \
  --config configs/dataset/hyperkvasir_23class_official.yaml
```

The official protocol uses each fold once as test, the next fold as validation,
and the remaining three folds as training. The materialized files live under:

```text
data/splits/hyperkvasir_official_5fold/fold_0.csv
data/splits/hyperkvasir_official_5fold/fold_1.csv
...
data/splits/hyperkvasir_official_5fold/fold_4.csv
```

## Running Experiments

All reportable experiments are declared in:

```text
configs/experiment_matrix.yaml
```

Train one experiment:

```bash
uv run python scripts/train.py \
  --config configs/experiment_matrix.yaml \
  --experiment 11_triple_weighted_finetune_wide_official \
  --fold 0 \
  --device cuda
```

Run the selected 5-fold CV experiment:

```bash
uv run python scripts/run_cv.py \
  --config configs/experiment_matrix.yaml \
  --experiment 11_triple_weighted_finetune_wide_official \
  --folds 0 1 2 3 4 \
  --device cuda
```

Run deterministic TTA evaluation for the frozen final model:

```bash
uv run python scripts/evaluate.py \
  --experiment 11_triple_weighted_finetune_wide_official \
  --folds 0 1 2 3 4 \
  --tta \
  --device cuda
```

Aggregate 5-fold summaries and confidence intervals:

```bash
uv run python scripts/aggregate_cv.py \
  --experiment 11_triple_weighted_finetune_wide_official

uv run python scripts/compute_ci.py \
  --experiment 11_triple_weighted_finetune_wide_official \
  --predictions predictions_tta.npz
```

Generate report-ready figures and tables from saved predictions/logs:

```bash
uv run python scripts/generate_report_tables.py
uv run python scripts/analyze_frozen_model.py
uv run python scripts/plot_results.py
```

## Experiment Matrix Summary

The experiments cover:

| Axis | Covered variants |
|---|---|
| Single CNN | ResNet50, MobileNetV2, EfficientNetB0 |
| Pair fusion | R+M concat, R+E concat, M+E concat |
| Triple fusion | concat, weighted, GMU |
| Transfer learning | frozen extraction, fine-tune last 3 blocks |
| Extra ablations | focal loss, TTA, leakage-free seed ensemble |

Key selected 5-fold results before TTA:

| Experiment | Method | Fusion | CV macro-F1 |
|---|---|---|---:|
| `13_single_efficientnetb0_finetune_wide_official` | EfficientNetB0 | none | 0.5690 +/- 0.0158 |
| `14_pair_m_e_finetune_wide_official` | MobileNetV2 + EfficientNetB0 | concat | 0.5670 +/- 0.0089 |
| `10_triple_concat_finetune_wide_official` | R + M + E | concat | 0.5691 +/- 0.0064 |
| `11_triple_weighted_finetune_wide_official` | R + M + E | weighted | 0.5892 +/- 0.0102 |
| `15_triple_gmu_finetune_wide_official` | R + M + E | GMU | 0.5672 +/- 0.0049 |

The weighted triple fusion model is the best 5-fold CV configuration. GMU was
implemented and tested, but did not outperform simpler fusion under this
protocol.

## Final Inference Recipe

The final model uses deterministic 4-view TTA:

1. `base`: Resize(256) -> CenterCrop(224)
2. `hflip`: base + horizontal flip
3. `scale`: Resize(224) -> CenterCrop(224)
4. `scale_hflip`: scale + horizontal flip

Softmax probabilities are averaged per image, then argmax is taken. The TTA
policy is implemented in `scripts/evaluate.py` and documented in
`docs/FINAL_MODEL.md`.

## Training Recipe

Main fine-tuning config: `configs/training/finetune_wide.yaml`.

| Setting | Value |
|---|---|
| Epoch budget | 60 |
| Batch size | 64 |
| Mixed precision | enabled |
| Unfrozen blocks | last 3 blocks |
| Optimizer | AdamW |
| Head LR | 1e-3 |
| Backbone LR | 1e-4 |
| LLRD decay | 0.75 |
| Scheduler | cosine with 5 warmup epochs |
| Loss | cross-entropy with label smoothing 0.1 |
| Augmentation | RandAugment, horizontal flip, CutMix |
| EMA | decay 0.999, start epoch 5 |
| Early stopping | validation macro-F1, patience 8 |
| Sampler | weighted sampler from training fold only |

Focal loss was tested as an ablation in
`configs/training/finetune_wide_focal.yaml`. It did not beat the cross-entropy
champion.

## Result Artifacts

Useful files for report and audit:

```text
docs/FINAL_MODEL.md
docs/results_progress.md
docs/decisions.md
results/tables/ablation_table.md
results/tables/per_class_frozen_tta.md
results/tables/training_time.md
results/tables/ci_11_triple_weighted_finetune_wide_official_tta.json
results/tables/extra_metrics_11_triple_weighted_finetune_wide_official_tta.json
results/figures/confusion_matrix_frozen_tta.png
results/figures/per_class_f1_frozen_tta.png
results/figures/comparison_bar_chart.png
reports/final/sections/
reports/final/figures/
```

Each headline metric in the report should trace to either a `metrics.json`,
`results/tables/*.json`, or a documented paper location under `references/`.

## Testing

Run everything:

```bash
uv run pytest tests/ -q
```

The suite covers:

- backbone feature dimensions and frozen BatchNorm behavior
- projection and fusion modules
- frozen-head training path
- fine-tuning trainer smoke behavior
- metrics and statistical confidence intervals
- focal loss
- EMA, LLRD, TTA, ensembling
- data split and feature-cache utilities

Latest local verification during README refresh:

```text
176 passed
```
## Design Integrity Rules

The following rules are intentional and should not be changed silently:

- Use `uv sync` and `uv run ...`; do not manually maintain `requirements.txt`.
- Keep final model logic under `src/`.
- Use torchvision for ResNet50, MobileNetV2, and EfficientNetB0.
- Keep the classifier fixed as an MLP.
- Do not average across folds before computing macro-F1 unless pooling
  predictions for a documented pooled estimate.
- Do not compare against external papers as direct baselines unless dataset,
  split, class count, metric, and architecture protocol match.
- Do not commit raw data, feature caches, checkpoints, or large generated
  artifacts.

## References and Documentation

Start here:

- `project_structure.md`: architecture facts and module contracts
- `project_plan.md`: locked project decisions and experiment plan
- `docs/FINAL_MODEL.md`: frozen final model and inference recipe
- `docs/results_progress.md`: chronological experiment record
- `docs/decisions.md`: locked decisions and later rationale
- `references/INDEX.md`: paper index and comparability caveats

Primary dataset citation:

- Borgli et al. 2020, "HyperKvasir, a comprehensive multi-class image and video
  dataset for gastrointestinal endoscopy", Scientific Data.
