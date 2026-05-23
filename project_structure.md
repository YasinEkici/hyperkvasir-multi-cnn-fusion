# Project Structure — Multi-CNN Feature Fusion for HyperKvasir Classification

> **HOW TO USE THIS DOCUMENT (READ FIRST)**
>
> This file is the **single source of truth** for the project's file layout,
> module contracts, architectural facts, and naming conventions. It is given
> to coding agents (Codex, Cursor, Claude Code) at the start of every session.
>
> **Rules for the agent:**
> 1. Every fact in the "Verified Facts" section was confirmed against live
>    library execution or original papers. Do not "correct" them.
> 2. Every module signature in "Module Contracts" is binding. Do not invent
>    new function names; if the signature feels wrong, ask the human.
> 3. Every entry in "Forbidden Patterns" has a documented reason. Do not work
>    around them silently.
> 4. When in doubt, prefer the option already locked in this document over
>    introducing a new alternative.

---

## 1. Project Goal (one paragraph)

Build a multi-CNN feature-fusion classifier for the **HyperKvasir 23-class
labeled subset** (gastrointestinal endoscopy). Three pre-trained backbones —
**ResNet50, MobileNetV2, EfficientNetB0** — are used as feature extractors;
their penultimate features are fused (concatenation, weighted, attention-based)
and classified by an MLP. Mandatory ablations cover 1/2/3-backbone combinations
and frozen vs. fine-tuned backbones. Cross-protocol comparison reproduces the
**Effimix (Ramamurthy 2022)** and **GastroViT (2025)** evaluation protocols on
the same dataset for direct head-to-head numbers.

Secondary dataset (domain-generality test): **DeepWeeds** (9-class, agricultural).

---

## 2. Verified Facts (DO NOT MODIFY)

These facts were confirmed against `torchvision==0.20+` source and original
papers. The agent must use these exact values.

### 2.1 Backbone architecture (torchvision implementations)

| Property                             | ResNet50            | MobileNetV2                   | EfficientNetB0      |
|--------------------------------------|---------------------|-------------------------------|---------------------|
| Top-level children                   | conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc | features (Sequential of 19), classifier (Sequential of 2) | features (Sequential of 9), avgpool, classifier (Sequential of 2) |
| Penultimate feature dim              | **2048**            | **1280**                      | **1280**            |
| Penultimate access path              | `model.avgpool(model.layer4(...)).flatten(1)` | `model.features(x).mean([2,3])` | `model.avgpool(model.features(x)).flatten(1)` |
| InvertedResidual / bottleneck count  | layer3 = 6 bottlenecks, layer4 = 3 bottlenecks | **17** InvertedResidual blocks (`features[1..17]`) — **NOT 19**, off-by-2 in original paper | 7 MBConv stages (`features[1..7]`) with sub-block counts (1,2,2,3,3,4,1) |
| `features[0]`                        | n/a                 | Conv2dNormActivation (3×3 stem) | Conv2dNormActivation (stem) |
| `features[-1]` (last index)          | n/a                 | `features[18]`: 1×1 Conv 320→1280 | `features[8]`: 1×1 Conv → 1280 |

### 2.2 Fine-tuning unfreeze targets ("last 2-3 conv blocks")

| Backbone        | Last 2 blocks unfreeze       | Last 3 blocks unfreeze        |
|-----------------|------------------------------|-------------------------------|
| ResNet50        | `layer4` (3 bottlenecks)     | `layer3` + `layer4` (6+3=9)   |
| MobileNetV2     | `features[15:]`              | `features[13:]`               |
| EfficientNetB0  | `features[6:9]`              | `features[5:9]`               |

Frozen layers: `requires_grad=False` AND set to `.eval()` mode each epoch.
BatchNorm running stats are **frozen** during fine-tuning unless an
ablation explicitly tests otherwise.

### 2.3 Image preprocessing

- Input resolution: **224×224** for all three backbones (controlled comparison).
- Normalization: ImageNet stats — `mean=[0.485, 0.456, 0.406]`,
  `std=[0.229, 0.224, 0.225]`.
- HyperKvasir native resolution is variable (mostly 720×576), so resize-shortest
  to 256 then center/random crop to 224.

### 2.4 HyperKvasir labeled subset

- **23 classes**, **10,662 labeled images**, JPEG.
- Source: `https://datasets.simula.no/hyperkvasir/`
- License: CC BY 4.0
- Class imbalance is severe (e.g., polyps ≈1028 images, hemorrhoids ≈6 images).

### 2.5 DeepWeeds

- **9 classes** (8 weed species + negatives), **17,509 images**, 256×256.
- Source: `https://github.com/AlexOlsen/DeepWeeds`
- License: CC BY 4.0
- ResNet-50 baseline (Olsen et al. 2019, *Sci. Reports* 9:2058): 95.7%.

---

## 3. Hardware & Environment

- **Primary:** RTX 5080 laptop GPU (16 GB VRAM, Blackwell), 64 GB RAM.
- **Secondary:** Google Colab Pro+ for parallel runs and backup.
- **Python:** 3.11
- **Stack:** torch≥2.5.0, torchvision≥0.20.0, numpy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, timm, albumentations, grad-cam, umap-learn
- Environment managed by `uv`. See `AGENTS.md` for commands.

---

## 4. File Hierarchy

```
multi-cnn-fusion/
├── AGENTS.md                          # READ-FIRST for all agents
├── README.md
├── project_structure.md               # THIS FILE — single source of truth
├── project_plan.md                    # timeline, PLDs, risk register
├── pyproject.toml
├── uv.lock
├── .python-version
│
├── docs/
│   ├── decisions.md                   # append-only log of design decisions
│   ├── experiment_log.md              # chronological run notes
│   └── exec-plans/                    # weekly execution plans
│       ├── TEMPLATE.md
│       ├── active/                    # the current week's plan
│       │   └── {YYYY}-W{NN}-{slug}.md
│       └── completed/                 # finished week plans move here
│           └── {YYYY}-W{NN}-{slug}.md
│
├── references/                        # Per-paper folders from extraction tool
│   ├── INDEX.md                       # ordered list of all references
│   ├── primary_baselines/
│   │   ├── 01_ramamurthy_2022_effimix/
│   │   ├── 02_gastrovit_2025/
│   │   ├── 03_borgli_2020_hyperkvasir_dataset/
│   │   ├── 04_ahmed_2023_hybrid_fusion/
│   │   ├── 05_he_2025_hemf/
│   │   ├── 06_ramachandran_2025_interpretable/
│   │   ├── 07_guo_2024_curriculum_ssl/
│   │   └── 08_montalbo_2024_mfure_cnn/
│   ├── methodology_fusion/
│   │   ├── aff_dai_2021_wacv/
│   │   ├── gmu_arevalo_2017_iclr_workshop/
│   │   ├── lmf_liu_2018_acl/
│   │   ├── se_block_hu_2018_cvpr/
│   │   └── cbam_woo_2018_eccv/
│   ├── methodology_training/
│   │   ├── cutmix_yun_2019_iccv/
│   │   ├── label_smoothing_muller_2019_neurips/
│   │   ├── llrd_howard_ruder_2018_acl/
│   │   ├── adamw_loshchilov_2019_iclr/
│   │   ├── cosine_annealing_loshchilov_2017/
│   │   ├── swa_izmailov_2018_uai/
│   │   ├── randaugment_cubuk_2020/
│   │   └── dropblock_ghiasi_2018_neurips/
│   ├── methodology_imbalance/
│   │   ├── focal_loss_lin_2017_iccv/
│   │   ├── class_balanced_loss_cui_2019_cvpr/
│   │   └── weighted_random_sampler_torch/
│   ├── methodology_evaluation/
│   │   ├── mcnemar_dietterich_1998/
│   │   ├── gradcam_pp_chattopadhyay_2018_wacv/
│   │   ├── gradcam_selvaraju_2017/
│   │   └── umap_mcinnes_2018_joss/
│
├── configs/
│   ├── dataset/
│   ├── method/
│   ├── training/
│   └── experiment_matrix.yaml
│
├── data/                              # gitignored — raw dataset
├── results/                           # gitignored — run outputs
│
├── src/
│   ├── data/
│   ├── models/
│   │   ├── backbones.py
│   │   ├── projections.py
│   │   ├── fusion/
│   │   ├── classifiers.py
│   │   └── full_model.py
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
├── notebooks/
├── tests/
└── env/
```

---

## 5. References Folder

The `references/` folder contains one folder per paper, organized by category.
Each paper folder is produced by our local PDF extraction tool and contains:

```
{paper_folder}/
├── assets/
│   ├── figures/         # extracted figure images (.jpeg)
│   └── pictures/        # extracted cover/other images (.jpeg)
├── metadata.yaml        # bibliographic metadata
├── original.pdf         # the source PDF
└── paper.md             # auto-extracted markdown — structure mirrors the original paper
```

**Do not edit these files.** They are the authoritative artifacts. `paper.md`
preserves the paper's section structure, tables, and inline figure references —
which is exactly what we need when implementing a method or citing a number
in the report. When `paper.md` is ambiguous or has OCR artifacts, spot-check
against `original.pdf` in the same folder.

### Folder name convention

`{NN}_{first_author_lastname}_{year}_{slug}/`

Examples:
- `01_ramamurthy_2022_effimix/`
- `03_borgli_2020_hyperkvasir_dataset/`
- `gmu_arevalo_2017_iclr_workshop/` (no `NN` prefix outside `primary_baselines/`)

`NN` is a two-digit zero-padded order index within `primary_baselines/` only,
for sort stability in cross-reference tables. Other categories don't use `NN`.
Lowercase, no spaces, no special characters.

### `references/INDEX.md`

A short index file lists every paper folder with one-line rationale and the
PLD numbers it supports. Updated when papers are added or removed.

---

## 6. Module Contracts

(Function/class signatures from the prior project_structure.md remain binding.
Reproduced briefly here; refer to the prior version for full docstrings.)

- `src/models/backbones.py::BackboneFeatureExtractor(name, pretrained, unfreeze_blocks)`
- `src/models/projections.py::BranchProjection(in_dim, out_dim=512)`
- `src/models/fusion/{concat,weighted,gmu,aff,lmf}.py::FusionModule(num_branches, feature_dim, **kwargs)`
- `src/models/classifiers.py::MLPClassifier(input_dim, num_classes, hidden_dims, dropout)`
- `src/models/full_model.py::MultiCNNFusionClassifier(backbone_names, unfreeze_blocks, projection_dim, fusion_type, num_classes, ...)`
- `src/data/splits.py::make_stratified_folds(samples, n_splits, seed, output_dir)`
- `src/data/feature_cache.py::cache_frozen_features(backbones, dataset_config, split, output_dir, ...)`
- `src/training/optimizers.py::build_adamw_with_llrd(model, head_lr, backbone_lr, weight_decay, llrd_decay=0.75)`
- `src/evaluation/statistical.py::mcnemar_test(preds_a, preds_b, labels, exact_threshold=25)`
- `src/evaluation/statistical.py::bootstrap_ci(metric_fn, preds, labels, n_bootstrap=1000, ci=0.95, seed=42)`

---

## 7. Config Schema

YAML-based configs under `configs/dataset/`, `configs/method/`,
`configs/training/`, with master `configs/experiment_matrix.yaml`. See
prior project_structure.md for the field-level schema.

---

## 8. Naming Conventions

- **Experiment IDs:** `{NN}_{method}_{training_mode}_{protocol}` — e.g., `12_triple_gmu_finetune_own`.
- **Checkpoints:** `best.pt`, `last.pt`, `ema.pt`. Never `model.pt`.
- **Logs:** structured JSON, one line per epoch, in `results/runs/{id}/logs/`.
- **Paper folders:** `{NN}_{first_author}_{year}_{slug}/` (see §5).
- **Exec plan files:** `{YYYY}-W{NN}-{slug}.md` — e.g., `2026-W1-foundation.md`.
- **Git branches:** `feat/w{NN}-{slug}` matching the exec plan slug.
- **Variables:** snake_case for functions/variables, PascalCase for classes.

---

## 9. Forbidden Patterns

1. Hard-coded numerical values that should come from configs.
2. Using `timm` for the three required backbones (use torchvision only).
3. Updating BatchNorm running stats during fine-tuning by default.
4. Adding non-MLP classifiers as official project baselines.
5. Averaging across folds before computing macro-F1.
6. Reproducing competitor protocols by inventing splits.
7. Modifying verified facts in §2 silently.
8. Saving raw datasets, feature caches, or checkpoints to git.
9. Hard-coded path separators (use `pathlib.Path`).
10. Mixing `print()` and project logger.
11. Editing files inside `references/{paper_folder}/` — they are read-only authoritative artifacts.
12. Citing a numerical claim in the report without a paper-and-location reference (e.g., "Ramamurthy 2022, Table 3") or a `metrics.json` reference.

---

## 10. Working Style for Agents

1. Read `AGENTS.md` first, then this file, then `project_plan.md`, then the active exec plan, then the relevant `paper.md` files.
2. Implement one module at a time. Write or update the matching test first.
3. For each implemented method, open the relevant `paper.md`, locate the equations, and copy them verbatim into the implementing module's docstring before writing code.
4. Run the relevant tests with `uv run pytest tests/...`.
5. Save run artifacts: config, metrics, predictions, split manifest.
6. Ask before deviating from scope or locked decisions.
