# Week 1 Foundation Execution Plan

## 1. Goal

Establish the Week 1 Foundation milestone for the HyperKvasir multi-CNN fusion
project. By the end of this plan, the project should have a verified `uv`
environment, a working HyperKvasir data pipeline, saved split manifests, verified
torchvision backbone wrappers, a frozen feature-cache smoke path, and one
end-to-end Single ResNet50 frozen experiment that saves non-NaN metrics.

This is an execution plan only. Do not implement model, data, or training code
while creating this file.

## 2. Inputs

- `AGENTS.md`
- `project_structure.md`
- `project_plan.md`
- `docs/decisions.md`
- `docs/environment.md`
- `docs/experiment_log.md`
- `docs/known_issues.md`
- Current scaffold under `src/`, `scripts/`, `configs/`, and `tests/`
- HyperKvasir 23-class dataset config:
  `configs/dataset/hyperkvasir_23class_official.yaml`
- Official HyperKvasir 5-fold split source:
  `data/splits/official/5_fold_split.csv`
- Stage 0 experiment target:
  `01_single_resnet50_frozen_official`

## 3. Scope

- Set up the local `uv` environment and verify Python 3.11.
- Verify PyTorch, torchvision, and CUDA availability on the RTX 5080 laptop.
- Implement the HyperKvasir 23-class data pipeline.
- Log total sample count and per-class counts.
- Generate official HyperKvasir 5-fold split manifests.
- Check split manifests for train/validation/test leakage.
- Implement torchvision backbone wrappers for:
  - ResNet50
  - MobileNetV2
  - EfficientNetB0
- Verify documented penultimate feature dimensions:
  - ResNet50: 2048
  - MobileNetV2: 1280
  - EfficientNetB0: 1280
- Implement a frozen feature-cache smoke path using `results/feature_cache/`.
- Run one end-to-end frozen single-CNN experiment:
  `01_single_resnet50_frozen_official`.
- Ensure `uv run pytest tests/` passes.
- Ensure the single-CNN run saves non-NaN metrics.

## 4. Out of Scope

- GMU implementation.
- AFF or LMF implementation.
- DeepWeeds or other secondary datasets.
- Effimix or GastroViT cross-protocol reproduction.
- Full one-fold ablation matrix.
- Selected 5-fold cross-validation.
- Non-MLP classifiers.
- Report-ready result tables and figures beyond basic run artifacts.
- Grad-CAM++, UMAP, bootstrap confidence intervals, or McNemar analysis.
- Any change to locked architecture decisions or verified facts.

## 5. Files Expected To Be Modified Later

- `src/data/datasets.py`
- `src/data/manifests.py`
- `src/data/splits.py`
- `src/data/transforms.py`
- `src/data/feature_cache.py`
- `src/models/backbones.py`
- `src/models/projections.py`
- `src/models/classifiers.py`
- `src/models/full_model.py`
- `src/training/trainer.py`
- `src/training/losses.py`
- `src/training/optimizers.py`
- `src/training/schedulers.py`
- `src/evaluation/metrics.py`
- `src/utils/logging.py`
- `src/utils/checkpointing.py`
- `src/utils/paths.py`
- `scripts/prepare_data.py`
- `scripts/make_splits.py`
- `scripts/extract_features.py`
- `scripts/train.py`
- `configs/dataset/hyperkvasir_23class_official.yaml`
- `configs/experiment_matrix.yaml`
- Relevant tests under `tests/`
- `docs/environment.md`
- `docs/experiment_log.md`
- `docs/known_issues.md`
- `docs/decisions.md`, only if a locked decision, split policy, dataset
  definition, or evaluation rule must change.

## 6. Step-by-Step Implementation Plan

1. Run environment setup.
   - Run `uv sync`.
   - Confirm Python resolves to 3.11 from `.python-version`.
   - Confirm the lockfile and installed environment are usable.

2. Verify CUDA and package versions.
   - Check `torch.__version__`.
   - Check `torchvision.__version__`.
   - Check `torch.cuda.is_available()`.
   - If CUDA is unavailable, log the blocker in `docs/known_issues.md` and
     continue CPU-only smoke tests where feasible.

3. Log environment details.
   - Record Python version, OS, torch, torchvision, CUDA availability, GPU name
     if available, and relevant driver/CUDA notes in `docs/environment.md`.

4. Implement HyperKvasir manifest creation.
   - Read the dataset root from `configs/dataset/hyperkvasir_23class_official.yaml`.
   - Discover JPEG images under the labeled subset.
   - Store image path, label index, class name, and original index.
   - Log total sample count and per-class counts.
   - Validate that the configured class count is 23.

5. Implement preprocessing and transforms.
   - Use input size 224 for all backbones.
   - Use ImageNet normalization:
     `mean=[0.485, 0.456, 0.406]`,
     `std=[0.229, 0.224, 0.225]`.
   - Use resize-shortest to 256 and center crop for validation/test.
   - Use resize-shortest to 256 and random crop for training.

6. Implement split manifest generation.
   - Implement `make_stratified_folds(samples, n_splits, seed, output_dir)`.
   - Use the official HyperKvasir 5-fold split source as the primary protocol.
   - Materialize each fold as: test = official fold `k`, validation =
     official fold `(k + 1) % 5`, training = remaining official folds.
   - Save manifests under `data/splits/`.
   - Include path, label, class name, split, fold, and original index.
   - Add explicit leakage checks so the same original index/path cannot appear
     in conflicting splits.
   - If ultra-rare classes make stratification impossible, stop and document the
     issue before changing split policy.

7. Implement backbone wrappers.
   - Implement `BackboneFeatureExtractor` in `src/models/backbones.py`.
   - Use torchvision only for ResNet50, MobileNetV2, and EfficientNetB0.
   - Do not use `timm` for these three required backbones.
   - Verify feature outputs:
     - `resnet50`: `(B, 2048)`
     - `mobilenetv2`: `(B, 1280)`
     - `efficientnetb0`: `(B, 1280)`
   - Freeze parameters when `unfreeze_blocks=0`.
   - Preserve BatchNorm freeze rules for future fine-tuning work.

8. Implement minimal single-CNN frozen model path.
   - Use backbone features, projection layer, and MLP classifier.
   - Keep classifier as MLP only.
   - Use config-driven hyperparameters.
   - Save run artifacts under `results/runs/{exp_id}/`.

9. Implement frozen feature-cache smoke path.
   - Use `results/feature_cache/` as the cache location.
   - Cache features, labels, original indices, image paths, class names, and a
     config hash.
   - Validate feature row count equals manifest row count.
   - Validate labels and image paths remain aligned.

10. Run the first end-to-end experiment.
    - Run `01_single_resnet50_frozen_official`.
    - Confirm the run saves `config.yaml`, `metrics.json`, predictions, and the
      split manifest used for the run.
    - Confirm metrics contain finite values and no NaN values.

11. Run tests and document outcomes.
    - Run `uv run pytest tests/`.
    - Update `docs/experiment_log.md` with successful split/cache/train smoke
      results.
    - Update `docs/known_issues.md` for any unresolved blocker.
    - Update `docs/decisions.md` only if a locked decision or policy changes.

## 7. Risks

- RTX 5080, CUDA, PyTorch, or torchvision compatibility may fail locally.
- HyperKvasir may be unavailable, missing, or stored in a path that differs from
  the current dataset config.
- Severe class imbalance makes fold metrics fragile even with official splits.
- Ultra-rare classes may have very small validation/test support in some folds.
- Official split source files must match local raw dataset filenames and class
  folder names.
- Incorrect torchvision layer indexing could silently produce wrong features.
- Feature cache rows, labels, image paths, or indices may become misaligned.
- BatchNorm freeze behavior may be accidentally violated in future fine-tuning
  code if not tested early.
- Scope creep into Week 2 methods could delay the Week 1 checkpoint.
- Generated data, feature caches, checkpoints, or run artifacts may accidentally
  be added to git if ignore rules are not checked.

## 8. Acceptance Criteria

- `uv sync` completes successfully.
- Python 3.11 is used by the project environment.
- CUDA availability, torch version, torchvision version, and GPU information are
  logged in `docs/environment.md`.
- HyperKvasir sample count and 23 per-class counts are logged.
- Split manifests are generated under `data/splits/` from the official
  HyperKvasir 5-fold split source.
- Split manifests pass leakage checks.
- Backbone wrapper tests verify output feature dimensions for ResNet50,
  MobileNetV2, and EfficientNetB0.
- Feature-cache smoke test validates feature count, labels, image paths, class
  names, indices, and config hash presence.
- `uv run pytest tests/` passes.
- `01_single_resnet50_frozen_official` completes.
- The single-CNN run saves metrics and the metrics contain no NaN values.

## 9. Commands To Run

```bash
uv sync
```

```bash
uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

```bash
uv run pytest tests/
```

```bash
uv run python scripts/prepare_data.py --dataset hyperkvasir
```

```bash
uv run python scripts/make_splits.py --config configs/dataset/hyperkvasir_23class_official.yaml
```

```bash
uv run python scripts/extract_features.py --config configs/dataset/hyperkvasir_23class_official.yaml
```

```bash
uv run python scripts/train.py --config configs/experiment_matrix.yaml --experiment 01_single_resnet50_frozen_official
```

## 10. Documentation Updates Required

- Update `docs/environment.md` with:
  - Python version
  - OS
  - torch version
  - torchvision version
  - CUDA availability
  - GPU name and driver/CUDA notes if available

- Update `docs/experiment_log.md` after:
  - environment verification
  - manifest generation
  - split leakage check
  - feature-cache smoke test
  - first successful Single ResNet50 frozen run

- Update `docs/known_issues.md` for:
  - CUDA/PyTorch compatibility problems
  - dataset path/download problems
  - rare-class split failures
  - feature-cache alignment issues
  - any failed test or incomplete acceptance criterion

- Update `docs/decisions.md` only if:
  - a locked design decision changes
  - split policy changes
  - dataset class definition changes
  - evaluation rules change

## Assumptions

- Python remains pinned to 3.11.
- Week 1 uses HyperKvasir only.
- Classifier remains MLP only.
- Required backbones use torchvision only.
- `results/feature_cache/` is the feature-cache location.
- The first end-to-end run is `01_single_resnet50_frozen_official`.
- No authoritative reference artifact files are created or edited as part of
  Week 1 planning.
