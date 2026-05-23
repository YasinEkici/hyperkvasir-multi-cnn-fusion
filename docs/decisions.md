# Decisions

Project decisions are locked in `project_structure.md` and `project_plan.md`.

## 2026-05-23 - HyperKvasir Split Protocol

- Use the official HyperKvasir 5-fold split file as the primary Week 1 split
  protocol.
- Store the official split source under `data/splits/official/`.
- Keep any locally generated stratified split support as a fallback/internal
  utility only, not as the main reporting protocol.
- Materialize train/validation/test manifests deterministically from official
  folds: for fold `k`, use official fold `k` as test, official fold
  `(k + 1) % 5` as validation, and the remaining folds as training.
- The local raw dataset class folder names are aligned to the official class
  names, so no class-name mapping is required in the data pipeline.
