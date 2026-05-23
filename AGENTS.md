# AGENTS.md

This repository implements a multi-CNN feature-fusion classifier for the
HyperKvasir 23-class gastrointestinal endoscopy benchmark, as the term project
for a Deep Learning course (deadline: 8 June 2026).

## Read first

Always read these documents in order before starting any non-trivial task:

1. `project_structure.md` — file layout, module contracts, verified architecture facts
2. `project_plan.md` — timeline, locked design decisions (PLD-*), risk register
3. `docs/exec-plans/active/{current-week}.md` — the active execution plan for this week
4. `references/INDEX.md` — bibliography of papers in this project

For tasks involving a specific method or comparison paper, also read:

- `references/{category}/{paper_folder}/paper.md` — the auto-extracted markdown of the relevant paper
- `references/{category}/{paper_folder}/original.pdf` — only when paper.md is ambiguous or has OCR artifacts

For task history, locked decisions, and rationale, also check:

- `docs/decisions.md`
- `docs/exec-plans/completed/` (previous weeks' plans)

## Ground rules

- Use `uv sync` for setup and `uv run ...` for commands.
- Do not manually maintain `requirements.txt`. Use `pyproject.toml` + `uv.lock`.
- Keep core logic under `src/`.
- Notebooks are exploratory only; final model logic and training code go in `src/`.
- Do not commit large data, checkpoints, or run artifacts. Use `.gitignore`.
- Use `torchvision` only for ResNet50, MobileNetV2, and EfficientNetB0. Do not use `timm` for these three — layer indexing differs and breaks Section 2 of `project_structure.md`.
- The classifier is **MLP only**. Do not add SVM, RandomForest, or XGBoost classifiers as baselines, even if a referenced paper uses them.
- During fine-tuning, BatchNorm running statistics stay frozen by default. Use `.eval()` on BN layers in frozen blocks. Updating BN must be an explicit ablation.
- Frozen feature extraction runs use cached features (`results/feature_cache/`). Do not re-run backbones over the dataset for every experiment.
- Cross-protocol experiments (Effimix, GastroViT) must not invent missing split details. Document any approximate reproduction in `docs/decisions.md`.
- Update `docs/decisions.md` when changing locked decisions (PLD-*), data splits, dataset class definitions, or evaluation rules. Locked decisions cannot be changed silently.

## Working with references

- The `references/` folder contains one folder per paper, produced by our PDF extraction tool. Each paper folder contains `original.pdf`, `metadata.yaml`, `paper.md`, and `assets/` (extracted figures and pictures). **Do not edit these files** — they are the authoritative artifacts.
- When the active exec plan says "read X paper" for a task, open the corresponding `paper.md`. Its structure mirrors the paper itself: section headings, tables, equations, and inline figure references all preserved.
- When implementing a method (GMU, AFF, LMF, etc.), open the relevant `paper.md`, locate the equations, and copy them verbatim into the implementing module's docstring. Preserve the paper's notation (variable names, subscripts, indices). Do not paraphrase, do not rename.
- When `paper.md` is ambiguous (broken OCR equation, garbled number, missing table caption), spot-check against `original.pdf` in the same folder. Note the discrepancy in a code comment or in `docs/decisions.md` if it affects implementation.
- Every numerical claim that ends up in the final report must trace back to either (a) a `metrics.json` file in `results/runs/{exp_id}/` produced by our pipeline, or (b) a specific Section/Table/Figure/Equation in a `paper.md`. Cite the source location (e.g., "Ramamurthy 2022, Table 3") next to the claim in the report.
- Do not claim that an experiment reproduces a paper unless the dataset, split, class count, metrics, and architecture match. If approximate, say "approximate reproduction" and document the deviation in `docs/decisions.md`.

## Task workflow

For non-trivial tasks:

1. Read the relevant docs listed above.
2. Read the active execution plan under `docs/exec-plans/active/`.
3. Before editing, post a short plan that includes:
   - task goal,
   - expected file changes,
   - acceptance criteria from the active exec-plan,
   - risks or open questions.
4. Wait for user approval if the task changes module contracts, locked decisions, or affects more than 3 files outside `src/`.
5. Keep changes limited to the current task scope.
6. Run the required `uv run pytest tests/...` and document the result.
7. Update relevant docs when behavior, labels, paths, or evaluation rules change.
8. When the task is complete, append a brief entry to `docs/experiment_log.md` or `docs/decisions.md` as appropriate.

## Stop-and-ask triggers

Stop and ask the user before proceeding if:

- A paper equation in `paper.md` is ambiguous, or `paper.md` and `original.pdf` disagree on a critical equation or hyperparameter.
- Installed library behavior differs from the verified facts in `project_structure.md` §2.
- A test fails after an attempted fix.
- A new dependency outside the locked stack is needed.
- You want to simplify a locked decision (PLD-*).
- A competitor protocol's split cannot be reconstructed from the paper.
