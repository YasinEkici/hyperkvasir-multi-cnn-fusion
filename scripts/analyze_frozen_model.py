"""Week 4 Track A — final report figures/tables for the FROZEN model.

INFERENCE-FREE / DATA-ONLY: reads saved predictions (predictions_tta.npz),
epoch logs (epoch_log.jsonl), and fold manifests. Never loads a checkpoint, runs
the model, or trains.

Produces, for the frozen model (exp 11 triple weighted CE + TTA, pooled n=10,662):
  A1  results/figures/confusion_matrix_frozen_tta.png   (23x23, class-labelled)
  A2  results/tables/per_class_frozen_tta.csv / .md     (Table 3, support, rare flagged)
  A4  results/figures/per_class_f1_frozen_tta.png       (frozen pooled TTA, not single fold)
  A3  results/tables/training_time.csv / .md            (Table 4: epochs + approx time)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from src.data.manifests import read_manifest_csv  # noqa: E402
from src.evaluation.visualization import plot_confusion_matrix, plot_per_class_f1  # noqa: E402

FROZEN_EXP = "11_triple_weighted_finetune_wide_official"
NUM_CLASSES = 23
RARE_SUPPORT = 10  # flag classes with pooled test support <= this

# Approximate per-epoch wall-clock, marked APPROX in the table. Basis:
#  - fine-tune (image, RTX 5080): ~1.5 min/epoch from seed-123 run timestamps.
#  - frozen (cached features): seconds/epoch + a one-time ~3 min feature-cache build.
FINETUNE_MIN_PER_EPOCH = 1.5
FROZEN_NOTE = "fast (cached features; ~sec/epoch + one-time ~3 min cache build)"


def fold_dir(exp: str, fold: int) -> Path:
    return _ROOT / "results" / "runs" / (exp if fold == 0 else f"{exp}_fold_{fold}")


def load_class_names() -> list[str]:
    rows = read_manifest_csv(_ROOT / "data" / "splits" / "hyperkvasir_official_5fold" / "fold_0.csv")
    mapping: dict[int, str] = {}
    for r in rows:
        mapping[int(r["label"])] = str(r["class_name"])
    return [mapping[i] for i in range(NUM_CLASSES)]


def load_pooled_tta() -> tuple[np.ndarray, np.ndarray]:
    preds, labels = [], []
    for f in range(5):
        d = np.load(fold_dir(FROZEN_EXP, f) / "predictions_tta.npz")
        preds.append(d["preds"])
        labels.append(d["labels"])
    return np.concatenate(preds), np.concatenate(labels)


def a1_confusion_matrix(preds, labels, class_names) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    out = _ROOT / "results" / "figures" / "confusion_matrix_frozen_tta.png"
    plot_confusion_matrix(
        cm.tolist(), class_names, out,
        title="Confusion Matrix — frozen model (exp 11 + TTA, pooled 5-fold)",
    )
    print(f"[A1] {out}")


def a2_per_class(preds, labels, class_names) -> dict:
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )
    rows = []
    for i in range(NUM_CLASSES):
        rows.append({
            "label": i, "class_name": class_names[i],
            "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4), "support": int(sup[i]),
            "rare": "yes" if int(sup[i]) <= RARE_SUPPORT else "",
        })
    rows.sort(key=lambda x: x["support"])  # rare classes first

    macro_f1 = float(np.mean(f1))
    macro_p, macro_r = float(np.mean(p)), float(np.mean(r))

    csv_path = _ROOT / "results" / "tables" / "per_class_frozen_tta.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "class_name", "precision", "recall", "f1", "support", "rare"])
        w.writeheader()
        w.writerows(rows)

    md = ["# Table 3 — Per-class metrics (frozen model: exp 11 triple weighted CE + TTA)",
          "", f"Pooled official 5-fold test set, n={len(labels)}. "
          f"Macro: P={macro_p:.4f} R={macro_r:.4f} **F1={macro_f1:.4f}**. "
          f"Rare = pooled support ≤ {RARE_SUPPORT}.", "",
          "| Class | Precision | Recall | F1 | Support | Rare |",
          "|---|---:|---:|---:|---:|:--:|"]
    for x in rows:
        md.append(f"| {x['class_name']} | {x['precision']:.3f} | {x['recall']:.3f} | "
                  f"{x['f1']:.3f} | {x['support']} | {'⚠️' if x['rare'] else ''} |")
    md.append(f"| **macro avg** | **{macro_p:.4f}** | **{macro_r:.4f}** | **{macro_f1:.4f}** | {len(labels)} | |")
    (_ROOT / "results" / "tables" / "per_class_frozen_tta.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[A2] results/tables/per_class_frozen_tta.csv/.md  (macro-F1={macro_f1:.4f}, "
          f"rare classes={sum(1 for x in rows if x['rare'])})")
    return {str(x["label"]): {"f1": x["f1"], "support": x["support"]} for x in rows}


def a4_per_class_chart(per_class, class_names) -> None:
    out = _ROOT / "results" / "figures" / "per_class_f1_frozen_tta.png"
    plot_per_class_f1(per_class, class_names, out,
                      title="Per-class F1 — frozen model (exp 11 + TTA, pooled 5-fold)")
    print(f"[A4] {out}")


def a3_training_time() -> None:
    # epochs-to-early-stop = epoch_log line count (precise). Wall-clock is APPROX.
    specs = [
        ("01_single_resnet50_frozen_official", "ResNet50", "—", "frozen"),
        ("03_single_efficientnetb0_frozen_official", "EfficientNetB0", "—", "frozen"),
        ("07_pair_m_e_concat_frozen_official", "M+E", "concat", "frozen"),
        ("09_triple_weighted_frozen_official", "R+M+E", "weighted", "frozen"),
        ("13_single_efficientnetb0_finetune_wide_official", "EfficientNetB0", "—", "finetune-3blk"),
        ("14_pair_m_e_finetune_wide_official", "M+E", "concat", "finetune-3blk"),
        ("10_triple_concat_finetune_wide_official", "R+M+E", "concat", "finetune-3blk"),
        ("11_triple_weighted_finetune_wide_official", "R+M+E", "weighted", "finetune-3blk"),
        ("15_triple_gmu_finetune_wide_official", "R+M+E", "GMU", "finetune-3blk"),
    ]
    rows = []
    for exp, backbones, fusion, mode in specs:
        log = fold_dir(exp, 0) / "epoch_log.jsonl"
        # Use the FINAL entry's `epoch` field, not the line count: some logs were
        # appended across re-runs (e.g. exp 10 has 44 lines but last epoch=14).
        epochs = None
        if log.exists():
            last = None
            for line in log.open(encoding="utf-8"):
                if line.strip():
                    last = line
            if last:
                epochs = int(json.loads(last)["epoch"])
        if epochs is None:
            approx = "n/a"
        elif mode == "frozen":
            approx = FROZEN_NOTE
        else:
            approx = f"~{epochs * FINETUNE_MIN_PER_EPOCH:.0f} min/fold (APPROX, ~{FINETUNE_MIN_PER_EPOCH} min/epoch, RTX 5080)"
        rows.append({"experiment": exp, "backbones": backbones, "fusion": fusion,
                     "mode": mode, "epochs_to_early_stop_fold0": epochs, "approx_walltime": approx})

    csv_path = _ROOT / "results" / "tables" / "training_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "backbones", "fusion", "mode",
                                           "epochs_to_early_stop_fold0", "approx_walltime"])
        w.writeheader()
        w.writerows(rows)

    md = ["# Table 4 — Training cost (epochs to early-stop + approximate wall-clock)",
          "", "Epochs = final epoch_log.jsonl entry, fold 0 (handles re-run appends). "
          "Wall-clock is APPROXIMATE — "
          "epoch_log has no timestamps; fine-tune ≈ 1.5 min/epoch from observed RTX 5080 "
          "run timestamps, frozen uses cached features (seconds/epoch + one-time cache).", "",
          "| Experiment | Backbones | Fusion | Mode | Epochs→stop (fold 0) | Approx wall-clock |",
          "|---|---|---|---|---:|---|"]
    for x in rows:
        md.append(f"| {x['experiment']} | {x['backbones']} | {x['fusion']} | {x['mode']} | "
                  f"{x['epochs_to_early_stop_fold0']} | {x['approx_walltime']} |")
    (_ROOT / "results" / "tables" / "training_time.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[A3] results/tables/training_time.csv/.md  ({len(rows)} configs)")


def main() -> None:
    class_names = load_class_names()
    preds, labels = load_pooled_tta()
    assert len(labels) == 10662, f"expected pooled n=10662, got {len(labels)}"
    print(f"Frozen pooled TTA: n={len(labels)}, classes={len(set(labels.tolist()))}")
    a1_confusion_matrix(preds, labels, class_names)
    per_class = a2_per_class(preds, labels, class_names)
    a4_per_class_chart(per_class, class_names)
    a3_training_time()


if __name__ == "__main__":
    main()
