"""Regenerate report-friendly VECTOR (PDF) figures from saved data.

INFERENCE-FREE: reads results/runs/11_*/predictions_tta.npz, results/tables/
extra_metrics_*.json, and epoch_log.jsonl. No model load, no training. Outputs
are sized for an A4 portrait \\linewidth with legible fonts, saved as vector PDF
into reports/final/figures/ (drop-in for \\includegraphics, no \\resizebox needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.data.manifests import read_manifest_csv

OUT = _ROOT / "reports" / "final" / "figures"
TBL = _ROOT / "results" / "tables"
FROZEN = "11_triple_weighted_finetune_wide_official"
NUM = 23

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

CV = [
    ("EfficientNetB0", "13_single_efficientnetb0_finetune_wide_official"),
    ("M+E concat",     "14_pair_m_e_finetune_wide_official"),
    ("R+M+E concat",   "10_triple_concat_finetune_wide_official"),
    ("R+M+E weighted", "11_triple_weighted_finetune_wide_official"),
    ("R+M+E GMU",      "15_triple_gmu_finetune_wide_official"),
]


def fold_dir(exp: str, f: int) -> Path:
    return _ROOT / "results" / "runs" / (exp if f == 0 else f"{exp}_fold_{f}")


def class_names() -> list[str]:
    rows = read_manifest_csv(_ROOT / "data" / "splits" / "hyperkvasir_official_5fold" / "fold_0.csv")
    m = {int(r["label"]): str(r["class_name"]) for r in rows}
    return [m[i] for i in range(NUM)]


def pooled_tta(exp: str):
    P, L = [], []
    for f in range(5):
        d = np.load(fold_dir(exp, f) / "predictions_tta.npz")
        P.append(d["preds"]); L.append(d["labels"])
    return np.concatenate(P), np.concatenate(L)


def fig_bar() -> None:
    labels, accs, f1s = [], [], []
    for lab, exp in CV:
        m = json.load(open(TBL / f"extra_metrics_{exp}.json"))
        labels.append(lab); accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    b1 = ax.bar(x - w / 2, accs, w, label="Doğruluk", color="#4C72B0")
    b2 = ax.bar(x + w / 2, f1s, w, label="Makro-F1", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.4, 0.95); ax.set_ylabel("Skor (havuzlanmış, $n{=}10{,}662$)")
    ax.legend(loc="upper left")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "comparison_bar_chart.pdf"); plt.close(fig)
    print("[ok] comparison_bar_chart.pdf")


def _last_run(path: Path) -> list[dict]:
    """Return only the FINAL run's epochs (epoch_log.jsonl may be appended across
    re-runs, e.g. exp 10 has 44 lines from multiple runs)."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    starts = [i for i, r in enumerate(rows) if r.get("epoch") == 1]
    return rows[starts[-1]:] if starts else rows


def fig_curves() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for lab, exp in CV:
        rows = _last_run(fold_dir(exp, 0) / "epoch_log.jsonl")
        e = [r["epoch"] for r in rows]
        a1.plot(e, [r["val_accuracy"] for r in rows], label=lab)
        a2.plot(e, [r["val_macro_f1"] for r in rows], label=lab)
    a1.set_title("Doğrulama doğruluğu"); a2.set_title("Doğrulama makro-F1")
    for a in (a1, a2):
        a.set_xlabel("Epoch"); a.grid(alpha=0.3)
    a1.set_ylabel("Skor")
    a2.legend(fontsize=6.5, loc="lower right")
    fig.savefig(OUT / "training_curves.pdf"); plt.close(fig)
    print("[ok] training_curves.pdf (compact, 5 CV configs, fold 0)")


def fig_cm() -> None:
    P, L = pooled_tta(FROZEN); cn = class_names()
    cm = confusion_matrix(L, P, labels=list(range(NUM))).astype(float)
    rs = cm.sum(1, keepdims=True)
    cmn = np.where(rs > 0, cm / rs, 0.0)
    # Borgli-style: index labels 0..22 on axes (clean square, centers well);
    # the index->name legend goes in the LaTeX caption.
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM)); ax.set_yticks(range(NUM))
    ax.set_xticklabels(range(NUM), fontsize=7)
    ax.set_yticklabels(range(NUM), fontsize=7)
    ax.set_xlabel("Tahmin (sınıf indeksi)"); ax.set_ylabel("Gerçek (sınıf indeksi)")
    for i in range(NUM):
        ax.text(i, i, f"{cmn[i, i]:.2f}", ha="center", va="center",
                fontsize=5, color="white" if cmn[i, i] > 0.6 else "black")
    fig.savefig(OUT / "confusion_matrix_frozen_tta.pdf"); plt.close(fig)
    print("[ok] confusion_matrix_frozen_tta.pdf (index axes)")
    print("    CAPTION LEGEND -> " + "; ".join(f"{i}={cn[i]}" for i in range(NUM)))


def fig_perclass() -> None:
    P, L = pooled_tta(FROZEN); cn = class_names()
    _, _, f1, s = precision_recall_fscore_support(
        L, P, labels=list(range(NUM)), average=None, zero_division=0)
    order = np.argsort(f1)[::-1]  # descending: best -> worst, left -> right
    names = [f"{cn[i]} (n={int(s[i])})" for i in order]
    vals = [float(f1[i]) for i in order]
    colors = ["#d62728" if v < 0.3 else "#4C72B0" for v in vals]
    x = np.arange(NUM)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(x, vals, width=0.8, color=colors)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.axhline(0.3, ls="--", color="#d62728", lw=0.8, alpha=0.7)
    ax.set_ylim(0, 1.13)
    ax.set_xlim(-0.7, NUM - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.3)
    ax.set_ylabel("F1")
    ax.set_title("Sınıf-bazlı F1  (makro-F1 = 0.6075;  kırmızı: F1 < 0.3)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(OUT / "per_class_f1_frozen_tta.pdf"); plt.close(fig)
    print("[ok] per_class_f1_frozen_tta.pdf (vertical, centered)")


def fig_architecture() -> None:
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(9, 3.7))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.1, 4.1); ax.axis("off")
    bw, bh = 1.7, 0.8

    def box(x, yc, text, fc, w=bw):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, yc - bh / 2), w, bh, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0.8, edgecolor="#333333", facecolor=fc))
        ax.text(x + w / 2, yc, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9))

    yT, yM, yB = 3.4, 2.0, 0.6
    box(0.2, yM, "Girdi\n224×224×3", "#eeeeee")
    bx = 2.5
    box(bx, yT, "ResNet50\n2048-d", "#d6e4f0")
    box(bx, yM, "MobileNetV2\n1280-d", "#d6e4f0")
    box(bx, yB, "EfficientNetB0\n1280-d", "#d6e4f0")
    px, pw = 4.7, 1.1
    for yy in (yT, yM, yB):
        box(px, yy, "512-d", "#d8efd8", w=pw)
    fx, fw = 6.3, 2.3
    box(fx, yM, "Füzyon\nconcat / weighted / GMU", "#fae0c8", w=fw)
    mx, mw = 9.0, 1.6
    box(mx, yM, "MLP\n256 + Dropout", "#e6dcf0", w=mw)
    sx, sw = 10.9, 1.5
    box(sx, yM, "Softmax\n23 sınıf", "#eeeeee", w=sw)

    for yy in (yT, yM, yB):
        arrow(0.2 + bw, yM, bx, yy)       # input -> backbones
        arrow(bx + bw, yy, px, yy)        # backbone -> projection
        arrow(px + pw, yy, fx, yM)        # projection -> fusion
    arrow(fx + fw, yM, mx, yM)            # fusion -> MLP
    arrow(mx + mw, yM, sx, yM)            # MLP -> Softmax

    # dashed group boxes (no text labels -> explained in caption, no overlap)
    for (x0, w0) in [(bx - 0.18, bw + 0.36), (px - 0.15, pw + 0.30)]:
        ax.add_patch(mpatches.Rectangle((x0, yB - bh / 2 - 0.12), w0, (yT - yB) + bh + 0.24,
                     fill=False, linestyle="--", linewidth=0.7, edgecolor="#888888"))

    fig.savefig(OUT / "architecture.pdf"); plt.close(fig)
    print("[ok] architecture.pdf (compact, no overlapping labels)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_architecture(); fig_bar(); fig_curves(); fig_cm(); fig_perclass()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
