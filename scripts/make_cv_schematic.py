"""Presentation asset: clean 5-fold cross-validation schematic.

Matches the slide deck palette (periwinkle blue), transparent background so it
drops onto any slide. Protocol: official fold k -> test, (k+1)%5 -> validation,
rest -> train; pooled test covers each sample exactly once (n=10,662).
Outputs reports/final/figures/cv_5fold_schematic.{pdf,png}. Inference-free.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "reports" / "final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- slide palette ---
TEST, VAL, TRAIN = "#5B6FE0", "#A9B2F0", "#E7E9FB"
TXT = "#2E3A59"
N = 5

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.set_xlim(0, 11.3); ax.set_ylim(0, 6.5); ax.axis("off")

cw, ch = 1.6, 0.66
px, py = 1.78, 0.86
x0, ytop = 1.95, 5.0


def cell(cx, cy, color, label, tcolor):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx, cy - ch / 2), cw, ch,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=0, edgecolor="none", facecolor=color))
    if label:
        ax.text(cx + cw / 2, cy, label, ha="center", va="center",
                fontsize=8.5, color=tcolor, fontweight="bold")


for r in range(N):
    cy = ytop - r * py
    ax.text(x0 - 0.35, cy, f"Kat {r}", ha="right", va="center",
            fontsize=10, color=TXT, fontweight="bold")
    test_col, val_col = r, (r + 1) % N
    for c in range(N):
        cx = x0 + c * px
        if c == test_col:
            cell(cx, cy, TEST, "Test", "white")
        elif c == val_col:
            cell(cx, cy, VAL, "Doğr.", TXT)
        else:
            cell(cx, cy, TRAIN, "", TXT)

# legend
lx, ly = x0, 5.95
for name, col, tc in [("Test", TEST, "white"), ("Doğrulama", VAL, TXT),
                      ("Eğitim", TRAIN, TXT)]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx, ly - 0.18), 0.42, 0.36, boxstyle="round,pad=0,rounding_size=0.07",
        linewidth=0, edgecolor="none", facecolor=col))
    ax.text(lx + 0.56, ly, name, ha="left", va="center", fontsize=9.5, color=TXT)
    lx += 2.0

# bottom caption
xend = x0 + (N - 1) * px + cw
ax.plot([x0, xend], [0.66, 0.66], color="#C9CEF0", lw=1.2)
ax.text((x0 + xend) / 2, 0.34,
        "Havuzlanmış değerlendirme — her örnek tam bir kez test edilir  (n = 10.662)",
        ha="center", va="center", fontsize=9.5, color=TXT)

fig.savefig(OUT / "cv_5fold_schematic.pdf", bbox_inches="tight", transparent=True)
fig.savefig(OUT / "cv_5fold_schematic.png", bbox_inches="tight", dpi=220, transparent=True)
plt.close(fig)
print("[ok] cv_5fold_schematic.{pdf,png}")
