"""Presentation asset: a montage of raw HyperKvasir sample images.

Picks one real image per selected class (mix of common + rare) and lays them in
a 2x4 grid with class name + support, to illustrate variety and imbalance on the
"Veri" slide. Reads actual JPEGs from data/ via the official manifest. No training.
Output: reports/final/figures/data_samples.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.data.manifests import read_manifest_csv

OUT = _ROOT / "reports" / "final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# (class_name, support) — common -> rare, tells the imbalance story
SELECT = [
    ("bbps-2-3", 1148), ("polyp", 1028), ("normal-cecum", 1009),
    ("retroflex-stomach", 764), ("oesophagitis-a", 403),
    ("dyed-lifted-polyps", 1002), ("ileum", 9), ("hemorroids", 6),
]

rows = read_manifest_csv(_ROOT / "data" / "splits" / "hyperkvasir_official_5fold" / "fold_0.csv")
by_class: dict[str, str] = {}
for r in rows:
    cn = r["class_name"]
    if cn not in by_class:
        by_class[cn] = r["path"]


def square(im: Image.Image, size: int = 360) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((size, size))


fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.8))
plt.rcParams.update({"pdf.fonttype": 42})
for ax, (cn, sup) in zip(axes.ravel(), SELECT):
    p = by_class.get(cn)
    if p is None:
        ax.axis("off"); continue
    img = square(Image.open(_ROOT / p).convert("RGB"))
    ax.imshow(img)
    rare = "  ·  NADİR" if sup <= 10 else ""
    ax.set_title(f"{cn}\n(n={sup}){rare}", fontsize=11, color="#2E3A59",
                 fontweight="bold" if sup <= 10 else "normal")
    ax.axis("off")

fig.suptitle("HiperKvasir — temsili sınıflar (yaygın → nadir)",
             fontsize=13, color="#5B6FE0", fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "data_samples.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[ok] data_samples.png")
