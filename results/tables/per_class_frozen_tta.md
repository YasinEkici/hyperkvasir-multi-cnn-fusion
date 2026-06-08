# Table 3 — Per-class metrics (frozen model: exp 11 triple weighted CE + TTA)

Pooled official 5-fold test set, n=10662. Macro: P=0.6162 R=0.6158 **F1=0.6075**. Rare = pooled support ≤ 10.

| Class | Precision | Recall | F1 | Support | Rare |
|---|---:|---:|---:|---:|:--:|
| hemorroids | 0.500 | 0.167 | 0.250 | 6 | ⚠️ |
| ileum | 0.000 | 0.000 | 0.000 | 9 | ⚠️ |
| ulcerative-colitis-grade-1-2 | 0.000 | 0.000 | 0.000 | 11 |  |
| ulcerative-colitis-grade-2-3 | 0.071 | 0.107 | 0.086 | 28 |  |
| ulcerative-colitis-grade-0-1 | 0.368 | 0.200 | 0.259 | 35 |  |
| barretts | 0.116 | 0.195 | 0.145 | 41 |  |
| short-segment-barretts | 0.086 | 0.170 | 0.115 | 53 |  |
| impacted-stool | 0.668 | 0.985 | 0.796 | 131 |  |
| ulcerative-colitis-grade-3 | 0.563 | 0.669 | 0.612 | 133 |  |
| ulcerative-colitis-grade-1 | 0.519 | 0.483 | 0.500 | 201 |  |
| oesophagitis-b-d | 0.651 | 0.769 | 0.706 | 260 |  |
| retroflex-rectum | 0.916 | 0.982 | 0.948 | 391 |  |
| oesophagitis-a | 0.521 | 0.365 | 0.429 | 403 |  |
| ulcerative-colitis-grade-2 | 0.690 | 0.603 | 0.643 | 443 |  |
| bbps-0-1 | 0.980 | 0.968 | 0.974 | 646 |  |
| retroflex-stomach | 0.992 | 0.988 | 0.990 | 764 |  |
| normal-z-line | 0.836 | 0.825 | 0.831 | 932 |  |
| dyed-resection-margins | 0.912 | 0.913 | 0.913 | 989 |  |
| normal-pylorus | 0.967 | 0.997 | 0.982 | 999 |  |
| dyed-lifted-polyps | 0.914 | 0.913 | 0.914 | 1002 |  |
| normal-cecum | 0.932 | 0.988 | 0.959 | 1009 |  |
| polyp | 0.977 | 0.936 | 0.956 | 1028 |  |
| bbps-2-3 | 0.992 | 0.943 | 0.967 | 1148 |  |
| **macro avg** | **0.6162** | **0.6158** | **0.6075** | 10662 | |