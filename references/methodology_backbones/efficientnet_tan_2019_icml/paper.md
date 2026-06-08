# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

> **Manual stub (NOT auto-extracted).** Added 2026-06-08 because EfficientNetB0 is
> a foundational backbone we implement but it is not part of the auto-extracted
> survey in `references/INDEX.md`. Metadata is standard and widely verified.
> This stub exists to satisfy the AGENTS rule "no paper.md → no cite".

```yaml
metadata:
  title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
  authors: ["Tan, Mingxing", "Le, Quoc V."]
  venue: "International Conference on Machine Learning (ICML)"
  year: 2019
  arxiv: "1905.11946"
  pages: "6105-6114"
  directly_comparable: not_comparable
  paper_type: peer_reviewed
  bibtex_key: tan2019efficientnet
```

## Why we cite this paper
EfficientNetB0 is one of the three required backbones and the strongest single
frozen feature extractor in our experiments; it provides a 1280-dimensional
penultimate feature used by our fusion pipeline.

## What we use
The torchvision ImageNet-pretrained EfficientNetB0 as a frozen / fine-tuned
feature extractor. Architecture facts (7 MBConv stages, 1280-d features) are
verified in `project_structure.md §2.1`.
