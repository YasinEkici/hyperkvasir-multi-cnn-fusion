# MobileNetV2: Inverted Residuals and Linear Bottlenecks

> **Manual stub (NOT auto-extracted).** Added 2026-06-08 because MobileNetV2 is a
> foundational backbone we implement but it is not part of the auto-extracted
> survey in `references/INDEX.md`. Metadata is standard and widely verified.
> This stub exists to satisfy the AGENTS rule "no paper.md → no cite".

```yaml
metadata:
  title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
  authors: ["Sandler, Mark", "Howard, Andrew", "Zhu, Menglong", "Zhmoginov, Andrey", "Chen, Liang-Chieh"]
  venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"
  year: 2018
  doi: "10.1109/CVPR.2018.00474"
  arxiv: "1801.04381"
  pages: "4510-4520"
  directly_comparable: not_comparable
  paper_type: peer_reviewed
  bibtex_key: sandler2018mobilenetv2
```

## Why we cite this paper
MobileNetV2 is one of the three required backbones; its inverted-residual design
provides a 1280-dimensional penultimate feature used by our fusion pipeline.

## What we use
The torchvision ImageNet-pretrained MobileNetV2 as a frozen / fine-tuned feature
extractor. Architecture facts (17 inverted-residual blocks, 1280-d features) are
verified in `project_structure.md §2.1`.
