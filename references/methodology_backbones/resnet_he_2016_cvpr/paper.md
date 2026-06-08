# Deep Residual Learning for Image Recognition (ResNet)

> **Manual stub (NOT auto-extracted).** Added 2026-06-08 because ResNet50 is a
> foundational backbone we implement but it is not part of the auto-extracted
> survey in `references/INDEX.md`. Metadata is standard and widely verified.
> This stub exists to satisfy the AGENTS rule "no paper.md → no cite".

```yaml
metadata:
  title: "Deep Residual Learning for Image Recognition"
  authors: ["He, Kaiming", "Zhang, Xiangyu", "Ren, Shaoqing", "Sun, Jian"]
  venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"
  year: 2016
  doi: "10.1109/CVPR.2016.90"
  arxiv: "1512.03385"
  pages: "770-778"
  directly_comparable: not_comparable
  paper_type: peer_reviewed
  bibtex_key: he2016resnet
```

## Why we cite this paper
ResNet50 is one of the three required backbones; its residual-block architecture
provides the 2048-dimensional penultimate feature used by our fusion pipeline.

## What we use
The torchvision ImageNet-pretrained ResNet50 as a frozen / fine-tuned feature
extractor. Architecture facts (layer indexing, 2048-d features) are verified in
`project_structure.md §2.1`.
