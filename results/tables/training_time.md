# Table 4 — Training cost (epochs to early-stop + approximate wall-clock)

Epochs = final epoch_log.jsonl entry, fold 0 (handles re-run appends). Wall-clock is APPROXIMATE — epoch_log has no timestamps; fine-tune ≈ 1.5 min/epoch from observed RTX 5080 run timestamps, frozen uses cached features (seconds/epoch + one-time cache).

| Experiment | Backbones | Fusion | Mode | Epochs→stop (fold 0) | Approx wall-clock |
|---|---|---|---|---:|---|
| 01_single_resnet50_frozen_official | ResNet50 | — | frozen | 16 | fast (cached features; ~sec/epoch + one-time ~3 min cache build) |
| 03_single_efficientnetb0_frozen_official | EfficientNetB0 | — | frozen | 16 | fast (cached features; ~sec/epoch + one-time ~3 min cache build) |
| 07_pair_m_e_concat_frozen_official | M+E | concat | frozen | 21 | fast (cached features; ~sec/epoch + one-time ~3 min cache build) |
| 09_triple_weighted_frozen_official | R+M+E | weighted | frozen | 12 | fast (cached features; ~sec/epoch + one-time ~3 min cache build) |
| 13_single_efficientnetb0_finetune_wide_official | EfficientNetB0 | — | finetune-3blk | 13 | ~20 min/fold (APPROX, ~1.5 min/epoch, RTX 5080) |
| 14_pair_m_e_finetune_wide_official | M+E | concat | finetune-3blk | 13 | ~20 min/fold (APPROX, ~1.5 min/epoch, RTX 5080) |
| 10_triple_concat_finetune_wide_official | R+M+E | concat | finetune-3blk | 14 | ~21 min/fold (APPROX, ~1.5 min/epoch, RTX 5080) |
| 11_triple_weighted_finetune_wide_official | R+M+E | weighted | finetune-3blk | 12 | ~18 min/fold (APPROX, ~1.5 min/epoch, RTX 5080) |
| 15_triple_gmu_finetune_wide_official | R+M+E | GMU | finetune-3blk | 14 | ~21 min/fold (APPROX, ~1.5 min/epoch, RTX 5080) |