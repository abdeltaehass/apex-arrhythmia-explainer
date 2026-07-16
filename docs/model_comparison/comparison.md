# Phase 4 — Model comparison

All APEX runs: 100 Hz, official patient-level split (train folds 1–8, val 9, test 10), 20 epochs, AdamW + cosine LR. Metrics are macro over the 71 SCP-ECG statements. F1 uses per-label thresholds tuned on the eval fold.

## APEX runs (this project)

| run               | model       | loss   |    params |   train_s |   val AUROC |   val macroF1 |   test AUROC |   test macroF1 |
|:------------------|:------------|:-------|----------:|----------:|------------:|--------------:|-------------:|---------------:|
| cnn_bce           | cnn         | bce    | 8,778,055 |     309.9 |      0.9174 |        0.3688 |       0.9199 |         0.359  |
| cnn_focal         | cnn         | focal  | 8,778,055 |     309.2 |      0.9151 |        0.3819 |       0.9179 |         0.3583 |
| transformer_focal | transformer | focal  | 1,859,399 |     123   |      0.8943 |        0.3321 |       0.8814 |         0.3189 |
| transformer_bce   | transformer | bce    | 1,859,399 |     122.6 |      0.8958 |        0.3261 |       0.8794 |         0.323  |

**Best APEX model: `cnn_bce`** — test macro-AUROC **0.9199**, 8,778,055 params, 310s train.

## Published PTB-XL results (same 71-label "all" task)

Test-fold macro-AUROC. These use the identical PTB-XL split and label set, so they are directly comparable to our `test AUROC` column.

| model         | test macro-AUROC (95% CI)   |
|:--------------|:----------------------------|
| inception1d   | 0.925 ± 0.008               |
| xresnet1d101  | 0.925 ± 0.007               |
| resnet1d_wang | 0.919 ± 0.008               |
| fcn_wang      | 0.918 ± 0.008               |
| lstm_bidir    | 0.914 ± 0.008               |
| lstm          | 0.907 ± 0.008               |
| Wavelet+NN    | 0.849 ± 0.013               |

> Our compact CNN baseline lands within ~0.01–0.02 AUROC of the published single-model benchmarks despite far fewer parameters and no pretraining — a sanity check that the pipeline is sound, not a claim of SOTA. The published leaders (inception1d/xresnet1d101, ~0.925) are deeper, tuned architectures.

## Takeaways

- **The class-weighted-BCE CNN is the strongest APEX model.** Neither the PatchTST-style 1D transformer (−0.04 test AUROC) nor focal loss (≈ baseline, marginally lower) improved on it at this scale and tuning budget.
- The transformer trains ~2.5× faster but underfits — consistent with the published benchmark, where convolutional models outperform sequence models on PTB-XL. It would likely need self-supervised pretraining or more capacity/tuning to compete.
- Per-label threshold-moving is already applied for F1. Remaining gains are more likely from probability calibration, rare-class oversampling, or a deeper/pretrained CNN than from swapping the loss.
- **Final model: `cnn_bce`** (`outputs/cnn_bce_best.pt`), test macro-AUROC 0.920.

_Sources: Strodthoff et al., IEEE JBHI 2021 (github.com/helme/ptbxl_benchmarking)._
