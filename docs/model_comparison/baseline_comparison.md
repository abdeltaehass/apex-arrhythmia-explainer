# Phase 12 — APEX vs published PTB-XL baselines

APEX detector (`APEX (cnn_bce)`) on the **PTB-XL test split** (fold 10, 2198 records, never used for tuning). Regenerate with `python scripts/eval_baselines.py`.

## Detection: macro-AUROC vs the published benchmark

| model | all-task AUROC | superclass AUROC | source |
|---|---:|---:|---|
| inception1d | 0.925 | 0.921 | Strodthoff 2021 |
| xresnet1d101 | 0.925 | 0.928 | Strodthoff 2021 |
| **APEX (cnn_bce)** | **0.920** | 0.893 | **this work** |
| resnet1d_wang | 0.919 | 0.930 | Strodthoff 2021 |
| fcn_wang | 0.918 | 0.925 | Strodthoff 2021 |
| lstm_bidir | 0.914 | 0.921 | Strodthoff 2021 |
| lstm | 0.907 | 0.927 | Strodthoff 2021 |
| Wavelet+NN | 0.849 | 0.874 | Strodthoff 2021 |

Published numbers are the PTB-XL benchmark of **Strodthoff et al. 2021** ([helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking)) — *the* landmark PTB-XL baseline. On the **71-code "all" task APEX matches `resnet1d_wang` (0.920 vs 0.919)** and sits just under the inception1d / xresnet1d101 top of 0.925 — competitive with a compact 1D-CNN, no ensembling.

> **On "Ribeiro et al. 2020"** (the landmark deep-ECG paper): that model was trained on the *CODE* dataset (2M+ Brazilian ECGs, 6 classes) and reports F1, not AUROC on PTB-XL — a different dataset and task, so no direct PTB-XL row exists. Its residual-1D-CNN architecture is the lineage of `resnet1d_wang` above, which is the correct like-for-like PTB-XL comparison.

## Per-superclass AUROC (APEX)

| superclass | APEX test AUROC |
|---|---:|
| NORM | 0.935 |
| MI | 0.891 |
| STTC | 0.915 |
| CD | 0.897 |
| HYP | 0.826 |
| **macro** | **0.893** |

Macro F1 0.320 · micro F1 0.598 (per-label thresholds tuned on validation, applied to test) · ECE 0.897. The superclass numbers are **pooled** from the 71-code head (max over each superclass's members), not from a model trained on the 5-class superdiagnostic task — a slight handicap versus the published superdiagnostic-trained rows, so read the "all"-task column as the apples-to-apples comparison. HYP (hypertrophy, the rarest superclass) is the weakest, matching the Phase-3 per-label finding.

Explanation-quality (BLEU/ROUGE) and the GPT-4o zero-shot comparison are in `docs/model_comparison/gpt4o_comparison.md`.
