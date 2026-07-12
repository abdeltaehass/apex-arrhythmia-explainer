# APEX — Target Metrics

These are the metrics we commit to tracking from day one. Every training/eval run logs
all of them to Weights & Biases so regressions are visible across experiments.

## 1. Detection quality (per the 71 SCP-ECG labels)

| Metric | Definition | v1 target |
|---|---|---|
| **AUROC (per label)** | Area under ROC, computed independently for each of the 71 labels | report full table; investigate any label < 0.75 |
| **Macro AUROC** | Unweighted mean of per-label AUROC | ≥ 0.90 |
| **Macro F1** | Unweighted mean of per-label F1 at tuned thresholds | ≥ 0.75 |
| **Micro F1** | Globally pooled F1 | ≥ 0.80 |
| **Per-label F1 table** | F1 for every label, sorted ascending | surfaces weak/rare classes |

Rationale: PTB-XL is highly imbalanced (some SCP codes appear a few dozen times).
Macro metrics keep rare arrhythmias honest; micro metrics track overall utility. AUROC
is threshold-independent and is the primary model-selection metric. F1 is reported at
per-label thresholds tuned on the validation fold.

## 2. Calibration & review-gating

| Metric | Definition | v1 target |
|---|---|---|
| **ECE** | Expected Calibration Error (15-bin) on prediction confidences | ≤ 0.05 |
| **Manual-review rate** | Fraction of records with ≥1 prediction below threshold routed to review | tune so that review set captures ≥ 90% of the model's errors |
| **Coverage @ risk** | Accuracy on the auto-accepted (above-threshold) subset | ≥ 0.95 on auto-accepted |

## 3. Explanation quality (generation + grounding)

| Metric | Definition | v1 target |
|---|---|---|
| **Consistency rate** | % of explanations whose asserted findings ⊆ detector's surfaced labels | ≥ 0.98 |
| **Hallucination rate** | % of explanations asserting a finding the detector did NOT surface | ≤ 0.02 |
| **Grounding coverage** | % of asserted findings with a supporting lead+time-window saliency region | ≥ 0.95 |

Consistency/hallucination are checked automatically by `src/eval/consistency.py` and
`src/eval/hallucination.py`, not by human rating, so they run on every eval.

## 4. Systems / latency

| Metric | Definition | v1 target |
|---|---|---|
| **p50 inference latency** | Median end-to-end (signal → labels → explanation) | ≤ 1.5 s |
| **p95 inference latency** | 95th percentile end-to-end | ≤ 4 s |
| **Throughput** | Records/sec at batch size 32 (detector only) | report |

## Evaluation protocol

- **Splits:** use PTB-XL's official `strat_fold` (1–10). Folds 1–8 = train, fold 9 =
  validation (threshold + calibration tuning), fold 10 = test (report once, at the end).
- **Never** tune on fold 10. Test-set numbers are reported a single time per milestone.
- All metrics are logged per-run to W&B; the per-label AUROC/F1 tables are logged as
  W&B Tables so they are sortable and diff-able across runs.
