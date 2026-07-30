# Demographic performance breakdown

`APEX (cnn_bce)` on the **PTB-XL test (fold 10)** (2198 records). Regenerate with `python scripts/demographic_breakdown.py`.

**Method.** Every subgroup is scored on the *same* label set — the labels evaluable in all subgroups of a comparison (64 labels for sex, 26 for age) — because macro-AUROC silently skips labels with only one class present, and averaging over different label sets would not be a like-for-like comparison. For the age comparison that set is derived from the **adult** bands only: the 13-record `<18` band makes almost no label evaluable, so including it in the intersection would have collapsed the whole comparison to a dozen labels. Every band, pediatric included, is then scored on that same set. Each figure carries a percentile bootstrap CI (200 resamples). Subgroups under 50 records are marked; their point estimates should not be quoted without the interval.

## By sex

| group | n | macro-AUROC | 95% CI (bootstrap) | |
|---|---:|---:|---|---|
| male | 1132 | 0.9250 | 0.914 – 0.934 |  |
| female | 1066 | 0.9062 | 0.896 – 0.919 |  |

**Gap (male − female): +0.0188** (95% CI +0.0044 – +0.0325). The interval **excludes zero**: a real difference at this sample size.

Per-superclass, by sex:

| superclass | male | female |
|---|---:|---:|
| NORM | 0.941 | 0.929 |
| MI | 0.898 | 0.882 |
| STTC | 0.934 | 0.891 |
| CD | 0.904 | 0.889 |
| HYP | 0.818 | 0.838 |
| macro | 0.899 | 0.886 |

## By age band

| group | n | macro-AUROC | 95% CI (bootstrap) | |
|---|---:|---:|---|---|
| <18 | 13 | 0.7414 | 0.626 – 0.819 | ⚠ small n |
| 18-39 | 271 | 0.9057 | 0.876 – 0.925 |  |
| 40-59 | 641 | 0.9026 | 0.884 – 0.919 |  |
| 60-74 | 721 | 0.8891 | 0.879 – 0.916 |  |
| 75+ | 518 | 0.8639 | 0.846 – 0.879 |  |

The widest gap between two adequately-sized age bands is **18-39 vs 75+: +0.0417** (95% CI +0.0065 – +0.0683), which **excludes zero** — a real, if modest, age effect.

The `<18` band is reported to *measure* pediatric coverage, not because the model is intended for it. PTB-XL's age sentinel (300 = "older than 89", 293 records dataset-wide) is excluded rather than bucketed into `75+`, since treating it as a real age would invent precision the dataset deliberately removed.

## Coverage of the out-of-scope populations

The model card declares pediatric ECGs and pacemaker rhythms out of scope. Those claims are measured, not assumed:

| population | records | share of dataset | train | test |
|---|---:|---:|---:|---:|
| pediatric (age < 18) | 133 | 0.61% | 111 | 13 |
| pacemaker rhythm (`PACE`) | 294 | 1.35% | 237 | 28 |

> **Correction to a common assumption:** PTB-XL is often described as adult-only. It is not — the youngest patient in the dataset is **2 years old**, and 133 records (0.61%) are under 18. The out-of-scope conclusion is unchanged, but the *reason* is different and stronger: pediatric ECGs are present yet vanishingly rare (111 training records), far too few to train or validate on. Pediatric ECGs also differ physiologically from adult ones (faster rates, right-dominant axis, benign T-wave inversion patterns), so adult norms misread them.
