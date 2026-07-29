# Phase 13 — Adversarial & Edge-Case Report

Deployed pipeline `APEX (cnn_bce)` on the PTB-XL **test split** (2198 records, fold 10). Everything is measured at the **shipped surfacing rule** — a finding is surfaced when its probability ≥ 0.5 — so these are the system's real outputs, not a post-hoc tuned optimum. Regenerate with `python scripts/edge_case_report.py`.

Two failure axes, because they cost differently:

- a **miss** is a present label the system did *not* surface — a silent false negative. A missed **urgent** code (ST-elevation / injury) is a *dangerous miss*.
- an **over-flag** is a surfaced label that is not present — a false positive, and the driver of alarm fatigue.

## Cohorts

| cohort | n | label recall | over-flag / rec | dangerous misses | routed to review | clean & silent |
|---|---:|---:|---:|---:|---:|---:|
| overall (all test) | 2198 | 83.7% | 5.09 | 16 (0.7%) | 79.1% | 13.0% |
| normal (NORM-only) | 912 | 87.5% | 2.00 | 2 (0.2%) | 62.1% | 29.2% |
| clean (no annotated noise) | 1670 | 84.1% | 5.05 | 13 (0.8%) | 79.2% | 13.0% |
| noisy — significant artifact | 160 | 81.7% | 5.59 | 1 (0.6%) | 83.1% | 10.0% |
| noisy — whole-record | 93 | 79.4% | 6.27 | 0 (0.0%) | 80.7% | 10.8% |
| borderline (present @ ±0.1) | 441 | 75.6% | 6.78 | 6 (1.4%) | 95.9% | 0.4% |
| rare labels (12 rarest) | 24 | 72.9% | 11.29 | 6 (25.0%) | 100.0% | 0.0% |
| multi-condition (≥5 codes) | 224 | 83.0% | 9.89 | 8 (3.6%) | 98.7% | 0.0% |

"Routed to review" is a proxy: a record trips review if any surfaced finding is below the low-confidence bar (0.7) or two surfaced labels are mutually exclusive (the text-dependent consistency and grounding checks aren't included here). "Label recall" is per-label: of all present labels in the cohort, the share the system surfaced.

## What the numbers say

**The normal ECG — does it stay quiet?** Of 912 diagnostically-normal (NORM-only) records, the system surfaces NORM on **87.2%** and over-flags a diagnostic pathology on **48.7%**. So it *mostly* says the right thing on a clean normal — but a non-trivial minority pick up a spurious diagnostic label, each of which would (correctly, given the flag) route an otherwise-normal patient to review. That is the alarm-fatigue tax.

**Why so many over-flags? The operating point, not the ranking.** The deployed rule surfaces every label at probability ≥ 0.5, but the detector was trained with heavy class weighting (Phase 3) that deliberately inflates probabilities — its pooled calibration error is large (ECE ≈ 0.90), so 0.5 is far too low a bar. That is why the system averages **5.1 surfaced-but-absent labels per record** and tags a spurious diagnostic code on nearly half of normal ECGs. The Phase-12 per-label F1-tuned thresholds already cut this sharply (micro-F1 0.60 at tuned thresholds vs the flood at 0.5) — and the *same* model still scores 0.92 AUROC, which is threshold-free. So most of the over-flagging here is a **calibration** problem, not a discrimination one; proper probability calibration is the real fix and the subject of the next phase.

**Noise degrades it, as expected.** Records with significant artifact drop to **81.7%** label recall against **84.1%** on clean records, and over-flag more per record (5.59 vs 5.05). Whole-record ("alles") noise is the worst bucket at **79.4%** recall.

**Rare labels are the biggest blind spot.** Across the 12 rarest labels (training support 8–24 examples), cohort label recall is **72.9%** — the system most often says *nothing* rather than flag an uncertain rare finding.

**Multi-condition records lose the secondary findings.** On records carrying ≥5 codes at once, label recall is **83.0%**: the dominant abnormality surfaces, the co-morbid ones often don't.

**Silent near-misses are common.** **31.6%** of all missed labels across the test set had a probability in 0.35–0.5 — findings the model nearly surfaced and then dropped with no trace. To a user, a 0.49 miss is indistinguishable from a confident negative.

## Failure-mode taxonomy

**F1 — Co-morbid under-call.** On multi-condition records the detector surfaces the dominant abnormality and misses secondary findings (cohort recall 83.0%). Subtle STTC/HYP alongside a loud rhythm or infarct are the usual casualties. _Example: ECG 274._

**F2 — Silent miss on rare labels.** Labels with little training support have low recall (72.9% on the 12 rarest); the system emits a confident-looking negative instead of abstaining. _Example: ECG 968._

**F3 — Borderline suppression / near-miss.** 31.6% of misses sat in 0.35–0.5. A finding just under threshold is dropped silently — no surfacing and no flag — so a near-miss is invisible to the reader. _Example: ECG 63._

**F4 — Artifact-driven error.** Significant artifact cuts recall to 81.7% (whole-record 79.4%) and raises over-flagging. The system does not itself refuse a corrupted trace. _Example: ECG 75._

**F5 — Over-flagging / alarm fatigue.** Even on clean normals, 48.7% pick up a spurious diagnostic label, and the low-confidence gate routes a large share of all records to review (Phase 7 measured 75.9% on validation). High sensitivity is bought with reviewer load. _Example: ECG 9._


**F6 — Dangerous miss (missed urgent).** The highest-severity mode: an urgent ST-elevation / injury code (ANEUR, INJAL, INJAS, INJIL, INJIN, INJLA, STE_) present but not surfaced, so no red banner fires. Per-cohort counts are in the "dangerous misses" column above; this is the number to watch in production.

## Concrete examples

Each was run through the *full* deployed pipeline (`analyze_signal`, grounding on).

**Normal, handled correctly — ECG 57.**

Present (ground truth): `NORM, SR` · superclasses: NORM.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| SR | 0.99 | ok | — |
| NORM | 0.96 | ok | — |

Severity banner: **green** · review recommended: **False**.

**Multi-condition record — ECG 274.**

Present (ground truth): `ASMI, INVT, ISC_, LAO/LAE, LVH, SR, STD_, VCLVH` · superclasses: HYP, MI, STTC.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| INJAL | 0.98 | review | grounding_conflict |
| ISCAL | 0.96 | review | grounding_conflict |
| INVT | 0.95 | ok | — |
| ASMI | 0.94 | review | grounding_conflict |
| INJAS | 0.94 | ok | — |
| LVH | 0.84 | ok | — |
| SR | 0.84 | ok | — |
| ISC_ | 0.82 | ok | — |
| STD_ | 0.80 | ok | — |
| LAFB | 0.80 | ok | — |
| ABQRS | 0.69 | review | low_confidence |
| VCLVH | 0.67 | review | low_confidence |
| ISCAN | 0.58 | review | low_confidence |
| ALMI | 0.55 | review | low_confidence |

Severity banner: **red** (urgent: INJAL, INJAS) · review recommended: **True**.

Missed: `LAO/LAE`.

Over-flagged (not in ground truth): `ABQRS, ALMI, INJAL, INJAS, ISCAL, ISCAN, LAFB`.

**Whole-record noise — ECG 75.**

Present (ground truth): `NORM, SR` · superclasses: NORM.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| NORM | 0.99 | ok | — |
| SBRAD | 0.88 | ok | — |

Severity banner: **green** · review recommended: **False**.

Missed: `SR`.

Over-flagged (not in ground truth): `SBRAD`.

**Borderline / near-miss — ECG 63.**

Present (ground truth): `ABQRS, ASMI, SR` · superclasses: MI.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| ABQRS | 0.89 | ok | — |
| AMI | 0.86 | ok | — |
| ASMI | 0.86 | ok | — |
| IMI | 0.71 | review | grounding_conflict |
| QWAVE | 0.59 | review | low_confidence |
| SARRH | 0.55 | review | low_confidence |

Severity banner: **yellow** · review recommended: **True**.

Missed: `SR`.

Over-flagged (not in ground truth): `AMI, IMI, QWAVE, SARRH`.

**Rare label — ECG 968.**

Present (ground truth): `ASMI, INJIL, INVT, SR, STD_` · superclasses: MI.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| IMI | 0.94 | review | grounding_conflict |
| INVT | 0.89 | ok | — |
| ISCLA | 0.88 | review | grounding_conflict |
| ISC_ | 0.88 | ok | — |
| ABQRS | 0.87 | ok | — |
| AMI | 0.84 | review | grounding_conflict |
| ISCAL | 0.82 | review | grounding_conflict |
| QWAVE | 0.81 | ok | — |
| ISCIL | 0.79 | review | grounding_conflict |
| LVH | 0.79 | ok | — |
| IVCD | 0.74 | ok | — |
| ISCIN | 0.73 | review | grounding_conflict |
| SR | 0.73 | ok | — |
| STD_ | 0.71 | ok | — |
| ASMI | 0.66 | review | low_confidence |
| SARRH | 0.66 | review | low_confidence |
| LOWT | 0.59 | review | low_confidence |

Severity banner: **yellow** · review recommended: **True**.

Missed: `INJIL`. ⚠ **dangerous (urgent):** INJIL

Over-flagged (not in ground truth): `ABQRS, AMI, IMI, ISCAL, ISCIL, ISCIN, ISCLA, ISC_, IVCD, LOWT, LVH, QWAVE, SARRH`.

**Normal, over-flagged — ECG 9.**

Present (ground truth): `NORM, SR` · superclasses: NORM.

| surfaced | conf | gate | flags |
|---|---:|---|---|
| NORM | 0.94 | ok | — |
| SARRH | 0.82 | ok | — |
| SR | 0.61 | review | low_confidence |

Severity banner: **yellow** · review recommended: **True**.

Over-flagged (not in ground truth): `SARRH`.

## Recommended deployment guardrails

1. **Human over-read stays mandatory.** APEX is decision *support*; a green banner is never a clearance. The disclaimer and review gate apply at every severity level.
2. **Surface a sub-threshold tier for high-consequence codes.** For MI / ST-change / injury labels, show a "possible — below confidence" note when the probability lands in ~0.35–0.5 instead of dropping it silently (mitigates F3, F6).
3. **Refuse or hard-flag corrupted input.** Route records with whole-record or multi-type artifact (auto-detected SNR, or annotations where available) to mandatory review; never emit a green banner on a trace the system can't trust (F4).
4. **Abstain, don't assert, on rare labels.** Below a training-support floor, express low confidence / defer rather than implying a confident negative (F2).
5. **Report the full surfaced set, not just the top finding, and state that a missing secondary finding is not its exclusion** (F1).
6. **Tune the confidence gate to the setting.** A screening deployment should bias toward sensitivity (accept more review load); a confirmatory one toward precision. This is the calibration lever (F5) — see the calibration follow-up.
7. **Monitor red-banner (urgent) recall as the primary safety metric in production**, and audit every dangerous miss (F6).

## Honest limitations

- Ground truth is PTB-XL's human cardiologist labels, which themselves carry annotation noise; a "miss" or "over-flag" against them is not always a true error.
- The artifact cohort uses PTB-XL's own signal-quality annotations, which are incomplete — some noisy records are unlabelled, so the clean cohort is a mild over-estimate of clean performance.
- Text-dependent checks (consistency, grounding-conflict) are exercised only on the concrete examples, not across whole cohorts, because they require running generation on every record. The cohort "routed to review" proxy therefore under-counts.
- All numbers are for the single shipped checkpoint at threshold 0.5; a different operating point trades misses against over-flags.
