---
language:
  - en
license: cc-by-4.0
tags:
  - ecg
  - electrocardiogram
  - healthcare
  - medical
  - time-series
  - multi-label-classification
  - clinical-decision-support
  - explainable-ai
datasets:
  - ptb-xl
metrics:
  - roc_auc
  - f1
model-index:
  - name: APEX (cnn_bce)
    results:
      - task:
          type: multi-label-classification
          name: ECG diagnostic statement classification (PTB-XL "all", 71 codes)
        dataset:
          type: ptb-xl
          name: PTB-XL v1.0.3
          split: test (fold 10)
        metrics:
          - type: roc_auc
            name: macro-AUROC (all-task, 71 codes)
            value: 0.920
          - type: roc_auc
            name: macro-AUROC (5 diagnostic superclasses, pooled)
            value: 0.893
          - type: f1
            name: micro-F1 (thresholds tuned on validation)
            value: 0.598
          - type: f1
            name: macro-F1 (thresholds tuned on validation)
            value: 0.320
---

# Model Card — APEX (Arrhythmia Pattern Explainer)

APEX reads a 12-lead ECG, detects abnormalities across the 71 SCP-ECG diagnostic
statements in PTB-XL, grounds each finding back onto the signal region that drove it,
writes a structured plain-English report, and flags low-confidence or self-contradictory
output for clinician review.

> **⚠️ Not a medical device.** APEX is a research and educational project. It is not
> FDA-cleared, not CE-marked, and has not been clinically validated. It must not be used
> to make or withhold a clinical decision about a real patient.

| | |
|---|---|
| **Model type** | 1D residual CNN (`ECGResNet1d`), multi-label sigmoid head |
| **Input** | 12-lead ECG, 10 s, 100 Hz (upstream resampling supported); paper-ECG images via classical-CV digitization |
| **Output** | 71 per-code probabilities → structured report (findings, impression, per-finding leads, reliability flags) |
| **Training data** | PTB-XL v1.0.3, folds 1–8 (17,418 records) |
| **Language** | English (reports); source annotations German/English/Swedish |
| **License** | `cc-by-4.0`, following PTB-XL's own terms (attribution). ⚠ The repository currently ships **no `LICENSE` file** — add one to make this binding. |
| **Repository** | [`abdeltaehass/apex-arrhythmia-explainer`](https://github.com/abdeltaehass/apex-arrhythmia-explainer) |

---

## Intended use

**Primary intended use: clinical decision support, with a qualified human in the loop for
every output.** APEX is designed to sit *beside* a clinician reading an ECG — surfacing
candidate findings, showing which part of the signal drove each one, and explicitly
flagging what it is unsure about.

Appropriate uses:

- **Second-reader / triage assistance** for a clinician who reads the ECG themselves. The
  severity banner (green / yellow / red) is a *prompt to look*, never a verdict.
- **Draft-report generation** that a clinician edits and signs. Every report ships with a
  disclaimer and a review-recommended gate.
- **Research and teaching** — reproducing PTB-XL benchmarks, studying ECG explainability,
  demonstrating grounding and reliability-checking methods.
- **Method development** — a baseline for calibration, fairness, or explanation work.

**Intended users:** clinicians, clinical researchers, and ML researchers. Not intended for
patients or the general public to self-interpret an ECG.

### Out-of-scope and prohibited use

**Autonomous diagnosis is out of scope, categorically.** APEX must not gate, delay, or
replace a clinical decision without a qualified human reading the ECG. A green banner is
not a clearance; the system's own measured false-negative rate makes that unsafe (see
[Known limitations](#known-limitations)).

Specifically out of scope:

| Out of scope | Why (measured, where measurable) |
|---|---|
| **Pediatric ECGs** | Only **133 of 21,799** PTB-XL records (**0.61%**) are under 18 — **111 in the training folds**. Pediatric ECGs also differ physiologically (faster rates, right-dominant axis, benign T-wave inversion), so adult norms misread them. Measured pediatric test AUROC is **0.741 (n=13)** vs 0.86–0.91 for adult bands. |
| **Pacemaker rhythms** | The `PACE` code appears in **294 records (1.35%)**, **237 in training**. Paced morphology invalidates most conduction and infarct criteria the model learned; APEX has not been validated on it. |
| **Use without clinical oversight** | The whole design (review gates, reliability flags, disclaimer, severity banner) assumes a clinician reads the output. Removing that removes the only safeguard against the failure modes below. |
| **Non-12-lead recordings** | The input convolution is fixed at 12 leads; single-lead / wearable / Holter data is hard-rejected (HTTP 422), not silently coerced. |
| **Emergency / time-critical autonomous triage** | Documented dangerous misses exist (an urgent injury code present but not surfaced, showing only a yellow banner — see [`docs/edge_cases/report.md`](docs/edge_cases/report.md)). |
| **Populations unlike PTB-XL's** | PTB-XL was collected 1989–1996, largely in Germany, on specific devices. Distribution shift to other populations, eras, or hardware is unvalidated. |
| **Any clinical product use** | Not out of scope by *licence* — PTB-XL is CC BY 4.0, which permits commercial use — but out of scope because APEX is **not a cleared or validated medical device**. Shipping it in a product that informs patient care would be a regulatory matter (FDA SaMD / EU MDR), not a licensing one. |

> **A correction worth stating plainly.** PTB-XL is frequently described as adult-only —
> including in this project's own earlier planning. It is not: the youngest patient is
> **2 years old**. The out-of-scope conclusion for pediatrics is unchanged, but the real
> reason is stronger than "absent from the data": pediatric ECGs are *present yet
> vanishingly rare*, which is more dangerous, because the model will still emit a
> confident-looking output on one.

---

## Known limitations

Measured on the PTB-XL test split (fold 10, 2,198 records) at the **shipped** surfacing
rule (probability ≥ 0.5) — i.e. what the system actually does, not a tuned optimum. Full
analysis in [`docs/edge_cases/report.md`](docs/edge_cases/report.md).

### 1. Label imbalance cripples rare-arrhythmia detection

Across the **12 rarest labels** (8–24 training examples each), label recall is **72.9%**
versus 83.7% overall, and **25% of that cohort suffered a dangerous miss** (an urgent
ST-elevation / injury code present but not surfaced). The system does not abstain on rare
findings — it emits a confident-looking negative. Rare-label output should be treated as
unvalidated.

### 2. Heavy over-flagging — a calibration problem, not a ranking one

At threshold 0.5 the model surfaces **5.09 absent labels per record** and tags a spurious
diagnostic code on **48.7%** of diagnostically normal ECGs. This is the direct consequence
of the class-weighted training loss, which deliberately inflates probabilities: mean
predicted probability is 0.118 against a true base rate of 0.039, roughly 3x too high
(pooled **ECE 0.079**). The *same* model scores 0.920 AUROC, which is threshold-free — so
the discrimination is sound and the operating point is wrong.

**Fixed in Phase 17.** Per-label vector scaling fitted on the validation fold cuts **ECE
0.079 → 0.002** and spurious surfaced labels **5.09 → 0.35 per record**, with AUROC
unchanged — see [`docs/calibration/report.md`](docs/calibration/report.md). The calibrator
is shipped in `outputs/calibration.json` but is **not yet applied by default in the
serving path**, so the probabilities returned by `analyze_signal` today are still the
uncalibrated ones. Do not read them as calibrated risks until that is wired in.

> An earlier version of this card quoted "ECE ≈ 0.90". That figure came from a
> mis-specified metric (a multi-class formulation applied to independent sigmoids) and was
> about 11x too high; the correct uncalibrated value is 0.079. The correction is documented
> in `docs/calibration/report.md`.

### 3. Silent near-misses

**31.6%** of all missed labels had a probability in 0.35–0.5 — findings the model nearly
surfaced, then dropped with no trace. To a reader, a 0.49 miss is indistinguishable from a
confident negative.

### 4. Degradation on noisy and digitized input

Records with significant artifact drop to **81.7%** recall (whole-record noise: **79.4%**)
and over-flag more. For **scanned or photographed paper ECGs**, the classical-CV
digitization pipeline round-trips at Pearson **0.885 (clean render) → 0.815 (mild photo) →
0.706 (heavy photo)**; everything downstream inherits that loss, and the system does not
refuse a low-quality scan. Image input should be treated as materially less reliable than
native signal input.

### 5. Co-morbid under-call

On records carrying ≥5 conditions at once, recall is 83.0% and the over-flag rate is
highest. The dominant abnormality surfaces; secondary findings often do not. **A finding's
absence from an APEX report is not evidence of its absence in the patient.**

### 6. Explanation-layer caveats

The shipped explanation backend is **template-based** (deterministic, derived from the
detected codes) — it cannot hallucinate a finding the detector didn't produce, but it also
adds no clinical nuance. The LoRA fine-tuned generative backend is implemented but was
only smoke-tested on a small model; it has not been validated at scale. Grounding is
guided Grad-CAM, which is an *attribution*, not a causal explanation.

---

## Privacy

**No patient data is retained.** Inference is **stateless**: a recording is validated,
preprocessed, classified, explained, and returned. Nothing about the request is written to
disk or to a database; the only persistence in the service is an **in-memory, bounded
latency histogram** (`/metrics`: request count and p50/p95/p99) which contains **no signal
data, no identifiers, and no report content**. Model weights are loaded once and cached;
uploads are not.

PTB-XL itself is a **de-identified, open-access** dataset (PhysioNet, CC-BY-4.0
attribution terms, ages >89 masked to the sentinel value 300). No re-identification is
attempted, and APEX does not link records to any external source.

### HIPAA and equivalent regimes — considerations for real deployment

None of the following is legal advice, and **none of it is satisfied by this repository as
shipped**. If real patient data were ever processed, at minimum:

- **A 12-lead ECG with any identifier attached is PHI.** The waveform plus a date, MRN, or
  device ID falls under the HIPAA Privacy Rule; treat the whole request as PHI.
- **Business Associate Agreement (BAA).** Any hosted deployment, and any third-party
  inference/LLM API in the path, requires a BAA with the covered entity. The optional
  hosted-LLM explanation backend would send clinical text off-premise — **do not enable it
  with PHI** without one.
- **Encryption in transit and at rest** (TLS 1.2+; the dev server ships plain HTTP), and
  network isolation of the inference service.
- **Access control and audit logging.** The shipped API-key auth and fixed-window rate
  limiter are a development convenience, not an access-control regime; HIPAA requires
  unique user identification and a tamper-evident audit trail of PHI access — which
  directly tensions with the no-retention design and must be designed deliberately.
- **Minimum necessary.** Send the waveform alone; strip identifiers before the request.
- **Breach notification, retention, and disposal policies**, plus a documented risk
  analysis (HIPAA Security Rule §164.308).
- **Outside the US**, GDPR Art. 9 treats health data as a special category (lawful basis,
  DPIA, data-residency); the EU AI Act classes clinical decision support as high-risk, and
  a device claim would trigger MDR/FDA SaMD obligations.

---

## Demographic performance breakdown

**The question — does AUROC differ by patient age or sex? — was measured, and the answer
is yes in both dimensions.** Full method and tables:
[`docs/model_card/demographics.md`](docs/model_card/demographics.md).

Method notes that make these numbers trustworthy: every subgroup is scored on the **same
label set** (macro-AUROC silently skips labels with only one class present, so unequal
label coverage would make a naive comparison apples-to-oranges), and every figure carries
a **percentile bootstrap CI** (200 resamples) because subgroup sizes differ by an order of
magnitude.

### By sex

| group | n | macro-AUROC | 95% CI |
|---|---:|---:|---|
| male | 1,132 | **0.9250** | 0.914 – 0.934 |
| female | 1,066 | **0.9062** | 0.896 – 0.919 |

**Gap (male − female): +0.0188, 95% CI +0.0044 – +0.0325.** The interval excludes zero, so
this is **a real disparity, not sampling noise** — the model performs measurably better on
male patients. Per-superclass, the gap is widest on **STTC (ST/T change): 0.934 male vs
0.891 female**. It reverses on HYP (0.818 male vs 0.838 female).

This matters clinically: female patients already face documented under-recognition of
ischemic presentations, and a model that is weaker on exactly the ST/T-change category
risks compounding that. **This disparity is not corrected in the current model.** Anyone
deploying APEX should treat it as an open safety issue, monitor sex-stratified performance
in production, and not apply a single confidence threshold uniformly across groups without
re-validating.

### By age band

| band | n | macro-AUROC | 95% CI | |
|---|---:|---:|---|---|
| <18 | 13 | 0.7414 | 0.626 – 0.819 | ⚠ tiny n — out of scope |
| 18–39 | 271 | 0.9057 | 0.876 – 0.925 | |
| 40–59 | 641 | 0.9026 | 0.884 – 0.919 | |
| 60–74 | 721 | 0.8891 | 0.879 – 0.916 | |
| 75+ | 518 | **0.8639** | 0.846 – 0.879 | |

Performance **declines monotonically with age**. The widest adult gap — **18–39 vs 75+:
+0.0417, 95% CI +0.0065 – +0.0683** — also excludes zero, so the age effect is real. Older
patients carry more co-morbid findings at once, which is precisely the multi-condition
regime where recall drops (limitation 5).

The `<18` figure (0.741) is reported to *measure* pediatric coverage, not to license it;
with 13 records its CI is very wide, but it is the worst band by a wide margin and
supports the out-of-scope designation.

PTB-XL's age sentinel (300 = "older than 89", 293 records) is **excluded** rather than
bucketed into `75+`, since treating it as a real age would invent precision the dataset
deliberately removed.

---

## Ethics statement

**Automation bias is the central risk.** A system that is right most of the time trains
its users to stop checking — and APEX's failures are concentrated exactly where checking
matters: rare arrhythmias, co-morbid records, and older patients. Every design choice that
looks like friction (the review gate, the reliability flags, the disclaimer on every
report, the refusal to fake an urgent call) exists to resist that.

**Sensitivity is bought with reviewer load, and that trade-off is a clinical decision, not
an engineering one.** At the shipped threshold APEX over-flags heavily; the fix (proper
calibration) will necessarily trade some sensitivity for precision. Where to sit on that
curve depends on the deployment — screening versus confirmation — and belongs to the
clinicians who will carry the consequences, not to a default in a config file.

**Fairness findings are published even though they are unflattering.** The measured sex
disparity is real, is not corrected, and is documented above rather than omitted or
buried. The same applies to the transformer that failed to beat the CNN, the dangerous
miss in the edge-case report, and the correction to the "PTB-XL is adult-only" assumption.
A model card that only reports favourable numbers is worse than no model card, because it
transfers unearned confidence to the reader.

**Known ethical gaps in this release:**

- **No race/ethnicity breakdown is possible** — PTB-XL does not record it. Fairness here is
  therefore verified only along age and sex; disparities on unrecorded axes cannot be ruled
  out and should not be assumed absent.
- **The probabilities returned by the serving path are not yet calibrated** (ECE 0.079),
  so a displayed confidence is not a risk estimate. A fitted calibrator exists (Phase 17,
  ECE → 0.002) but is not applied by default; until it is, presenting a confidence to a
  clinician as if it were a risk would be misleading.
- **Ground truth is human annotation**, which carries its own error and its own historical
  biases; the model inherits them.
- **No prospective or external validation** has been performed. All numbers are
  retrospective on one dataset's held-out fold.

---

## Training and evaluation

| | |
|---|---|
| **Architecture** | Residual 1D-CNN, 71-way sigmoid head |
| **Loss** | Class-weighted BCE (per-label `pos_weight`, capped at 50) |
| **Optimizer** | AdamW + cosine schedule, 20 epochs, best checkpoint by val macro-AUROC |
| **Preprocessing** | Resample 500→100 Hz → band-pass 0.5–40 Hz → Pan-Tompkins R-peaks → per-lead z-score |
| **Splits** | PTB-XL official patient-level `strat_fold`: 1–8 train / 9 val / 10 test (no patient crosses splits; asserted in code) |
| **Thresholds** | Tuned on **validation**, applied to test — never tuned on test |

**Test-split results** (fold 10, never used for tuning):

| metric | value |
|---|---:|
| macro-AUROC (71-code "all" task) | **0.920** |
| macro-AUROC (5 superclasses, pooled) | 0.893 |
| micro-F1 / macro-F1 (val-tuned thresholds) | 0.598 / 0.320 |
| ECE (pooled, uncalibrated) | 0.079 |
| ECE after Phase-17 calibration | **0.002** |

For context, the published PTB-XL benchmark (Strodthoff et al., 2021) reports 0.925 for
its best models (`inception1d`, `xresnet1d101`) and 0.919 for `resnet1d_wang` on the same
task — APEX is competitive with a compact CNN and no ensembling. A GPT-4o zero-shot
ECG-image baseline reaches ~41% multiclass accuracy (published, JMIR AI 2025). Details:
[`docs/model_comparison/baseline_comparison.md`](docs/model_comparison/baseline_comparison.md).

**Compute and environmental cost:** training is deliberately small — ~16 s/epoch × 20
epochs on a consumer Apple-silicon laptop (MPS), no GPU cluster, no pretraining. Warm
inference is ~6 ms/record on CPU. The environmental footprint is negligible relative to
foundation-model training, which was a design goal rather than an accident.

---

## How to use

```python
from src.serving.serializer import analyze_signal

# signal: (12, T) float array — 12-lead, 10 s
report = analyze_signal(signal, sampling_rate=100, with_grounding=True)

report.findings            # per-code label, description, confidence, leads, flags
report.impression          # plain-English impression
report.review_recommended  # True -> a clinician must read this before any action
report.disclaimer          # always present; do not strip it
```

Or via the service: `POST /analyze` (JSON body or `.npy`/`.csv`/`.json`/image upload).
Interactive dashboard: `make ui`.

**Always check `review_recommended` and surface the disclaimer.** A caller that ignores
both is using the model outside its intended design.

---

## Citation

APEX builds directly on PTB-XL; please cite the dataset:

```bibtex
@article{wagner2020ptbxl,
  title   = {{PTB-XL}, a large publicly available electrocardiography dataset},
  author  = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and
             Kreiseler, Dieter and Lunze, Fatima I. and Samek, Wojciech and Schaeffter, Tobias},
  journal = {Scientific Data},
  volume  = {7},
  number  = {1},
  pages   = {154},
  year    = {2020},
  doi     = {10.1038/s41597-020-0495-6}
}

@article{strodthoff2021benchmarking,
  title   = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB-XL}},
  author  = {Strodthoff, Nils and Wagner, Patrick and Schaeffter, Tobias and Samek, Wojciech},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  volume  = {25},
  number  = {5},
  pages   = {1519--1528},
  year    = {2021},
  doi     = {10.1109/JBHI.2020.3022989}
}
```

---

*Model card last regenerated against the shipped checkpoint on the PTB-XL test split.
Quantitative claims are reproducible via `make eval-baselines`, `make edge-cases`, and
`make demographics`.*
