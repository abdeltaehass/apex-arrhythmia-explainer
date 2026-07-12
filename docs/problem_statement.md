# APEX — Problem Statement

## One-paragraph statement

APEX (Arrhythmia Pattern Explainer) is a clinical decision-support tool that reads
12-lead ECG signals, detects cardiac abnormalities across 71 diagnostic categories,
generates a structured plain-English clinical explanation, and flags low-confidence
predictions for manual review — designed to assist clinicians, not replace them.

## Scope & intent

- **Assistive, not autonomous.** APEX produces a ranked list of likely diagnoses with
  calibrated confidence scores and a written rationale grounded in the signal. A
  clinician remains in the loop for every decision. Any prediction below the
  configured confidence threshold is explicitly surfaced for manual review rather than
  presented as a conclusion.
- **Explainable by construction.** Every generated explanation must be traceable to
  specific leads, time windows, and morphological features via the grounding layer
  (attention / saliency). Explanations that cannot be grounded are flagged as potential
  hallucinations and withheld.
- **Multi-label.** A single ECG can carry several concurrent findings (e.g. atrial
  fibrillation + left bundle branch block). Detection is framed as multi-label
  classification over the 71 SCP-ECG statement categories in PTB-XL.

## Non-goals (v1)

- Not a diagnostic device; no regulatory (FDA/CE) clearance is claimed or sought in v1.
- No real-time / on-device inference; batch and interactive API only.
- No treatment recommendations — detection + explanation + confidence only.
- No single-lead or wearable-strip support in v1 (12-lead resting ECG only).

## Users

Primary: cardiologists, ED physicians, and cardiology fellows reviewing 12-lead ECGs.
Secondary: ML/clinical-informatics researchers evaluating explainable ECG models.

## Data

**PTB-XL** (PhysioNet, open access, no credentialing required):
~21,800 clinical 12-lead ECG records, 10 s each, at 100 Hz and 500 Hz, with full
demographic metadata and SCP-ECG diagnostic labels. Chosen over MIMIC specifically
because it is freely downloadable without a credentialing / data-use agreement, which
keeps the project reproducible for any contributor.

> Note on record count: the widely-cited "21,837" figure is from the original v1.0.1.
> This project pins **v1.0.3**, which contains **21,799** records (a small number were
> removed for data quality across releases). The label space is 71 SCP-ECG statements
> (44 diagnostic + 19 form + 12 rhythm, with category overlaps) — verified against the
> downloaded `scp_statements.csv`.

## Safety posture

- Confidence thresholding with an explicit "needs manual review" state.
- Hallucination / consistency checking between the detector's labels and the generated
  text (the text may never assert a finding the detector did not surface).
- All outputs carry a visible "decision support — verify before acting" disclaimer.
