# Phase 29 — ML4H 2026 workshop paper draft

`main.tex` + `references.bib`: a complete draft targeting the **ML4H 2026 Symposium**
(Machine Learning for Health), Proceedings track.

## Venue

| | |
|---|---|
| Venue | ML4H 2026 — [ml4h.ahli.cc](https://ml4h.ahli.cc/submit/call-for-papers/) |
| Track | **Proceedings** (archival, PMLR), 8 pages + 1 on acceptance |
| Deadline | **10 Sep 2026** (AoE); decisions 22 Oct; event 6–7 Dec |
| Review | Double-blind — the draft is anonymised (`\finalfalse`) |
| Subject area | *Impact & Society* (or *Models & Methods*) |

**Findings track instead?** ML4H's Findings track (4 pages, non-archival) explicitly invites
"negative results" and reproducibility studies, which is exactly what this paper is. It is
arguably the better fit, and it is a two-line change:

```latex
\mlhtrack{findings}   % was: proceedings
```

then cut §7 (specialist vs generalist) and §8 (related work) and compress §2 to fit 4 pages.
The Proceedings version is the default here because it is archival and rejected Proceedings
submissions are automatically considered for Findings — so submitting to Proceedings
strictly dominates.

## Building it

The ML4H template is not redistributable, so it is not vendored here. To compile:

1. Open the [ML4H 2026 template on Overleaf](https://www.overleaf.com/latex/templates/machine-learning-for-health-ml4h-2026-template/sqgwhtyswgcy).
2. Replace its `main.tex` with this one and add `references.bib`.
3. Compile (it uses `\documentclass[pmlr,twocolumn,10pt]{jmlr}`).

> **This draft has not been compiled.** No TeX toolchain is installed in the environment it
> was written in, so the page count below is an *estimate*, not a measurement, and the
> layout is unverified. Check length and float placement on the first build.
>
> Estimated length: ~2,750 words of body text (~3 pages two-column) plus 4 tables, 1 figure
> and references — **roughly 5–6 of the 8 permitted pages**. It is comfortably inside the
> limit rather than filling it. If reviewers expect a fuller Proceedings paper, the
> strongest additions available from committed artefacts are the federated-learning
> simulation (`docs/federated/`) and the EHR/FHIR integration (`docs/ehr/`), both of which
> were cut for focus rather than for lack of results.

The LaTeX was validated structurally (balanced environments and braces, every `\ref`
resolved, every `\cite` present in the `.bib`) but not typeset.

## Before submitting

- [ ] **Compile and check the 8-page limit.** Gross formatting violations are desk-rejected.
- [ ] **Anonymity**: the code-availability statement deliberately withholds the repository
      URL, which contains the author's username. Do not paste the real link until
      camera-ready (`\finaltrue`).
- [ ] **Reciprocal reviewing**: ML4H requires at least one author registered to review
      ≥3 papers, nominated at submission.
- [ ] **Ethics statement**: present (public de-identified data, no IRB required). Confirm
      this wording matches your institution's expectations.
- [ ] Re-run `make` targets if any number changes; every figure in the paper is generated
      by a script in `scripts/` and stored under `docs/`.

## Provenance of every number

The draft cites only measured results. Each maps to a committed artefact:

| Claim | Source |
|---|---|
| 0.9199 / 0.8929 macro-AUROC | `docs/model_comparison/baseline_comparison.json` |
| ECE 0.0793 → 0.0020 | `docs/calibration/report.json` |
| Distillation, 34× fewer params | `docs/distillation/report.json` |
| Sex gap 0.0188 [0.0044, 0.0325] | `docs/model_card/demographics.json` |
| RAG 0.060 → 0.120, McNemar p=0.064 | `docs/rag/report.json` |
| Feedback 9↑/0↓, 10↑/0↓, 1↑/18↓ | `docs/feedback/simulation.json` |
| Spanish gate 100% → 0% → 100% | `docs/i18n/eval.json` |
| Bootstrap 0.035 vs Hanley–McNeil 0.373 | `docs/synthesis/power.json` |
| Augmentation arms, QRS 227 vs 112 ms | `docs/synthesis/ablation.json` |
| Specialist vs generalist, latency, cost | `docs/benchmark/benchmark.json` |

## What the paper deliberately does not claim

- **Not state of the art.** 0.9199 against 0.925 for the best published PTB-XL baseline.
  The contribution is the failure analysis, and the paper says so in the introduction.
- **No clinical validation.** No deployment, no prospective evaluation, no clinician review
  of the Spanish output.
- **No frontier multimodal model was run** (no API credentials); published GPT-4o figures
  are cited, not measured, and are labelled as such.
