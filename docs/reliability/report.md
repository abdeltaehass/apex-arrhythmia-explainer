# Phase 7 — Consistency & Reliability Checker: validation-set report

Four independent checks (`src/eval/reliability.py`) run against the **entire
validation fold** (2,183 records, PTB-XL's official fold 9 — the same split
`src/detection/train.py` tunes thresholds on) using the real Phase-3/4 detector
(`outputs/final_best.pt`, class-weighted-BCE CNN, test macro-AUROC 0.920). Regenerate
with `make reliability` (`scripts/run_reliability_report.py`).

## Method

For every validation record: run the detector, threshold at `CFG.review_threshold`
(0.5) to get the surfaced label set — exactly what the model actually flags — then
render the deterministic template report (`src/generation/templater.py`) from those
labels **using the detector's own confidences**, not synthetic ones like Phase 6's
training data. This is the first time the templater has been exercised against live
model probabilities at scale rather than hand-picked or synthetic examples. Re-parse
the rendered text (`src.generation.parse.asserted_findings`) and run all four checks.

The consistency, low-confidence, and mutual-exclusivity checks are cheap (no
gradients) and ran over all 2,183 records. The grounding-conflict check needs a
Grad-CAM backward pass per (record, finding) — run over a random sample of 250 records
(of 1,140 eligible, i.e. carrying at least one lead-localizing finding), mirroring the
sample size used for the Phase-5 grounding sanity sweep.

## Results

| Check | Rate | Count |
|---|---:|---:|
| **Consistency warning** (text asserts an unsurfaced finding) | **0.0%** | 0 / 2,183 |
| **Low confidence** (a surfaced finding < 0.70, the Phase-7 tunable bar) | **75.9%** | 1,657 / 2,183 records (5,352 individual flags) |
| **Mutual exclusivity** (two contradictory labels both surfaced) | **33.3%** | 727 / 2,183 records (1,416 conflicts) |
| **Grounding conflict**, per record | 82.8% | 207 / 250 sampled records |
| **Grounding conflict**, per cited lead | **15.7%** | 443 / 2,823 lead-citations |
| **Any flag** | 79.9% | 1,744 / 2,183 |

The consistency-warning rate of exactly 0% is the **expected floor for a
template-rendered explanation**, not a vacuous result: the template is built directly
from the surfaced label set, so `asserted ⊆ surfaced` holds by construction — this run
is the first confirmation that holds at scale (2,183 real, multi-label detector
outputs, not just the 71 synthetic single-code cases checked in Phase 6) rather than
just on paper. It also matches the `docs/target_metrics.md` consistency-rate target
(≥ 0.98) trivially. The real test of this number comes once the Phase-6 LoRA
fine-tune is actually run on a GPU and swapped in as the backend — this harness is
what will measure it when that happens. The unit tests (`tests/test_reliability.py`)
confirm the checker *does* fire when a text asserts something ungrounded (verified with
injected inconsistencies), so a 0% rate here reflects the template's determinism, not a
blind checker.

### Grounding conflict: why report two rates

A record with several cited leads compounds even a modest per-citation flag rate into a
much higher per-record one (multiple independent chances for "at least one" to fire) —
the 250-record sample averaged **11.3 lead-citations per record**. The per-citation rate
(15.7%) is the fairer "how often does this actually happen" number; the per-record rate
(82.8%) answers a different question ("would at least one flag show up on this
record's full report"). Both are reported so neither figure is read out of context.

## What's driving each rate

**Low confidence** — top codes: `ABQRS` (315), `IVCD` (302), `VCLVH` (262), `NST_`
(249), `SR` (239), `QWAVE` (229), `STD_` (218), `IMI` (215), `AMI` (204), `NDT` (201).
These are exactly the subtler/rarer statements the Phase-3 baseline already flagged as
weak (`docs/baseline/`, e.g. posterior MI AUROC 0.63) and the class-weighted-BCE
training that Phase 3 noted inflates probabilities generally (ECE ≈ 0.90) — a record
with several surfaced findings has a good chance at least one sits in the 0.5–0.7 band,
which is exactly the point of this check: it's a *defense-in-depth* re-flag, not a
claim that the whole record is unreliable.

**Mutual exclusivity** — top pairs: `NORM`+`VCLVH` (177), `NORM`+`NDT` (147),
`NORM`+`IMI` (146), `NORM`+`IVCD` (111), `NORM`+`LVH` (101), `CRBBB`+`IRBBB` (76),
`NORM`+`NT_` (64), `NORM`+`IRBBB` (64), `NORM`+`NST_` (59), `NORM`+`LAO/LAE` (50).
**9 of the top 10 are `NORM` co-occurring with a real abnormality** — this is the exact
same pattern the Phase-6 manual review caught in 2 hand-picked examples
(`docs/generation/examples_review.md`, cases 4 & 17), now confirmed at scale: PTB-XL's
`NORM` label evidently functions more like "no *acute* finding" than "zero abnormality
codes present," co-occurring with `IMI`/`LVH`/conduction findings hundreds of times in
this fold alone. `CRBBB`+`IRBBB` (76) is a *different*, sharper finding — these are
definitionally exclusive (a bundle can't be both incompletely and completely blocked
by the same QRS-duration criterion) and their frequent co-occurrence is a genuine
PTB-XL annotation-consistency issue, not a softer labeling-convention question like the
`NORM` pattern.

**Grounding conflict** — top codes: `IMI` (121), `ISCAL` (48), `ILMI` (41), `ISCIL`
(35), `AMI` (34), `ISCIN` (32), `ALMI` (23), `ASMI` (22), `ISCLA` (21), `IPLMI` (14) —
entirely infarction/ischemia codes, which also carry the widest lead territories (3–8
leads each per `vocab.TERRITORIES`), so they dominate by citation volume as much as by
any code-specific weakness. This also lines up with the Phase-5 grounding writeup: the
CNN's stem mixes all 12 leads at its very first layer, so the per-lead signal recovered
via guided Grad-CAM is real but comparatively mild — a 15.7% per-citation "bottom-2"
rate is consistent with a *soft*, not absent, lead-localization signal, matching Phase
5's own framing of `LeadSaliency` as a sensitivity attribution, not proof of exclusive
lead use.

## A methodology note worth keeping

The grounding-conflict check was **not** built the way it first shipped. The initial
design reused `src/grounding/saliency.is_grounded` (an absolute-magnitude,
sustained-duration threshold) per cited lead and returned a **100% conflict rate** on
a 200-record pilot run — checked against raw values before trusting it, and the
cause turned out to be a scale mismatch: `is_grounded`'s threshold was tuned for a
single whole-signal trace (Phase 5), not for comparing one lead against eleven others
inside a single, globally-normalized `(12, T)` tensor, where only the single most
salient lead+instant in the whole strip ever approaches the threshold. The check was
redesigned around each cited lead's **rank** among the 12 by `lead_importance` (bottom-2
of 12 by default) instead of an absolute bar — calibrated against real detector output
first (see the module docstring in `src/eval/reliability.py`) rather than picked to hit
a target number.

## Known limitations

- **Consistency is only as informative as the backend under test.** A template
  backend can never show a real hallucination — it literally cannot generate a finding
  it wasn't given. This report validates the *harness*, not generation quality; that
  needs the fine-tuned LoRA model from Phase 6.
- **`NORM` co-occurrence is a data-quality/labeling-convention question, not obviously
  a checker bug** — but the mutual-exclusivity rate (33.3%) should be read as "how
  often does PTB-XL's own label set contain this tension," not "how often does the
  generator make a contradictory claim."
- **The per-record grounding rate (82.8%) is inflated by citation volume**, not a
  claim that most records are poorly grounded — see the per-citation rate (15.7%) and
  the two-rates discussion above.
- **Mutual-exclusivity rules are a curated, documented list**
  (`src/eval/reliability.py`, `EXCLUSIVITY_PAIRS`), not learned or exhaustive — they
  cover rate direction, sinus-vs-disorganized-atrial-rhythm, bundle-branch
  completeness, AV-block degree, and the `NORM` compatibility whitelist reused from
  Phase 6. Real but rarer contradictions may exist outside this list.
- **Grounding-conflict sampling (n=250) is smaller than the full validation set**
  because Grad-CAM needs a backward pass per (record, finding); the per-citation rate
  is stable at this sample size (matches the 200-record pilot's 16.0% within noise).
