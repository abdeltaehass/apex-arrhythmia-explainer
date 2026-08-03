# I Built an ECG Interpretation System That Tells You When It's Wrong

### 71 diagnostic labels, 0.920 AUROC, and a reliability layer that flags its own output — plus every failure I found along the way

---

Most machine learning write-ups show you the number that worked. This one shows you the
model that lost, the metric that lied, and the demographic gap I found in my own system and
decided to publish instead of quietly fixing the plot.

I spent several weeks building **APEX (Arrhythmia Pattern Explainer)**: a clinical
decision-support tool that reads a 12-lead ECG, detects abnormalities across 71 diagnostic
categories, explains each finding in plain English, points at the part of the signal that
drove it, and — the part I care about most — flags its own output when it shouldn't be
trusted.

It reaches **0.920 macro-AUROC** on the PTB-XL test split, which puts it level with the
published benchmark. That's the headline. The rest of this post is about everything the
headline hides.

---

## 1. The problem: ECG misreading is a real, documented failure

The ECG is one of the most-ordered tests in medicine and one of the hardest to read well.
That combination is the problem.

The best evidence here is a **systematic review and meta-analysis of 78 studies covering
10,056 participants** (Cook, Oh & Pusic, *JAMA Internal Medicine*, 2020). Pooled
interpretation accuracy before training:

| | accuracy |
|---|---:|
| Medical students | **42.0%** |
| Residents | **55.8%** |
| Practicing physicians | **68.5%** |

Read that bottom row again. Among *practicing physicians*, pooled ECG interpretation
accuracy was roughly two in three. The authors' conclusion is blunt: physicians at all
training levels had deficiencies, **even after educational interventions**.

The consequences aren't abstract. A missed ST-elevation MI is a delayed catheterization
lab activation. A missed conduction block is a patient sent home. Errors run in both
directions — false positives drive unnecessary admissions and angiograms, false negatives
drive discharge of patients who are actively infarcting.

So: automate it? That's been tried, and it produced a second, subtler failure mode.
Computer-interpreted ECGs have been standard on machines for decades, and the literature
on them is consistent (see Schläpfer & Wellens, *JACC*, 2017, doi:10.1016/j.jacc.2017.07.723):
the algorithms are useful, they are also wrong often enough to matter, and — critically —
**clinicians under-correct them.** An inexperienced reader who sees a confident machine
label at the top of the printout tends to accept it. The standing recommendation in the
literature is that computer interpretations must be over-read by an experienced reader.

That framing shaped this entire project. The interesting problem is not "can a model
classify ECGs" — deep learning solved that well enough years ago. The interesting problem
is **building a system that a busy clinician can safely disbelieve**: one that surfaces its
own uncertainty loudly enough to resist automation bias.

That's the design constraint. Everything below follows from it.

---

## 2. The architecture

Four stages, each independently testable:

```
12-lead ECG ─▶ preprocessing ─▶ 1D CNN ─▶ grounding ─▶ report generation ─▶ reliability
   raw           resample         71-way    per-lead      Findings /          4 checks
   500 Hz        bandpass         sigmoid   saliency      Impression          ↓
                 R-peaks                                                  flags + gate
```

### Signal preprocessing

PTB-XL ships 10-second, 12-lead recordings at 500 Hz. The chain: resample to 100 Hz
(polyphase), band-pass 0.5–40 Hz to kill baseline wander and mains noise, Pan-Tompkins
R-peak detection for beat segmentation, per-lead z-score normalization. Baseline wander
energy drops ~97%.

One decision that mattered more than it should have: **caching**. The preprocessing chain
is deterministic, so running it every epoch was pure waste. Precomputing each split to a
`.npz` took a 20-epoch training run from hours to ~16 s/epoch on an Apple-silicon laptop.
No GPU cluster was used at any point in this project.

### Detection: a 1D CNN (and the transformer that lost)

The detector is a residual 1D CNN — convolutional stages over the (12, 1000) signal, global
average pooling, a 71-way sigmoid head. Multi-label, because real ECGs carry several
findings at once. Class-weighted BCE, because the label distribution is brutally long-tailed.

I also built a PatchTST-style 1D transformer and swept four configurations (CNN/transformer
× BCE/focal loss).

**The CNN won. The transformer came in 0.04 AUROC lower.**

That result is in the repository's comparison document as an explicit "no improvement,"
with the reason — a transformer of that size underfits without pretraining, and 17k
training records is not enough to close the gap. It trains 2.5× faster, which is a real
finding, just not the one I was hoping for. Focal loss was a wash.

I'm leading with this because a sweep that produces no improvement is the normal outcome of
most experiments, and reporting it is the difference between an engineering log and a
marketing document.

### Grounding: which leads actually drove the call

A finding without evidence is an assertion. I wanted per-lead attribution: for "anteroseptal
MI," *which leads* did the model use?

This turned out to be non-obvious. Standard Grad-CAM on the last conv layer gives a
**time-only** heatmap — because the network's very first layer, `Conv1d(12 → 64)`, mixes all
twelve leads together. The lead axis is destroyed at layer one; there is nothing left to
attribute to by the final block.

The fix is guided Grad-CAM: multiply the input-gradient magnitude (which retains the lead
axis) by the class-discriminative CAM envelope over time.

```
per_lead_saliency = |∂logit/∂x| · cam(t)      → shape (12, T)
```

Sanity checks on real records, which are the part I'd actually defend:

- **ST/T findings ground on repolarization** (the ST segment and T wave) in **57/57** cases.
- **Atrial fibrillation grounds *off* the P-wave region** onto the irregular baseline in
  **59/60** cases — which is clinically correct, since AF is defined by the *absence* of
  organized P waves. Corresponding RR-interval variability: 0.22 vs 0.09 in sinus rhythm.

The one outlier is in the report, not hidden: record 8392, where saliency stayed
P-dominant.

### Report generation: template first, LoRA second

Two backends. The shipped default is **template-based**: detected codes map through a
hand-authored clinical vocabulary into a structured `Findings:` / `Impression:` report. It's
deterministic and it *structurally cannot* hallucinate a finding the detector didn't
produce. For a medical application that property is worth more than fluency.

The second is a **LoRA fine-tune** (Mistral-7B-Instruct, `all-linear` target modules,
prompt/completion format with `completion_only_loss`). It's implemented and verified
end-to-end on a small model — a real smoke run on SmolLM2-135M moved loss 2.62 → 1.73 in one
epoch. The full 7B run needs a GPU this laptop doesn't have, and the write-up says exactly
that rather than implying a training run that never happened.

Building the report layer surfaced a genuine bug I'd otherwise have shipped: a record
labeled `NORM` *plus* a real pathology (PTB-XL does this) had the pathology silently
dropped from the impression. Found it by manually reviewing 20 generated reports against
PTB-XL's own human cardiologist text. Manual review of a small sample is still the highest
yield-per-hour debugging technique I know.

---

## 3. The reliability layer: the actual point of the project

If clinicians under-correct confident machines, then the most valuable thing the system can
do is **be visibly unconfident at the right moments**. Four independent checks run on every
record:

**1. Consistency.** Does the generated text assert a finding the detector never surfaced
above threshold? This catches hallucination at the text layer.

**2. Grounding conflict.** The text cites specific leads. Does the model's own saliency
agree? A cited lead that ranks in the bottom 2 of 12 by importance is flagged.

**3. Low confidence.** A separate, *higher* bar (0.7) than the surfacing threshold (0.5).
A finding can clear surfacing and still be tagged for review.

**4. Mutual exclusivity.** Clinical contradictions — sinus rhythm firing alongside atrial
fibrillation, complete right *and* left bundle branch block, two degrees of AV block at
once. Detector-only, independent of any text.

None of these asks the model whether it's confident. Checks 1–2 test the text against
independent signals; 3–4 test the detector against itself.

### The bug that taught me the most

My first implementation of the grounding-conflict check reported a **100% conflict rate** on
a 200-record pilot.

A 100% failure rate is not a finding, it's a bug — no measurement that extreme should be
believed before the raw values are inspected. It had reused `is_grounded()` from the
grounding module, which uses an *absolute* magnitude threshold tuned for evaluating one
whole-signal trace. Inside a globally-normalized (12, T) tensor, only the single most
salient lead at its single most salient instant ever approaches that bar. Every other lead
looks "unsupported" by construction.

The redesign uses **rank** instead of magnitude — bottom-k of 12 — calibrated against real
validation saliency *before* picking the number. Post-fix rates on the full validation set:

| check | rate |
|---|---:|
| Consistency violations | **0.0%** |
| Low confidence | 75.9% |
| Mutual exclusivity | 33.3% |
| Grounding conflict (per citation) | 15.7% |
| Grounding conflict (per record) | 82.8% |

Two things worth reading carefully. The **0.0%** consistency rate is the expected floor for
a template backend — it validates the round-trip at scale rather than proving anything
impressive. And I report grounding conflict **both** per-citation (15.7%) and per-record
(82.8%), because records average ~11 citations and the per-record number compounds. Quoting
only the flattering one would be a lie of framing.

The mutual-exclusivity result was its own finding: **9 of the top 10 firing pairs are
`NORM` + some pathology** — the same tension the manual report review had surfaced,
confirmed at scale.

### Serving

FastAPI: `/analyze`, `/validate`, `/health`, `/metrics`. Inference is stateless — nothing
about a request is written to disk. Warm latency is **~6 ms** on CPU (~12 ms over HTTP).

That number was originally **1345 ms**, because `analyze_signal` reloaded the checkpoint
from disk on every single call. Memoizing the model by `(checkpoint, device)` fixed it. A
192× speedup from four lines of caching is not clever engineering; it's a reminder to
profile before optimizing anything else.

There's also a **paper-ECG digitization** path — photograph a printed ECG, recover the
signal via classical CV (grid detection, adaptive luminance thresholding, per-column
centroid tracing). Round-trip fidelity: **0.885** on clean renders, **0.815** on simulated
phone photos, **0.706** on heavy ones. It's classical CV rather than a learned model
because no dataset of real paper-ECG photos paired with ground-truth signals exists, and
I'd rather ship an honest 0.885 than a learned model trained on data I don't have.

---

## 4. Results

Evaluated on the **PTB-XL test split** (fold 10, 2,198 records, official patient-level
stratification, never touched during development). Thresholds tuned on validation and
applied to test.

| Model | All-task AUROC | Source |
|---|---:|---|
| xresnet1d101 | 0.925 | Strodthoff et al. 2021 |
| inception1d | 0.925 | Strodthoff et al. 2021 |
| **APEX (this work)** | **0.920** | — |
| resnet1d_wang | 0.919 | Strodthoff et al. 2021 |
| lstm | 0.907 | Strodthoff et al. 2021 |
| Wavelet + NN | 0.849 | Strodthoff et al. 2021 |

APEX lands third, level with `resnet1d_wang`, 0.005 off the top — from a compact CNN with
no ensembling, trained on a laptop.

A note on baseline selection, because it's the kind of thing that quietly inflates a lot of
write-ups. My original plan was to compare against **Ribeiro et al. 2020**, the landmark
deep-ECG paper. It isn't a valid comparison: that model was trained on the **CODE** dataset
(different data, six classes, reports F1 not AUROC). Citing it next to a PTB-XL number
would have looked more impressive and meant nothing. The correct PTB-XL benchmark is
Strodthoff et al. 2021, which is what the table uses.

For scale on how much domain-specific training buys: **GPT-4o reading the same ECGs as
images, zero-shot, reaches ~41% multiclass accuracy** (published, *JMIR AI* 2025). A
generalist multimodal model is not close.

---

## 5. Honest failure analysis

I ran a dedicated adversarial phase: five curated hard-case cohorts, scored at the
**shipped** operating rule (probability ≥ 0.5) rather than a post-hoc tuned optimum,
because the question is what the system *actually does*, not what it could do at its best
threshold.

| Cohort | n | label recall |
|---|---:|---:|
| Overall | 2,198 | 83.7% |
| Normal ECGs only | 912 | 87.5% |
| Significant artifact | 160 | 81.7% |
| Whole-record noise | 93 | **79.4%** |
| Borderline confidence | 441 | **75.6%** |
| 12 rarest labels | 24 | **72.9%** |
| ≥5 simultaneous conditions | 224 | 83.0% |

Six failure modes, all measured:

**Rare labels are the biggest blind spot.** Recall falls to 72.9% on labels with 8–24
training examples, and **25% of that cohort had a dangerous miss** — an urgent
ST-elevation/injury code present and not surfaced. The system doesn't abstain on rare
findings; it emits a confident-looking negative.

**Silent near-misses.** **31.6%** of all missed labels had a probability between 0.35 and
0.5. To a reader, a 0.49 miss is indistinguishable from a confident negative. That's the
mode I find most dangerous, because it's invisible.

**One dangerous miss, documented in full.** Record 968: urgent `INJIL` (inferolateral
injury) present in ground truth. The system surfaced 17 other findings, missed that one, and
showed a **yellow** banner rather than red. It's written up completely, with the full
surfaced list, in the repository. Publishing your worst case is the only version of a
safety analysis that's worth anything.

**Massive over-flagging — and it's a calibration problem, not a ranking problem.** At
threshold 0.5 the model surfaces **5.09 absent labels per record** and tags a spurious
diagnostic code on **48.7%** of normal ECGs. That looks damning until you notice the same
model scores 0.920 AUROC, which is *threshold-free*. The discrimination is fine; the
probabilities are inflated by design, because class-weighted BCE inflates them — mean
predicted probability 0.118 against a base rate of 0.039, about 3x too high. The ranking is
good and the operating point is wrong. Diagnosing that correctly is the difference between
"retrain everything" and "calibrate the outputs" — and calibrating them is exactly what
fixed it (below).

### The demographic result I published anyway

I measured whether AUROC differs by age or sex, with two guards that make the answer
trustworthy: every subgroup scored on the **same label set** (macro-AUROC silently skips
single-class labels, so unequal coverage makes naive comparisons meaningless), and
**bootstrap confidence intervals** on every figure, since subgroup sizes range from 13 to
721.

**By sex:**

| | n | macro-AUROC | 95% CI |
|---|---:|---:|---|
| Male | 1,132 | **0.9250** | 0.914 – 0.934 |
| Female | 1,066 | **0.9062** | 0.896 – 0.919 |

Gap **+0.0188, CI +0.0044 – +0.0325**. The interval excludes zero: this is a real
disparity, not noise. It's widest on **ST/T change: 0.934 male vs 0.891 female** — which is
the category that matters clinically, given the documented under-recognition of ischemic
presentations in women.

**By age**, performance declines monotonically: 0.906 (18–39) → 0.903 (40–59) → 0.889
(60–74) → **0.864 (75+)**. The 18–39 vs 75+ gap is +0.0417, CI +0.0065 – +0.0683 — also
excluding zero.

**Neither disparity is corrected in the current model.** Both are in the model card as open
safety issues. A fairness section that only ever reports "no significant difference found"
should make you suspicious about how hard anyone looked.

One more correction from that phase. The plan said "PTB-XL is adult-only, so pediatric use
is out of scope." PTB-XL is **not** adult-only — the youngest patient is **2 years old**,
and 133 records are under 18. The conclusion survives but the real reason is worse than the
assumed one: pediatric ECGs are *present yet vanishingly rare*, so the model will happily
emit confident output on one. (Measured pediatric AUROC: 0.741.) I also had the dataset
license wrong in a first draft — PTB-XL is CC BY 4.0, not non-commercial — which matters,
because it means commercial restriction comes from *device regulation*, not licensing.

---

## 6. What I'd do next

**~~Calibration, first and by a wide margin.~~ Done — and it was the whole ballgame.**
Post-hoc calibration fitted on the validation fold cut **ECE 0.079 → 0.002** and spurious
surfaced labels **5.09 → 0.35 per record**, with macro-AUROC unchanged at 0.920.

Two things I did not expect. First, plain **temperature scaling made ECE slightly worse**
(0.079 → 0.088): temperature can only sharpen or soften a distribution, and this model's
error is a *bias* from class-weighted BCE, which needs a per-label intercept to undo —
vector scaling did it. Second, while writing the calibration code I found that the
**"ECE ≈ 0.90" I had been quoting for four phases was wrong**: the old implementation used
the multi-class formulation (mean probability vs *accuracy*) on independent sigmoids, which
charges ~0.99 of error to the correctly-predicted negatives that are 78% of all
predictions. The true uncalibrated value was 0.079. The diagnosis survived — the model was
genuinely ~3x over-confident — but the magnitude was an artifact of my own metric.

**A sub-threshold tier for high-consequence codes.** Rather than silently dropping a 0.42
ST-elevation probability, surface it as "possible — below confidence." Directly targets the
31.6% silent-near-miss mode and the dangerous misses.

**More data for the tail.** The rare-label failure is a data problem, not an architecture
problem. Merging PTB-XL with CPSC, Chapman-Shaoxing, and Georgia (the PhysioNet/CinC 2020
family) would put the rarest labels into the hundreds of examples instead of single digits.

**Pretraining, then revisit the transformer.** The transformer lost because it underfits on
17k records. Self-supervised pretraining on unlabeled ECG is the standard fix, and it's the
one result in this project I'd expect to flip with a GPU budget.

**Investigate the sex gap rather than just reporting it.** Is it label prevalence, signal
amplitude differences, or genuinely harder presentations? Reweighting is easy; understanding
first is correct.

**Prospective validation.** Everything here is retrospective on one dataset's held-out
fold, collected in Germany between 1989 and 1996. No claim in this post survives
distribution shift without re-validation.

---

## What I'd want a reader to take from this

The model is competitive — 0.920 against a published 0.925 top line. But the number I'd
actually defend in a review is a different one: **31.6% of misses sit silently between 0.35
and 0.5**, and I know that because I went looking for it.

Building a system that performs well is table stakes. Building one that tells you *when it
doesn't* — and publishing the times it didn't, including a demographic gap in my own model
and a dangerous miss on a real record — is the part that would make me trust it near a
patient.

It still isn't a medical device, it isn't calibrated, and it shouldn't touch a clinical
decision without a clinician reading the ECG themselves. That's in the model card too.

---

**Code, model card, and every report referenced here:**
[github.com/abdeltaehass/apex-arrhythmia-explainer](https://github.com/abdeltaehass/apex-arrhythmia-explainer)

*Built with PyTorch, FastAPI, and Gradio on PTB-XL v1.0.3. 277 tests. No GPU cluster.*

---

### References

1. Cook DA, Oh S-Y, Pusic MV. **Accuracy of Physicians' Electrocardiogram Interpretations:
   A Systematic Review and Meta-analysis.** *JAMA Internal Medicine*. 2020;180(11).
   doi:10.1001/jamainternmed.2020.3989
2. Schläpfer J, Wellens HJ. **Computer-Interpreted Electrocardiograms: Benefits and
   Limitations.** *Journal of the American College of Cardiology*. 2017.
   doi:10.1016/j.jacc.2017.07.723
3. Wagner P, Strodthoff N, Bousseljot R-D, et al. **PTB-XL, a large publicly available
   electrocardiography dataset.** *Scientific Data*. 2020;7:154.
   doi:10.1038/s41597-020-0495-6
4. Strodthoff N, Wagner P, Schaeffter T, Samek W. **Deep Learning for ECG Analysis:
   Benchmarks and Insights from PTB-XL.** *IEEE Journal of Biomedical and Health
   Informatics*. 2021;25(5):1519–1528. doi:10.1109/JBHI.2020.3022989
