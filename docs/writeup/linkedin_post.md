# LinkedIn cross-post

Paste the body below directly into LinkedIn. Add the article link where marked. Target
length is deliberately short — LinkedIn truncates at ~210 characters before "…see more",
so the first two lines carry the whole click decision.

---

## Version A — leads with the honest-engineering angle (recommended)

Most ML posts show you the number that worked. Here's the model that lost, the metric that
lied, and the demographic gap I found in my own system and published anyway.

I built APEX: an ECG interpretation system that reads a 12-lead ECG, detects abnormalities
across 71 diagnostic categories, explains each finding in plain English, and — the part I
care about — flags its own output when it shouldn't be trusted.

It hits 0.920 macro-AUROC on PTB-XL, level with the published benchmark (0.925 top line).
Trained on a laptop, no GPU cluster.

But the number I'd actually defend in a review is different: 31.6% of everything the model
misses sits silently between 0.35 and 0.5 confidence. To a clinician, a 0.49 miss looks
identical to a confident negative. I know that because I went looking for it.

A few things I found and decided to publish rather than quietly fix:

→ The transformer I built lost to the plain CNN by 0.04 AUROC. It's in the repo as an
explicit "no improvement."

→ My first reliability metric reported a 100% failure rate. That's not a finding, it's a
bug — it was comparing one lead against a threshold designed for a whole signal. Redesigned
around rank instead of magnitude.

→ The model performs measurably worse on female patients: 0.925 AUROC male vs 0.906 female,
95% CI on the gap excludes zero. Worst on ST/T changes — the category that matters most for
ischemia, where under-recognition in women is already documented. Not corrected. Documented
as an open safety issue.

→ It over-flags heavily at the shipped threshold — but that's a calibration problem, not a
ranking problem (ECE ≈ 0.90 while AUROC stays 0.920). Diagnosing that correctly is the
difference between "retrain everything" and "calibrate the outputs."

Why the reliability layer exists at all: a meta-analysis of 78 studies (Cook et al., JAMA
Internal Medicine 2020) found pooled ECG interpretation accuracy of 68.5% among practicing
physicians — and the literature on computer-interpreted ECGs is consistent that clinicians
under-correct confident machines. So the interesting problem isn't "can a model classify
ECGs." It's building one a busy clinician can safely disbelieve.

Full technical write-up — architecture, reliability layer, results vs published baselines,
and the complete failure analysis: [LINK]

Code and model card: github.com/abdeltaehass/apex-arrhythmia-explainer

#MachineLearning #Healthcare #DeepLearning #MedicalAI #Python #PyTorch

---

## Version B — shorter, leads with the result

0.920 macro-AUROC on PTB-XL — level with the published benchmark, trained on a laptop.

That's the headline for APEX, the ECG interpretation system I just finished. It reads a
12-lead ECG, detects abnormalities across 71 diagnostic categories, explains each finding
in plain English, points at the signal region that drove it, and flags its own output when
it shouldn't be trusted.

The headline isn't the interesting part. These are:

→ 31.6% of everything it misses sits silently between 0.35 and 0.5 confidence — invisible
to the reader.

→ It performs measurably worse on female patients (0.925 vs 0.906, CI excludes zero),
worst on ST/T changes. Published, not corrected.

→ It over-flags heavily — but that's calibration (ECE ≈ 0.90), not ranking (AUROC 0.920).

→ The transformer I built lost to the plain CNN. That's in the repo too.

Why bother with a reliability layer? Pooled ECG interpretation accuracy among practicing
physicians is 68.5% (Cook et al., JAMA Intern Med 2020, 78 studies), and clinicians
demonstrably under-correct confident machines. The hard problem isn't classification — it's
building something a clinician can safely disbelieve.

Write-up: [LINK]
Code: github.com/abdeltaehass/apex-arrhythmia-explainer

#MachineLearning #MedicalAI #DeepLearning #Python

---

## Publishing notes

**Medium**
- Paste `technical_post.md` into a new story; Medium accepts pasted Markdown for headings,
  lists, and code fences, but **rebuild the tables by hand** — Medium has no native
  Markdown table support. Options: screenshot them, use a GitHub Gist embed, or reformat as
  short lists.
- Suggested tags: `Machine Learning`, `Healthcare`, `Deep Learning`, `Data Science`,
  `Artificial Intelligence`.
- Subtitle: use the italic line under the title.

**Personal site**
- The file is standard Markdown and drops into Jekyll/Hugo/Astro with a front-matter block.
- Tables and code fences render natively — this is the better home for the piece.

**LinkedIn**
- Post the body above **natively** (not as a link-only post) — LinkedIn suppresses reach on
  posts whose primary content is an outbound link. Put the link at the end, as written.
- The first two lines are all that show before "…see more". Both versions are written with
  that cut in mind.
- Best timing is generally Tue–Thu morning.

**Cross-linking**
- Add the published article URL to the repository README so the code and the write-up point
  at each other.
