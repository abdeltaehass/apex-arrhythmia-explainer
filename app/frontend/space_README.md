---
title: APEX — Arrhythmia Pattern Explainer
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Grounded, explained 12-lead ECG readings from a signal or a photo.
---

# APEX — Arrhythmia Pattern Explainer

Clinical decision-support demo: upload a 12-lead ECG (a signal file **or a photo of a
paper ECG**) and get a grounded, explained reading — detected findings with confidence,
per-finding saliency overlays on the waveform, a plain-English report, reliability
flags, and a red/yellow/green triage banner.

> ⚕️ APEX is a clinical decision-support tool. All outputs require clinical review and
> should not be used as a standalone diagnosis.

**This file is the Space's `README.md`** — copy it to `README.md` at the Space repo root
when deploying (the repo's own `README.md` is the project readme). See
`docs/frontend/deploy.md` for the full steps, including uploading the model checkpoint.
