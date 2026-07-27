# Phase 11 — Clinical dashboard & Hugging Face Spaces deployment

The dashboard ([`app/frontend/app.py`](../../app/frontend/app.py), Gradio) wraps the full
pipeline:

- **Upload** a 12-lead ECG — a signal file (`.npy`/`.csv`/`.json`) **or a photo of a
  paper ECG** (digitized via Phase 10).
- **Left panel** — the 12-lead waveform with per-finding grounding overlays, one colour
  per finding; pick a finding to highlight just its saliency (the ECG redraws with that
  finding emphasised, others dimmed).
- **Right panel** — the structured report: findings with confidence bars, reliability
  flags, the impression, and the full explanation.
- **Severity banner** — 🟢 green (nothing needs review) / 🟡 yellow (review recommended)
  / 🔴 red (urgent ST-elevation / injury pattern), from `src/serving/severity.py`.
- **Disclaimer banner** — always visible.

Previews (real app output): [`preview/ecg_mi.png`](preview/ecg_mi.png),
[`preview/ecg_afib_from_photo.png`](preview/ecg_afib_from_photo.png) (an ECG digitized
from a paper-ECG photo), and the report cards `preview/report_*.html`.

## Run locally

```bash
make ui          # or: python app.py   (Gradio at http://localhost:7860)
```

Needs the detector checkpoint at `outputs/final_best.pt`. The bundled
`data/raw/ptbxl/scp_statements.csv` (12 KB) supplies the label dictionary, so the full
waveform download is **not** required to run the demo.

## Deploy to Hugging Face Spaces

The Space is a git repo running the Gradio SDK. `app.py` (repo root) is the entry point.
Two things aren't in git and must be added to the Space: the **model checkpoint**
(34 MB, via LFS) and a **Space `README.md`** carrying the config front-matter
([`app/frontend/space_README.md`](../../app/frontend/space_README.md)).

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login                                  # your HF token

# create the Space (Gradio SDK)
huggingface-cli repo create apex-arrhythmia-explainer --type space --space_sdk gradio

git clone https://huggingface.co/spaces/<you>/apex-arrhythmia-explainer space && cd space

# bring in the app + library + label dict + entry point + slim requirements
cp -r ../app ../src ../app.py .
mkdir -p data/raw/ptbxl && cp ../data/raw/ptbxl/scp_statements.csv data/raw/ptbxl/
cp ../app/frontend/requirements.txt requirements.txt   # slim runtime deps for the Space
cp ../app/frontend/space_README.md README.md           # the config front-matter

# the model checkpoint (34 MB) via git-lfs
git lfs install && git lfs track "*.pt"
mkdir -p outputs && cp ../outputs/final_best.pt outputs/

git add -A && git commit -m "APEX dashboard" && git push
```

The Space builds from `requirements.txt` and launches `app.py`; first request warms the
model (~1–2 s), then reads are fast (Phase-9 numbers). Free CPU hardware is enough — the
detector is a small 1D-CNN and the default explanation backend is the deterministic
template (no LLM download).

> **Note:** the actual `huggingface-cli login` + push has to be run with your own HF
> account — it can't be done from this repo's tooling. Everything else (the app, the
> entry point, the Space README, the slim requirements, the bundled label dictionary) is
> ready in the repo.

## Notes / limitations

- **"Hover to highlight"** is implemented as a **click/select** (a radio of the grounded
  findings) — Gradio doesn't expose true hover events between the report list and the
  plot, so selecting a finding is the equivalent interaction.
- The demo uses the **template** explanation backend by default (deterministic, no LLM).
  Set `APEX_BACKEND=local` with a fine-tuned adapter (Phase 6) or `claude` with
  `ANTHROPIC_API_KEY` for richer prose.
- Grounding overlays reflect the model's *actual* saliency, imperfections included (see
  the Phase-5 report) — e.g. an inferior-MI finding may highlight a lateral lead. That is
  shown faithfully rather than corrected.
