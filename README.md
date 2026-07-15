# APEX — Arrhythmia Pattern Explainer

APEX is a clinical decision-support tool that reads 12-lead ECG signals, detects
cardiac abnormalities across **71 diagnostic categories**, generates a structured
plain-English clinical explanation, and flags low-confidence predictions for manual
review — **designed to assist clinicians, not replace them.**

> ⚠️ Decision support only. Not a diagnostic device. Verify every output against the
> full clinical picture.

See [`docs/problem_statement.md`](docs/problem_statement.md) for scope and
[`docs/target_metrics.md`](docs/target_metrics.md) for the metrics we track.

## Repository layout

```
data/                 raw signals, processed splits, annotation manifests (gitignored)
src/
  preprocessing/      filtering, segmentation, normalization
  detection/          1D CNN / transformer model + dataset
  generation/         LLM prompting / fine-tuning + inference
  grounding/          attention / saliency explainability layer
  eval/               metrics, consistency checker, hallucination flagging
  data/               PTB-XL download helpers + SCP label handling
  config.py           single source of truth (paths, targets, W&B)
app/
  backend/            FastAPI service
  frontend/           Gradio UI
notebooks/            EDA and experiment logs
scripts/              dataset download, W&B init
configs/              experiment configs
tests/                unit tests
```

## Quickstart

APEX targets **Python 3.11** (some ML wheels — torch, wfdb — may not yet publish
3.14 builds). Create an isolated env:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt   # or: make setup
```

Get the data (PTB-XL is open access, no credentialing):

```bash
make data-meta      # just the metadata CSVs — fast, for EDA
make data           # full dataset, several GB
```

Run the EDA + build the patient-level splits (needs only `make data-meta`):

```bash
make manifests                 # data/manifests/{train,val,test}.csv (split by patient)
make eda                       # docs/eda/ figures, prevalence tables, summary.md
jupyter lab notebooks/01_eda.ipynb
```

Preprocessing (resample → band-pass 0.5–40 Hz → Pan-Tompkins → z-score):

```bash
make data-sample               # a few MB of curated waveforms (no need for full data)
jupyter lab notebooks/02_preprocessing.ipynb   # raw vs. clean across 6 diagnostic groups
```

Train the baseline detector (needs the 100 Hz waveforms):

```bash
make data-100                  # ~0.5 GB of 100 Hz records (parallel, S3 mirror)
make train                     # 20-epoch 1D-ResNet -> docs/baseline/ + outputs/baseline_best.pt
# WANDB_MODE=offline is fine without a login; `wandb login && wandb sync wandb/latest-run` later
```

Set up experiment tracking:

```bash
cp .env.example .env      # fill in WANDB_ENTITY, ANTHROPIC_API_KEY
wandb login
make wandb-init           # creates the W&B project + logs target baselines
```

Run the app skeleton:

```bash
make api    # FastAPI at http://localhost:8000  (/health, /analyze)
make ui     # Gradio UI
```

## Phase status

- **Phase 0:** problem statement, repo skeleton, target metrics, W&B init,
  PTB-XL download tooling. ✅
- **Phase 1:** data acquisition + EDA — label distribution across the 71 SCP
  statements, class-imbalance and demographic analysis, and patient-level
  train/val/test manifests (see [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
  and [`docs/eda/summary.md`](docs/eda/summary.md)). ✅
- **Phase 2:** signal preprocessing — resample → band-pass 0.5–40 Hz → Pan-Tompkins
  R-peak detection → per-lead z-score, wired into `PTBXLDataset`
  ([`src/preprocessing/`](src/preprocessing/),
  [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)). ✅
- **Phase 3:** baseline detector — 1D ResNet (residual blocks → global avg pool →
  71-way sigmoid), class-weighted BCE, 20 epochs, W&B logging. **Val macro-AUROC 0.914**
  ([`src/detection/`](src/detection/), table in
  [`docs/baseline/`](docs/baseline/baseline_summary.md)). ✅
- Phase 4+: grounding, generation, calibration, evaluation harness, app wiring.

## Data & ethics

PTB-XL is de-identified and openly licensed, but this project handles medical data
and produces medical-adjacent output. Keep the review gate on, never present outputs
as diagnoses, and keep the "verify before acting" disclaimer visible everywhere.
