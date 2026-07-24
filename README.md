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
  eval/               metrics, consistency checker, hallucination flagging, reliability checks
  serving/            structured JSON output schema + serializer + input validation
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

Model-improvement sweep (CNN vs. transformer × BCE vs. focal) + comparison table:

```bash
make experiments               # runs the sweep, logs to docs/model_comparison/runs.jsonl
make compare                   # -> docs/model_comparison/comparison.md (vs. published PTB-XL)
```

Grounding — per-lead saliency for a detected label, and the clinical sanity sweep:

```bash
python scripts/run_grounding.py --ecg-id 18550 --label NDT   # one record -> figure + JSON
make grounding                                               # AFIB + STTC sanity sweep
# -> docs/grounding/ (figures, scan JSONs, sanity_check.md)
```

Generation — build the Findings/Impression report dataset, then LoRA fine-tune:

```bash
make gen-data          # PTB-XL SCP codes -> data/processed/generation/{train,val,test}.jsonl
make gen-train-smoke   # tiny end-to-end LoRA check, runs anywhere (no GPU needed)
make gen-train         # the real run: LoRA on mistralai/Mistral-7B-Instruct-v0.3, needs a GPU
```

Reliability — consistency/grounding/confidence/mutual-exclusivity checks on the real detector:

```bash
make reliability   # runs the full validation set -> docs/reliability/report.md + report.json
```

Set up experiment tracking:

```bash
cp .env.example .env      # fill in WANDB_ENTITY, ANTHROPIC_API_KEY
wandb login
make wandb-init           # creates the W&B project + logs target baselines
```

Structured output — wrap the whole pipeline into the Phase-8 JSON schema:

```python
from src.serving import analyze_signal
report = analyze_signal(signal_12xT, sampling_rate=100, backend="template", with_grounding=True)
report.model_dump()        # findings[] + impression + explanation + consistency + review_recommended
report.review_recommended  # the single review gate
```

Run the app:

```bash
make api    # FastAPI at http://localhost:8000  (/health, /validate, /analyze -> APEXReport)
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
- **Phase 4:** model improvement — swept CNN vs. PatchTST-style 1D transformer × BCE vs.
  focal loss, all logged (arch/hparams/AUROC/time). **Best = class-weighted-BCE CNN at
  test macro-AUROC 0.920** (matches published `resnet1d_wang`); transformer and focal
  did not beat it. Comparison + published PTB-XL results in
  [`docs/model_comparison/comparison.md`](docs/model_comparison/comparison.md). ✅
- **Phase 5:** grounding — a Grad-CAM equivalent for 1D ECG that returns a **per-lead
  saliency trace** for any detected label (guided Grad-CAM: class-discriminative temporal
  CAM × per-lead input gradients). Sanity-checked against clinical intuition: ST/T
  findings ground on the ST/T segment (**57/57**), AF grounds off the P wave and on the
  irregular baseline (**59/60**), with the one disagreement documented, not hidden
  ([`src/grounding/`](src/grounding/),
  [`docs/grounding/sanity_check.md`](docs/grounding/sanity_check.md)). ✅
- **Phase 6:** generation — PTB-XL SCP codes → structured input → templated
  **Findings/Impression** report (the SFT target; `src/generation/templater.py`,
  71-code clinical vocabulary in `vocab.py`), a real measured heart rate per record
  (Pan-Tompkins, not guessed), and a LoRA fine-tune pipeline
  (`src/generation/train_lora.py`, `trl.SFTTrainer`, default
  `mistralai/Mistral-7B-Instruct-v0.3`) verified end-to-end with a tiny local model
  (loss 2.62 → 1.73 over one epoch) since this machine has no GPU. **20 manually
  reviewed examples** compare generated text against PTB-XL's own human report —
  [`docs/generation/examples_review.md`](docs/generation/examples_review.md) — and
  surfaced (and fixed) a real templater bug along the way. ✅
- **Phase 7:** consistency & reliability checker — four checks
  (`src/eval/reliability.py`) composed into one report: **consistency warnings**
  (text asserts an unsurfaced finding), **grounding conflicts** (a cited lead ranks
  among the least-important for that finding in the Phase-5 saliency, not just
  "ungrounded" at the whole-finding level), a tunable **low-confidence flag**
  (default 0.7, `CFG.low_confidence_threshold`), and a curated **mutual-exclusivity**
  rule set (e.g. sinus rhythm + atrial fibrillation, complete + incomplete RBBB, AV
  block degree). Run on the full validation set (2,183 records, real detector output):
  consistency 0% (expected — the template backend can't hallucinate by construction),
  low-confidence 75.9%, mutual-exclusivity 33.3% (dominated by `NORM` co-occurring with
  real pathology — the same tension the Phase-6 review found, now confirmed at scale),
  grounding-conflict 15.7% per cited lead.
  [`docs/reliability/report.md`](docs/reliability/report.md). ✅
- **Phase 8:** structured JSON output layer — one Pydantic response schema
  (`src/serving/schema.py`: `findings[]` with label/confidence/leads/flag-status,
  `impression`, full `explanation`, `consistency` result, `review_recommended` gate)
  and a serializer (`src/serving/serializer.py`) that folds detection + generation +
  grounding + the Phase-7 reliability flags into it. Input validation rejects non-12-lead
  recordings (HTTP 422) and flags sub-5s recordings as unreliable. The FastAPI backend
  now returns real reports (`/analyze`, `/validate`); full pipeline runs in ~0.7 s with
  grounding. Schema-validation test suite + sample outputs
  ([`docs/serving/schema.md`](docs/serving/schema.md),
  [`docs/serving/sample_report.json`](docs/serving/sample_report.json)). ✅
- Phase 9+: calibration, frontend.

## Data & ethics

PTB-XL is de-identified and openly licensed, but this project handles medical data
and produces medical-adjacent output. Keep the review gate on, never present outputs
as diagnoses, and keep the "verify before acting" disclaimer visible everywhere.
