# APEX — Arrhythmia Pattern Explainer

APEX is a clinical decision-support tool that reads 12-lead ECG signals, detects
cardiac abnormalities across **71 diagnostic categories**, generates a structured
plain-English clinical explanation, and flags low-confidence predictions for manual
review — **designed to assist clinicians, not replace them.**

> ⚠️ Decision support only. Not a diagnostic device. Verify every output against the
> full clinical picture.

📋 **Read [`MODEL_CARD.md`](MODEL_CARD.md) before using this** — intended use,
out-of-scope populations, measured limitations, privacy, and the demographic
performance breakdown (including a documented sex disparity).

📝 **[Technical write-up](docs/writeup/technical_post.md)** — how it's built, what it
scores against published baselines, and every failure mode I found.

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
  digitization/       paper-ECG image <-> signal (render + digitize)
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

Run the API service (Phase 9):

```bash
make api          # uvicorn at http://localhost:8000
#   POST /analyze   signal file (.npy/.csv/.json), ECG image, or JSON body -> APEXReport
#   POST /validate  input gate only (no model load)
#   GET  /health    model version + status
#   GET  /metrics   request count + p50/p95/p99 latency since startup
# Auth + rate limiting via env (APEX_API_KEYS, APEX_RATE_LIMIT); see .env.example.

make benchmark      # latency/throughput table -> docs/serving/benchmark.md
make digitize-eval  # paper-ECG digitization fidelity -> docs/digitization/report.md

# example: analyze an uploaded signal file
curl -F file=@ecg.npy -F sampling_rate=100 http://localhost:8000/analyze
# upload a *photo of a paper ECG* and get the same report back
curl -F file=@paper_ecg_photo.jpg http://localhost:8000/analyze
```

Run the clinical dashboard (Phase 11):

```bash
make ui   # or: python app.py   — Gradio at http://localhost:7860
# upload panel (signal file or paper-ECG photo), ECG + per-finding grounding overlays,
# structured report with confidence bars + flags, red/yellow/green severity banner.
# Deploy to Hugging Face Spaces: see docs/frontend/deploy.md
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
- **Phase 9:** FastAPI service — `POST /analyze` (signal file or JSON body → the
  Phase-8 report), `POST /validate`, `GET /health` (model version + status),
  `GET /metrics` (request count + p50/p95/p99 latency since startup). API-key auth + a
  fixed-window rate limiter (`src/serving/security.py`), the detector cached/warmed at
  startup so warm requests skip the checkpoint load. **Benchmarked** on CPU + Apple MPS:
  pipeline p50 **~6 ms** (168 req/s), through the HTTP stack **~12 ms** (80 req/s);
  grounding adds ~64 ms ([`docs/serving/benchmark.md`](docs/serving/benchmark.md)). ✅
- **Phase 10:** ECG image digitization — upload a *photo of a paper ECG* and get the
  full report back. `src/digitization/` renders signals to realistic paper-ECG images
  (grid + 12 labelled leads) and inverts them with a classical CV pipeline (grid-pitch
  detection → adaptive trace extraction → mm/mV calibration → resample); `POST /analyze`
  now accepts images (was 415). Round-trip fidelity on 100 PTB-XL records: **0.89 mean
  per-lead correlation** on clean renders, degrading gracefully to 0.82 / 0.71 under
  simulated phone-photo blur+noise+JPEG. Classical (no paired real-photo dataset exists
  to train on); a learned segmentation model is the documented upgrade for real photos
  ([`docs/digitization/report.md`](docs/digitization/report.md)). ✅
- **Phase 11:** clinical dashboard — a Gradio front end (`app/frontend/`, entry `app.py`)
  over the full pipeline: upload panel (signal file **or** paper-ECG photo), the 12-lead
  waveform with per-finding **grounding overlays** (one colour per finding, click a
  finding to highlight its saliency), a report panel with confidence bars + reliability
  flags + full explanation, a red/yellow/green **severity banner**
  (`src/serving/severity.py`: red on ST-elevation / injury), and the fixed
  decision-support disclaimer. Packaged for **Hugging Face Spaces**
  ([`docs/frontend/deploy.md`](docs/frontend/deploy.md)); the 12 KB SCP label dictionary
  is now bundled so the demo runs without the full waveform download. ✅
- **Phase 12:** evaluation vs published baselines — APEX on the PTB-XL **test split**
  vs the landmark PTB-XL benchmark (Strodthoff et al. 2021) + a **GPT-4o zero-shot**
  ECG-image baseline. On the 71-code task APEX matches `resnet1d_wang` (0.920 vs 0.919);
  a generalist LLM reading the ECG *image* sits at ~41% multiclass (published) — the gap
  is what domain-specific training buys. Eval notebook
  [`notebooks/03_baseline_comparison.ipynb`](notebooks/03_baseline_comparison.ipynb),
  tables in [`docs/model_comparison/`](docs/model_comparison/baseline_comparison.md). ✅
- **Phase 13:** adversarial & edge-case testing — five curated hard-case cohorts from the
  test split (significant artifact, borderline-confidence, rare labels, multi-condition,
  and plain-normal ECGs), measured at the *shipped* surfacing rule rather than a tuned
  optimum. Failure-mode taxonomy (F1–F6), concrete example records run through the full
  pipeline, and deployment guardrails in
  [`docs/edge_cases/report.md`](docs/edge_cases/report.md). Headline: label recall falls
  from 84% overall to **73% on the rarest labels** and 79% under whole-record noise; 32%
  of all misses sit silently in 0.35–0.5; and the model over-flags heavily at threshold
  0.5 — which is a **calibration** problem, not a ranking one. ✅
- **Phase 14:** model card & ethics statement — a full Hugging Face
  [`MODEL_CARD.md`](MODEL_CARD.md) (intended use, out-of-scope populations, known
  limitations, privacy/HIPAA, ethics) backed by a measured **demographic breakdown**
  ([`docs/model_card/demographics.md`](docs/model_card/demographics.md)). The breakdown
  found **real disparities in both dimensions**: macro-AUROC is **0.925 male vs 0.906
  female** (gap +0.019, 95% CI +0.004–+0.033) and declines monotonically with age
  (0.906 at 18–39 → 0.864 at 75+). Both CIs exclude zero. Documented, not corrected. ✅
- **Phase 15:** technical write-up — a full engineering post covering the clinical problem
  (with citations), the architecture, the reliability layer, results vs published
  baselines, the honest failure analysis, and what's next:
  [`docs/writeup/technical_post.md`](docs/writeup/technical_post.md). LinkedIn cross-post
  and publishing notes in [`docs/writeup/linkedin_post.md`](docs/writeup/linkedin_post.md). ✅
- **Phase 16 (stretch):** real-time wearable stream simulation — replays a PTB-XL record
  as a live 12-lead stream into a rolling 10 s window re-analyzed every second, with a
  continuously updating findings panel (`make stream-demo`, plus a **Live monitor** tab in
  the dashboard). Batch metrics don't transfer to a monitor, so findings are only promoted
  once they **persist across windows**: that cuts panel churn **76%**, at a measured cost
  of ~4 points of recall and 3.3 s of detection latency —
  [`docs/streaming/report.md`](docs/streaming/report.md) has the full trade-off curve.
  Inference is 6.6 ms/window, ~150x real-time headroom. ✅
- Phase 17+: calibration (the direct follow-up to Phase 13's over-flagging finding).

## Benchmark comparison (PTB-XL test split)

Macro-AUROC on PTB-XL fold 10 (`make eval-baselines`). Published rows are the landmark
PTB-XL benchmark ([Strodthoff et al. 2021](https://github.com/helme/ecg_ptbxl_benchmarking)).

| model | all-task AUROC | superclass AUROC | source |
|---|---:|---:|---|
| inception1d | 0.925 | 0.921 | Strodthoff 2021 |
| xresnet1d101 | 0.925 | 0.928 | Strodthoff 2021 |
| **APEX (cnn_bce)** | **0.920** | 0.893 | **this work** |
| resnet1d_wang | 0.919 | 0.930 | Strodthoff 2021 |
| lstm | 0.907 | 0.927 | Strodthoff 2021 |
| Wavelet+NN | 0.849 | 0.874 | Strodthoff 2021 |

APEX (compact 1D-CNN, no ensembling) matches `resnet1d_wang` on the 71-code task. Its
superclass column is pooled from the 71-code head, not trained on the 5-class task, so
the all-task column is the like-for-like comparison. A **GPT-4o zero-shot** ECG-image
baseline reaches only ~41% multiclass accuracy ([published](https://ai.jmir.org/2025/1/e74426)),
and its free-text diverges from APEX's clinical template (BLEU-4 ≈ 0.11) —
[`docs/model_comparison/gpt4o_comparison.md`](docs/model_comparison/gpt4o_comparison.md).
_Ribeiro et al. 2020 (the landmark deep-ECG paper) was trained on the CODE dataset, not
PTB-XL, so it has no direct row here._

## Data & ethics

PTB-XL is de-identified and openly licensed, but this project handles medical data
and produces medical-adjacent output. Keep the review gate on, never present outputs
as diagnoses, and keep the "verify before acting" disclaimer visible everywhere.
