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
  detection/          1D CNN / transformer model + dataset + knowledge distillation
  generation/         LLM prompting / fine-tuning + inference
  grounding/          attention / saliency explainability layer
  eval/               metrics, consistency checker, hallucination flagging, reliability checks
  serving/            structured JSON output schema + serializer + input validation
  digitization/       paper-ECG image <-> signal (render + digitize)
  federated/          FedAvg simulation over per-hospital data shards
  rag/                clinical reference corpus + vector index + retrieval
  longitudinal/       serial (prior vs current) ECG comparison + change reports
  ehr/                EHR integration: one-line impression, ICD-10-CM, HL7 FHIR R4
  feedback/           clinician feedback store + per-label threshold re-tuning
  synthesis/          diffusion ECG generation + rare-class augmentation ablation
  i18n/               Spanish clinical explanation + language-aware consistency gate
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

Retrieval-augmented generation — ground explanations in clinical reference text (Phase 21):

```bash
make rag-index    # fetch the openly-licensed corpus + build the vector index
make rag-eval     # paired RAG on/off hallucination comparison -> docs/rag/report.md

# rebuild the index without re-fetching (no network needed):
python scripts/build_rag_index.py --no-fetch
```

```python
from src.rag import load_index, retrieve_for_findings, format_context
ctx = retrieve_for_findings(structured_input, load_index(), k_per_finding=2)
report = analyze_signal(signal, 100)          # unchanged; RAG is opt-in per call
```

The corpus is **Wikipedia (CC BY-SA 4.0) + PTB-XL's SCP statement definitions (CC BY 4.0)**,
verbatim and with per-passage provenance — *not* ACC/AHA guidelines, which are copyrighted
and not redistributable. See [`data/reference/NOTICE.md`](data/reference/NOTICE.md).

Federated learning — train across simulated hospitals without pooling data (Phase 20):

```bash
make federated-smoke  # tiny end-to-end check
make federated        # FedAvg sweep + IID control + local-only baselines -> docs/federated/
make federated-report # rebuild the comparison from existing runs

# or one configuration directly:
python -m src.federated.train --by device --rounds 20 --local-epochs 5
python -m src.federated.train --by iid --rounds 20 --local-epochs 5   # the control
```

Distillation — compress the detector into a lightweight student (Phase 19):

```bash
make distill-smoke   # tiny end-to-end check on the sample records
make distill         # 3 student sizes x {distilled, from-scratch control} -> docs/distillation/
make distill-report  # rebuild the trade-off table from existing checkpoints

# or one student directly:
python -m src.detection.distill --width 16 --blocks 1 --alpha 0.7 --temperature 2.0
```

The student is a drop-in — same architecture family, same checkpoint schema — so serving,
Grad-CAM grounding and the dashboard take it with no code change:

```python
from src.serving import analyze_signal
report = analyze_signal(signal, 100, checkpoint="outputs/student_w16b1_kd_best.pt")
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
#   POST /analyze/ehr  same input -> impression + ICD-10-CM + FHIR R4 bundle (Phase 23)
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

Compare two ECGs from the same patient (Phase 22):

```bash
make longitudinal            # fit change thresholds + full eval -> docs/longitudinal/
make longitudinal-examples   # rebuild the 12 cardiologist-graded worked examples
```

```python
from src.longitudinal import compare_records

result = compare_records(prior_id=16404, current_id=16408)
print(result.report.text)
# Compared with the prior study of 1996-07-16 (53 minutes earlier): Atrial fibrillation
# has reverted to sinus rhythm. New since the prior study: first-degree AV block. ...
# Not compared — PR interval not compared: no P wave detected in the prior study
# (atrial fibrillation) — PR interval is undefined.

result.delta.significant_intervals()   # structured, noise-gated interval changes
result.consistency.consistent          # the change narrative asserts nothing unmeasured
```

Export a report for an EHR (Phase 23):

```bash
make ehr-examples        # validated FHIR bundles across 7 categories -> docs/ehr/
make verify-terminology  # re-check every ICD-10-CM / LOINC code against the live NLM service
```

```python
from src.ehr import to_ehr_export

export = to_ehr_export(report, record_identifier="ptbxl-00123")
export.impression      # 'Atrial fibrillation at 112 bpm with ST-segment depression; ...
                       #  computer-assisted interpretation, requires physician confirmation.'
export.icd10_codes()   # ['I48.91', 'R94.31']
export.fhir_bundle     # HL7 FHIR R4 Bundle (DiagnosticReport + Observations + Device)
export.valid           # False if the bundle failed schema *or* binding validation
```

```bash
# or over HTTP — same input contract as /analyze
curl -F file=@ecg.npy -F patient_reference=Patient/1234 http://localhost:8000/analyze/ehr
```

Collect and use clinician feedback (Phase 24):

```bash
make feedback-sim   # online-loop experiment (4 arms) -> docs/feedback/
make ui             # the dashboard now carries a "Review this report" panel
```

```python
from src.feedback import FeedbackStore, RatedFinding, ThresholdSet, update_thresholds

with FeedbackStore() as store:                       # outputs/feedback.db
    store.log_review([RatedFinding("AFIB", 0.93, "correct")],
                     reviewer_id="dr_a", missed=["1AVB"])
    update_thresholds(store).save()                  # per-label, with guardrails

from src.serving import analyze_signal
analyze_signal(signal, 100, thresholds=ThresholdSet.load())
```

Generate the explanation in Spanish (Phase 27):

```bash
make i18n-eval   # gate parity + terminology validation -> docs/i18n/
```

```python
from src.serving import analyze_signal
report = analyze_signal(signal, 100, backend="template", lang="es")
# Hallazgos: Ritmo irregularmente irregular sin ondas P identificables.
# Impresión: Hallazgos compatibles con fibrilación auricular.

report.consistency.consistent   # the Phase-7 gate, now language-aware
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
- **Phase 17:** multi-label confidence calibration — post-hoc calibrators fitted on the
  validation fold, with before/after reliability diagrams
  ([`docs/calibration/report.md`](docs/calibration/report.md), `make calibrate`).
  **ECE 0.079 → 0.002 (97% reduction)** and spurious surfaced labels **5.09 → 0.35 per
  record**, with macro-AUROC unchanged at 0.920. Two honest findings: plain temperature
  scaling made ECE *worse* (the error was a bias, not sharpness — it needed a per-label
  intercept), and the **"ECE ≈ 0.90" quoted in Phases 3 and 12–16 was wrong**, produced by
  a multi-class metric applied to independent sigmoids. Both documented. ✅
- **Phase 18:** per-label demographic subgroup analysis (`make subgroups`) — AUROC broken
  out by sex and age bracket for every label, with a ≥10-positive power floor and
  Benjamini-Hochberg FDR correction. The sex effect is **diffuse** (34 labels testable,
  none significant individually, yet the macro gap is real); the age effect is **localized
  and consistent** — all 9 significant pathology labels are worse in the 60+ group. Tables
  in [`docs/model_card/subgroup_performance.md`](docs/model_card/subgroup_performance.md),
  discussion in the model card's *Fairness and equity* section. ✅
- **Phase 19:** knowledge distillation to a lightweight student (`make distill`) — the
  8.8M-parameter detector compressed into a **254k-parameter student trained on the
  teacher's soft per-label probabilities**, not just the 0/1 labels.
  **35x smaller (35.2 MB → 1.1 MB), ~5x faster per forward pass (81% lower latency),
  for 0.28% macro-AUROC degradation** (0.920 → 0.917); end-to-end through the API the
  reduction is 57%, since preprocessing and report generation are untouched. Every student
  was also trained **from scratch as a control**, which is what makes the result a
  measurement rather than an assertion: distillation is worth **+1.45 points** of AUROC at
  254k parameters and only +0.10 at 988k — the gain is real, and largest where capacity is
  scarcest. A 988k student *beats* the teacher outright, so the shipped model was
  over-parameterized for this task.
  [`docs/distillation/report.md`](docs/distillation/report.md). ✅
- **Phase 20:** federated learning simulation (`make federated`) — PTB-XL's `device`
  column splits the training folds into **9 simulated hospitals** that never pool data,
  trained with **FedAvg**. The split is genuinely non-IID: NORM prevalence runs 29%->82%
  across clients, sizes 6,018->151 (40:1), and **47 of 71 labels are entirely absent from
  at least one client**. **FedAvg reaches test macro-AUROC 0.8751 vs the centralized
  0.9199 — a 4.48-point (4.9%) gap with no ECG leaving its hospital**, and it beats the
  best single hospital training alone (0.8268) by 4.8 points, closing **52% of the
  distance** to full pooling. An **IID control** with identical client sizes splits the
  gap: **1.82 points are heterogeneity, the rest is federation's own mechanics** (weight
  averaging + per-round optimizer restart). Two honest negatives: more local epochs per
  round *helped* (the optimizer restart binds harder than client drift), and swapping
  BatchNorm for GroupNorm — the textbook fix for averaged BN statistics — made it **5.9
  points worse**. [`docs/federated/report.md`](docs/federated/report.md). ✅
- **Phase 21:** retrieval-augmented clinical context (`make rag-index` / `make rag-eval`)
  — a hybrid dense+sparse vector index over **927 passages** of openly-licensed cardiology
  reference text, retrieved per detected finding and injected into the generator's prompt.
  **The honest result is a negative one: RAG *doubled* the hallucination rate** (0.060 →
  0.120 over 150 paired records, 11 → 22 fabricated findings; McNemar p=0.064), degraded
  format compliance (0.70 → 0.54) and 2.5x'd the rate of treatment recommendations the
  prompt forbids. The mechanism is specific to this task: APEX's generator may assert
  *only* the detector's findings, and retrieved cardiology text is full of other condition
  names — **`LVH` was fabricated 8 times with RAG, every one of them with LVH named in that
  record's retrieved passages**, versus 3 times without. Retrieval did improve finding
  coverage (+7 points), so it belongs on the *wording*, not the assertions; `with_rag` is
  off by default and Phase 7's consistency gate catches every fabrication before it reaches
  a clinician. [`docs/rag/report.md`](docs/rag/report.md).
  **Note:** ACC/AHA guidelines are copyrighted and *not* redistributable, so the corpus is
  Wikipedia (CC BY-SA 4.0) + PTB-XL SCP definitions (CC BY 4.0), verbatim and fully
  attributed — see [`data/reference/NOTICE.md`](data/reference/NOTICE.md). ✅
- **Phase 22:** longitudinal (serial) ECG comparison (`make longitudinal`) — compares two
  recordings from the same patient and reports what *changed*. PTB-XL's 2,111 repeat
  patients give **2,930 prior→current pairs** (294 held out). Intervals are measured from
  the waveform (PTB-XL ships none): median beat → 500 Hz spline → global delineation, giving
  PR/QRS/QT/QTc plus per-lead ST levels, validated against labels the delineator never sees
  (**QRS→LBBB AUROC 0.92, QRS→RBBB 0.93, PR→1AVB 0.84**). **The headline is the cost of
  differencing: the same detector on the same records scores F1 0.64 at "what is present"
  and F1 0.32 at "what is new"** — two noisy decisions differenced are noisier than either.
  Every change must clear a **measured** noise floor, and getting that floor was the
  interesting part: the obvious null cohort (327 same-day pairs) turned out to be *not* a
  null — its spread is **larger** than that of year-apart pairs (QRS SD 40.8 vs 27.1 ms),
  because you only get two ECGs in a day if something is acutely wrong (63% vs 34% carry an
  acute code). The floor is instead bracketed by within-record split-half repeatability and
  label-stable between-session pairs, fitted on folds 1–8. Two corrections that matter:
  the per-lead ST bar is **Bonferroni-widened** (8 leads at 95% would invent a regional ST
  change in **34%** of unchanged tracings), and raw QT/Bazett are suppressed when the rate
  moved (Bazett repeats to 34.1 ms between sessions vs Fridericia's 27.3, but the two are
  identical within a recording — the excess is purely rate-induced). 110 PTB-XL reports
  contain the reading cardiologist's own comparison sentence; for **12 the prior tracing is
  recoverable**, so the worked examples are graded against a physician rather than against
  me — 6/12 concordant on the principal change, with failures clustering on *stable* studies.
  [`docs/longitudinal/report.md`](docs/longitudinal/report.md) ·
  [`examples.md`](docs/longitudinal/examples.md). ✅
- **Phase 23:** EHR integration layer (`make ehr-examples`) — turns a report into the three
  things a hospital system can consume: a **single pasteable sentence**, **ICD-10-CM code
  suggestions**, and a validated **HL7 FHIR R4 `Bundle`**; served at `POST /analyze/ehr`.
  The work is not plumbing but deciding what APEX may *claim* once its output carries
  financial weight. **43 of 71 findings are not billable diagnoses at all** and map to
  **R94.31** (*Abnormal electrocardiogram*): an ECG suggests infarction but the Fourth
  Universal Definition requires troponin, and ECG voltage criteria for hypertrophy are not
  an anatomical diagnosis. The 25 findings the ECG genuinely *establishes* (AV block, AF,
  bundle branch block) get their specific code; the rest carry the specific code a
  clinician might reach for as a `candidate` alongside the evidence it would need — visible,
  never auto-suggested. **The brief's own example is a trap worth naming:** I48.0 is
  *paroxysmal* AF, and paroxysmal means terminating within seven days — invisible in ten
  seconds of signal, so the only honest code is **I48.91, unspecified**; anything else is
  upcoding, and that rule is an executable test. All **27 ICD-10-CM and 7 LOINC codes are
  verified against the live NLM Clinical Tables service** for existence, exact wording, and
  **billability** — the check that catches FY2024 subdividing I47.1 into I47.10/.11/.19 and
  quietly turning a billable code into a header. Bundles validate against the published R4B
  StructureDefinitions, but probing that validator with broken bundles showed it accepts
  `"status": "definitely-final"` and a Device on `DiagnosticReport.performer`, so a
  binding/reference-target checker was added to cover what the schema cannot see.
  7 validated examples across PTB-XL's 5 diagnostic superclasses plus AF and paced rhythm —
  a normal ECG correctly emits **no code at all**.
  [`docs/ehr/report.md`](docs/ehr/report.md) · [`examples.md`](docs/ehr/examples.md). ✅
- **Phase 24:** human-in-the-loop feedback (`make feedback-sim`) — a review panel in the
  dashboard (rate every finding **correct/incorrect/uncertain**, report what was *missed*),
  a local SQLite store, and a policy that turns ratings into **per-label decision
  thresholds** applied via `analyze_signal(..., thresholds=...)`. Then the part that
  matters: measuring whether the loop works, with a reviewer simulated from PTB-XL labels,
  feedback streamed from the validation fold and every score on the test fold. **It does
  not, in its obvious form.** A reviewer can only rate what was *shown*, so the data
  contains false positives and no false negatives — it can justify raising a threshold and
  never lowering one. Ratings alone moved **9 thresholds up and 0 down** and cost 2.1% of
  macro-F1. Letting clinicians report missed findings — the intuitive fix — **does not
  help**: 1,647 reported misses, still 10 up and 0 down, performance identical to four
  decimals, because knowing a true positive sits below the threshold says nothing about how
  many false positives sit beside it. What works is **exploration**: surfacing ~10% of
  sub-threshold findings for review flips the direction to **1 up, 18 down** and buys
  **+0.049 macro recall for −0.033 precision, +10.3% macro-F1** — and keeps +9.3% with a
  reviewer who is wrong 10% of the time. The experiment also caught a bug in my own policy:
  a "raise a notch when precision is below target" fallback that ratcheted 50 of 71 labels
  forever, since **only 21 can reach precision 0.70 at any threshold** — a threshold cannot
  manufacture precision the ROC curve lacks, so those labels are now flagged
  `target_unreachable` (retrain, don't re-tune) and a test keeps the fallback deleted.
  [`docs/feedback/report.md`](docs/feedback/report.md). ✅
- **Phase 25:** synthetic ECG augmentation for rare classes (`make synth-ablation`) — a
  conditional 1D diffusion model plus a five-arm ablation, and **a negative result**: no
  augmentation method beat the un-augmented baseline. First, the question could not be asked
  as posed — PTB-XL's 17 labels with <50 training examples have **1–5 test positives**, and
  the usual bootstrap *hides* this, reporting them as measured more tightly (CI 0.035) than
  well-supported labels (0.041), because with two positives it can only reshuffle the 2,196
  negatives. Hanley-McNeil, which depends on positive count, puts them **4.0× wider**
  (0.373). So rarity was **induced** on 8 labels with 40–112 test positives, keeping 50 real
  examples each, every arm adding the same 350 rows. Results (macro AUROC, 3 seeds):
  **baseline 0.8960**, oversample 0.8733, classical 0.8810, synthetic 0.8776,
  synthetic+classical 0.8829 — all worse, with plain oversampling worst and the synthetic
  arm the only one with several *significant harms*. The mechanism is measured, not guessed:
  the generator passes every check such papers usually report (memorization ratio 1.08,
  diversity 1.11, 100% delineable) and fails the one that matters — QRS **227 ms vs 112 ms**
  real, P waves in **31% vs 85%**. It learned the low-frequency envelope of an ECG, not the
  sharp deflections that carry the diagnosis, so its samples contradict the definitions of
  the very classes they were meant to teach.
  [`docs/synthesis/report.md`](docs/synthesis/report.md). ✅
- **Phase 27:** multilingual clinical explanation (`make i18n-eval`) — Spanish output via
  `analyze_signal(..., lang="es")`, with all 71 SCP statements hand-authored in clinical
  Spanish (*bloqueo de rama*, not the literal *bloqueo del haz*). **The finding is not the
  translation.** Phase 7's consistency gate — what stops APEX asserting findings the
  detector never surfaced — matched *English* phrases, so a Spanish report parsed as
  asserting nothing and passed **unconditionally**: over 400 records, fabricated findings
  were caught **100% in English and 0% in Spanish**. A Spanish report inventing *bloqueo
  completo de rama izquierda* on a patient with only atrial fibrillation was reported
  consistent. That is what a health-equity failure looks like in code — fluent text, valid
  JSON, disclaimer present, guardrail silently absent for the second-largest language group
  in US healthcare. Now at **100% parity** on both fabrication detection and findings
  round-trip. One renderer serves both languages and English output is **byte-identical** to
  the Phase-6 templater (200/200), with the parser agreeing with Phase 6 on English
  (300/300) — anti-drift enforced by test, not intention. The parity check also caught a bug
  in the *new* code: searching the whole report instead of the Impression section made the
  Spanish gate noisier than English (`ISCAN`'s finding text is verbatim `INVT`'s impression
  term), the same second-class treatment through the opposite door. Terminology checked
  against 26 Spanish cardiology articles: 34/67 confirmed, the rest a clinician review list
  rather than an error list. [`docs/i18n/report.md`](docs/i18n/report.md) ·
  [MODEL_CARD](MODEL_CARD.md#language-access-and-health-equity). ✅
- Phase 26/28+: apply the calibrator in the serving path and re-tune the operating threshold.

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
