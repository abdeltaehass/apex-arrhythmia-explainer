# Phase 8 — Structured JSON output schema

The single response contract APEX returns for one recording, defined in
[`src/serving/schema.py`](../../src/serving/schema.py) (Pydantic v2) and assembled by
[`src/serving/serializer.py`](../../src/serving/serializer.py). Every upstream stage —
detection, generation, grounding, the Phase-7 reliability checks — folds into this one
object. See [`sample_report.json`](sample_report.json) for a clean record and a
heavily-flagged record produced by the real pipeline.

## `APEXReport`

| field | type | meaning |
|---|---|---|
| `findings` | `FindingOut[]` | one entry per detected label (below) |
| `impression` | `string` | the interpretive summary (the Impression section) |
| `explanation` | `string` | the full generated report (Findings + Impression) |
| `consistency` | `ConsistencyOut` | the `src/eval/consistency.py` result |
| `review_recommended` | `bool` | the single review gate for the caller (below) |
| `input_validation` | `InputValidation \| null` | the lead-count / duration gate |
| `disclaimer` | `string` | fixed decision-support disclaimer |
| `schema_version` | `string` | `"1.0"` |

### `FindingOut`

| field | type | meaning |
|---|---|---|
| `label` | `string` | SCP-ECG code, e.g. `"AFIB"` |
| `description` | `string` | human-readable label, e.g. `"atrial fibrillation"` |
| `confidence` | `float` in [0, 1] | detector sigmoid probability |
| `leads` | `string[]` | leads implicated in this finding (empty if the code doesn't localize) |
| `flags` | `Flag[]` | per-finding flag status (below) |
| `needs_review` | `bool` | `confidence < review_threshold` **or** any flag present |

### `Flag`

`{type, message}`, where `type` is one of:

- `low_confidence` — the finding's confidence is below the tunable Phase-7 bar
  (`CFG.low_confidence_threshold`, default 0.7).
- `grounding_conflict` — a lead the finding cites ranks among the least-important for
  that finding in the Phase-5 per-lead saliency.
- `mutual_exclusivity` — this finding fired together with a clinically contradictory
  one (e.g. sinus rhythm + atrial fibrillation); attached to **both** codes involved.
- `unreliable_input` — the recording was flagged unreliable (see input validation);
  attached to the first finding so the reason travels with the finding list.

### `ConsistencyOut`

`{consistent, asserted[], surfaced[], unsupported[]}` — `unsupported` is the set of
findings the explanation text names but the detector never surfaced (hallucinations).
A template-generated explanation is consistent by construction; the field exists to
catch a fine-tuned or API backend that ignores its prompt.

## Input validation

`validate_signal(signal, sampling_rate)` → `InputValidation`
(`{n_leads, duration_s, sampling_rate, ok, reliable, errors[], warnings[]}`):

- **Hard reject** (`ok = false`, `/analyze` returns HTTP 422): not exactly 12 leads
  (the spec's "fewer than 12" plus the symmetric case, since the detector's input conv
  is fixed at 12), a non-positive sampling rate, or structurally broken input (empty,
  leads of unequal length → raises `InputValidationError`).
- **Soft flag** (`reliable = false`, still processed): a recording shorter than
  **5 s** — too short to be reliable. This forces `review_recommended` and adds an
  `unreliable_input` flag rather than refusing the record.

## `review_recommended`

True when **any** of: a finding needs review, the explanation is inconsistent, any
reliability flag fired, or the input was flagged unreliable. This is the one boolean a
caller needs to decide whether to route the record to a human — everything else is the
evidence behind it.

## Usage

```python
from src.serving import analyze_signal, validate_signal

report = analyze_signal(signal_12xT, sampling_rate=100, backend="template", with_grounding=True)
report.model_dump()        # -> the JSON above
report.review_recommended  # -> bool
```

Or via the API ([`app/backend/main.py`](../../app/backend/main.py)): `POST /analyze`
returns the report, `POST /validate` runs only the input gate (no model load).
