#!/usr/bin/env python3
"""Phase 23 — generate validated FHIR examples across the diagnostic categories.

The deliverable asks for FHIR output for at least five diagnostic categories. PTB-XL
supplies exactly five diagnostic superclasses — NORM, MI, STTC, CD, HYP — so those are the
five, plus two rhythm cases that the superclasses do not reach: atrial fibrillation, which
is the whole ICD-10 specificity argument in one record, and a paced rhythm, which produces
a status code (Z95.0) rather than a diagnosis.

Everything is end-to-end from real recordings: records come from the **held-out test fold**,
run through the actual detector, measured by the Phase-22 interval code, and rendered. No
bundle here was hand-written, which is the point — a hand-written example proves nothing
about what the system emits.

**Selection rule, stated plainly.** A record is chosen for a category when the *detector*
surfaces that category, searched in ecg_id order over records PTB-XL annotates for it. The
alternative — select purely on the annotation — produced two examples that did not exercise
their own mapping: record 116 is annotated NDT but the detector saw only sinus rhythm, and
record 382 is a paced rhythm the detector read as atrial fibrillation, so the Z95.0 path was
never reached. This phase demonstrates the EHR layer; detector accuracy is Phase 12's
subject and is not relitigated here. Each example prints its annotation next to what was
detected, so any disagreement is visible rather than hidden by the selection.

Every bundle is validated (StructureDefinitions + required bindings + reference targets)
and the script exits non-zero if any fails.

    python scripts/ehr_examples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import PTBXL_DIR, ROOT  # noqa: E402
from src.data.labels import diagnostic_superclass_map, load_scp_statements  # noqa: E402
from src.ehr import to_ehr_export  # noqa: E402

OUT_DIR = ROOT / "docs" / "ehr"
BUNDLE_DIR = OUT_DIR / "fhir"

# Category -> (SCP codes that define it, human label). The five PTB-XL diagnostic
# superclasses plus two rhythm cases they do not cover.
CATEGORIES: list[tuple[str, str]] = [
    ("NORM", "Normal ECG"),
    ("MI", "Myocardial infarction"),
    ("STTC", "ST/T change"),
    ("CD", "Conduction disturbance"),
    ("HYP", "Hypertrophy"),
    ("AFIB", "Atrial fibrillation (rhythm)"),
    ("PACE", "Paced rhythm"),
]


def _in_category(codes, key: str, superclass_map: dict[str, str]) -> bool:
    if key in ("AFIB", "PACE"):
        return key in codes
    return {superclass_map[c] for c in codes if c in superclass_map} == {key}


def pick_records(df, superclass_map: dict[str, str], analyse, max_candidates: int = 40
                 ) -> dict[str, tuple[int, object]]:
    """One held-out record per category where the *detector* surfaces that category.

    Candidates are records PTB-XL annotates unambiguously for the category, walked in
    ecg_id order; the first whose detector output also lands in the category wins, so the
    choice stays deterministic. Returns ``{category: (ecg_id, report)}`` — the analysis is
    kept rather than recomputed.
    """
    test = df[df["strat_fold"] == 10]
    chosen: dict[str, tuple[int, object]] = {}
    for key, _ in CATEGORIES:
        candidates = [int(i) for i, row in test.iterrows()
                      if _in_category(set(row["scp_codes"]), key, superclass_map)]
        fallback = None
        for ecg_id in candidates[:max_candidates]:
            report = analyse(ecg_id)
            detected = {f.label for f in report.findings}
            if fallback is None:
                fallback = (ecg_id, report)
            if _in_category(detected, key, superclass_map):
                chosen[key] = (ecg_id, report)
                break
        else:
            if fallback is not None:
                print(f"  {key}: no held-out record where the detector agrees within "
                      f"{max_candidates} candidates — using {fallback[0]} and reporting the "
                      "disagreement")
                chosen[key] = fallback
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="detector checkpoint (default: bundled)")
    args = ap.parse_args()

    import wfdb

    from src.data.labels import load_database
    from src.longitudinal.intervals import measure
    from src.serving.serializer import analyze_signal

    scp = load_scp_statements()
    df = load_database()

    _signals: dict[int, np.ndarray] = {}

    def signal_for(ecg_id: int) -> np.ndarray:
        if ecg_id not in _signals:
            _signals[ecg_id] = np.asarray(
                wfdb.rdsamp(str(PTBXL_DIR / df.loc[ecg_id, "filename_lr"]))[0],
                dtype=np.float32).T
        return _signals[ecg_id]

    def analyse(ecg_id: int):
        return analyze_signal(signal_for(ecg_id), 100, checkpoint=args.checkpoint,
                              backend="template")

    chosen = pick_records(df, diagnostic_superclass_map(scp), analyse)
    print(f"selected {len(chosen)} held-out records: "
          f"{ {k: v[0] for k, v in chosen.items()} }")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale bundles: the record chosen for a category can change when selection or
    # the detector changes, and leaving the old file behind would ship a bundle that
    # nothing regenerates and no index references.
    for old in BUNDLE_DIR.glob("*.json"):
        old.unlink()
    sections: list[str] = []
    index: list[dict] = []
    failures = 0

    for key, label in CATEGORIES:
        picked = chosen.get(key)
        if picked is None:
            print(f"  {key}: no held-out record matched — skipped")
            continue
        ecg_id, report = picked
        row = df.loc[ecg_id]
        intervals = measure(signal_for(ecg_id), 100)
        export = to_ehr_export(report, record_identifier=f"ptbxl-{ecg_id:05d}",
                               intervals=intervals)

        status = "VALID" if export.valid else f"INVALID ({len(export.validation_errors)})"
        print(f"  {key:5s} record {ecg_id:5d}  {len(report.findings)} findings  "
              f"ICD-10 {export.icd10_codes()}  FHIR {status}")
        if not export.valid:
            failures += 1
            for e in export.validation_errors[:5]:
                print(f"         {e}")

        path = BUNDLE_DIR / f"{key.lower()}_{ecg_id}.json"
        path.write_text(json.dumps(export.fhir_bundle, indent=2))

        icd_rows = "\n".join(
            f"| `{m.scp}` | **{m.icd10}** | {m.display} | {m.tier} | "
            f"{('`' + m.candidate + '` — ' + (m.requires or '')) if m.candidate else '—'} |"
            for m in export.icd10) or "| — | — | *no codable finding* | — | — |"

        detected_codes = {f.label for f in report.findings}
        agrees = _in_category(detected_codes, key, diagnostic_superclass_map(scp))
        agreement = "" if agrees else (
            "\n\n> **Note:** the detector did not reproduce the annotated category for this "
            "record. The example is kept as-is — it shows what the EHR layer emits for real "
            "detector output, including when that output is wrong.")

        bundle = export.fhir_bundle
        counts: dict[str, int] = {}
        for entry in bundle["entry"]:
            rt = entry["resource"]["resourceType"]
            counts[rt] = counts.get(rt, 0) + 1
        dr = next(e["resource"] for e in bundle["entry"]
                  if e["resource"]["resourceType"] == "DiagnosticReport")

        sections.append(f"""
### {label} — PTB-XL record {ecg_id}

**Annotated:** `{'`, `'.join(sorted(row['scp_codes']))}` ·
**Detected:** `{'`, `'.join(f.label for f in report.findings) or '(none)'}` ·
**Fold:** 10 (held out){agreement}

**Single-sentence impression** (copy-pasteable):

> {export.impression}

**ICD-10-CM suggestions**

| Finding | Code | Description | Tier | More specific code, and what it would require |
|---|---|---|---|---|
{icd_rows}

**FHIR R4 bundle** — [`fhir/{path.name}`](fhir/{path.name}) ·
`DiagnosticReport.status` = **`{dr['status']}`** ·
resources: {', '.join(f'{v}x {k}' for k, v in sorted(counts.items()))} ·
validation: **{status}**

```json
{json.dumps(dr, indent=2)}
```
""")
        index.append({"category": key, "label": label, "ecg_id": ecg_id,
                      "impression": export.impression, "icd10": export.icd10_codes(),
                      "fhir_valid": export.valid, "bundle": f"fhir/{path.name}",
                      "diagnostic_report_status": dr["status"]})

    n_def = sum(1 for r in index for c in r["icd10"] if c != "R94.31")
    header = f"""# Phase 23 — FHIR examples across {len(index)} diagnostic categories

Generated by `scripts/ehr_examples.py`. Every example is a real PTB-XL record from the
**held-out test fold**, run through the actual detector and the Phase-22 interval
measurement, then rendered to FHIR. Nothing here was hand-written.

All {len(index)} bundles validate against the published HL7 FHIR **R4B** StructureDefinitions
(via `fhir.resources`) *and* against the required ValueSet bindings and reference-target
rules that the schema check alone does not cover — see `src/ehr/fhir.py::check_bindings`.

Full bundles are in [`fhir/`](fhir/); each section below shows the `DiagnosticReport`, which
is the resource a receiving system reads first. Note `DiagnosticReport.status`: it is
`preliminary`, not `final`, whenever APEX has flagged anything for review — `final` asserts
a result fit to act on, and a flagged report is not that.

| Category | Record | ICD-10-CM | Report status | FHIR |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r['label']} | {r['ecg_id']} | {', '.join(f'`{c}`' for c in r['icd10']) or '—'} "
        f"| `{r['diagnostic_report_status']}` | {'valid' if r['fhir_valid'] else 'INVALID'} |"
        for r in index) + f"""

{n_def} of the suggested codes are specific diagnoses; the rest are `R94.31`
(*Abnormal electrocardiogram*), which is the correct code when the ECG shows an
abnormality that some other test has to explain. See
[`report.md`](report.md) for why that split exists.
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "examples.md").write_text(header + "\n" + "\n---\n".join(sections) + "\n")
    (OUT_DIR / "examples.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {(OUT_DIR / 'examples.md').relative_to(ROOT)} "
          f"and {len(index)} bundles in {BUNDLE_DIR.relative_to(ROOT)}")
    if failures:
        print(f"FAILED: {failures} bundle(s) did not validate")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
