#!/usr/bin/env python3
"""Select 20 clinically diverse test-split records for the Phase-6 manual review.

Pulls one (or two, for the highest-volume categories) record per clinical bucket from
the held-out test split — the same split `src/detection/train.py` reports test-fold
AUROC on, so these are records the generator (and the detector) never trained on.
For each, emits the structured input, the deterministic template rendering, and
PTB-XL's own (German) human-validated report text, so `docs/generation/examples_review.md`
can compare a "generated" text against a genuine human reference rather than one
invented for the occasion.

    python scripts/select_review_examples.py > docs/generation/review_candidates.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR  # noqa: E402
from src.generation.dataset import example_to_structured_input  # noqa: E402
from src.generation.prompts import serialize_structured_input  # noqa: E402
from src.generation.templater import render_report  # noqa: E402

TEST_FILE = PROCESSED_DIR / "generation" / "test.jsonl"
SEED = 7


def has(row, *codes):
    return any(c in row["codes"] for c in codes)


BUCKETS: list[tuple[str, callable, int]] = [
    ("normal", lambda r: set(r["codes"]) <= {"NORM", "SR"} and "NORM" in r["codes"], 2),
    ("afib", lambda r: has(r, "AFIB"), 2),
    ("aflt", lambda r: has(r, "AFLT"), 1),
    ("sttc_anterior", lambda r: has(r, "ISCAN", "ISCAS", "ISCAL", "NDT", "NST_"), 1),
    ("sttc_inferior", lambda r: has(r, "ISCIN", "ISCIL"), 1),
    ("mi_inferior", lambda r: has(r, "IMI", "ILMI", "IPMI", "IPLMI"), 2),
    ("mi_anterior", lambda r: has(r, "AMI", "ASMI", "ALMI"), 2),
    ("mi_lateral", lambda r: has(r, "LMI"), 1),
    ("crbbb", lambda r: has(r, "CRBBB"), 1),
    ("clbbb", lambda r: has(r, "CLBBB"), 1),
    ("av_block", lambda r: has(r, "1AVB", "2AVB", "3AVB"), 1),
    ("lvh", lambda r: has(r, "LVH"), 1),
    ("ectopy", lambda r: has(r, "PVC", "PAC"), 1),
    ("pacemaker", lambda r: has(r, "PACE"), 1),
    ("wpw", lambda r: has(r, "WPW"), 1),
    ("bradycardia", lambda r: has(r, "SBRAD"), 1),
]


def main() -> None:
    import numpy as np

    rows = [json.loads(line) for line in TEST_FILE.open()]
    rng = np.random.default_rng(SEED)
    used_ids: set[int] = set()
    out = []

    for name, pred, k in BUCKETS:
        candidates = [
            r for r in rows
            if pred(r) and r["ecg_id"] not in used_ids
            and 1 <= len(r["codes"]) <= 5              # keep illustrative, not overloaded
            and r.get("original_report") and len(r["original_report"].strip()) > 8
        ]
        if not candidates:
            print(f"# WARNING: no candidates for bucket {name!r}", file=sys.stderr)
            continue
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        for i in idx:
            row = candidates[i]
            used_ids.add(row["ecg_id"])
            si = example_to_structured_input(row)
            rep = render_report(si)
            out.append({
                "bucket": name,
                "ecg_id": row["ecg_id"],
                "codes": row["codes"],
                "confidences": row["confidences"],
                "structured_input": serialize_structured_input(si),
                "template_findings": rep["findings"],
                "template_impression": rep["impression"],
                "original_report_de": row["original_report"],
            })

    print(json.dumps(out, indent=2))
    print(f"# selected {len(out)} examples across {len(BUCKETS)} buckets", file=sys.stderr)


if __name__ == "__main__":
    main()
