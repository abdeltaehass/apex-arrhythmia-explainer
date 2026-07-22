#!/usr/bin/env python3
"""Phase 7 test report: how often each reliability flag fires on the validation set.

    python scripts/run_reliability_report.py                 # full val set + a grounding sample
    python scripts/run_reliability_report.py --ground-n 300   # bigger grounding sample
    python scripts/run_reliability_report.py --limit 200      # quick partial run

For every validation record: run the real detector, threshold at `CFG.review_threshold`
to get the surfaced label set (exactly what `PTBXLDataset`'s val fold and
`src/detection/train.py` use), render the deterministic template report from those
labels + the detector's own confidences (not synthetic — this is the actual model's
calibration), parse it back with `src.generation.parse.asserted_findings`, and run all
four `src.eval.reliability` checks. The consistency/low-confidence/mutual-exclusivity
checks run over the *entire* validation set (cheap — no gradients needed); the
grounding-conflict check runs Grad-CAM backward passes, so it runs over a sampled
subset (mirrors the sample sizes used in the Phase-5 grounding sanity sweep).

Writes docs/reliability/report.md + report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import CFG, ROOT  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.detection.dataset import PTBXLDataset  # noqa: E402
from src.eval.reliability import check_reliability, summarize  # noqa: E402
from src.generation.parse import asserted_findings  # noqa: E402
from src.generation.prompts import target_text  # noqa: E402
from src.generation.templater import build_structured_input, render_report  # noqa: E402
from src.generation.vocab import leads_for  # noqa: E402
from src.grounding import ground, load_detector  # noqa: E402

OUT_DIR = ROOT / "docs" / "reliability"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "final_best.pt"


def surfaced_and_confidences(y_prob: np.ndarray, label_space: list[str], threshold: float):
    above = [j for j in range(len(label_space)) if y_prob[j] >= threshold]
    surfaced = {label_space[j] for j in above}
    confidences = {label_space[j]: float(y_prob[j]) for j in above}
    return surfaced, confidences


def build_report_for_record(record_id: str, surfaced: set[str], confidences: dict[str, float],
                            saliency_by_code: dict | None = None) -> tuple:
    si = build_structured_input(sorted(surfaced), confidences=confidences,
                               review_threshold=CFG.review_threshold)
    rep = render_report(si)
    text = target_text(rep["findings"], rep["impression"])
    asserted = asserted_findings(text)
    leads_by_code = {c: leads_for(c) for c in surfaced if leads_for(c)} if saliency_by_code else None
    report = check_reliability(
        record_id, asserted, surfaced, confidences,
        leads_by_code=leads_by_code, saliency_by_code=saliency_by_code,
    )
    return report, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--limit", type=int, default=None, help="cap validation records (debugging)")
    ap.add_argument("--ground-n", type=int, default=200,
                    help="records sampled for the grounding-conflict check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, label_space, ck_args = load_detector(args.checkpoint, device=args.device)
    print(f"loaded {ck_args.get('model') or 'cnn'} checkpoint")

    X, Y = build_split_cache("val", CFG.sampling_rate)
    ecg_ids = PTBXLDataset("val").df["ecg_id"].tolist()
    assert len(ecg_ids) == len(X), "val cache / dataset order mismatch"
    if args.limit:
        X, ecg_ids = X[:args.limit], ecg_ids[:args.limit]
    n = len(X)
    print(f"val set: {n} records")

    model.eval()
    probs = np.empty((n, len(label_space)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, 256):
            batch = torch.from_numpy(X[start:start + 256]).to(args.device)
            probs[start:start + 256] = torch.sigmoid(model(batch)).cpu().numpy()

    reports, skipped_empty = [], 0
    for i, ecg_id in enumerate(ecg_ids):
        surfaced, confidences = surfaced_and_confidences(probs[i], label_space, CFG.review_threshold)
        if not surfaced:
            skipped_empty += 1
            continue
        report, _ = build_report_for_record(str(ecg_id), surfaced, confidences)
        reports.append(report)
    print(f"reliability checks: {len(reports)} records ({skipped_empty} had no label above "
         f"threshold {CFG.review_threshold}, skipped)")

    # Grounding-conflict sample: records with >=1 surfaced, lead-localizing finding.
    localizing = [i for i in range(n)
                 if any(leads_for(label_space[j]) for j in range(len(label_space))
                        if probs[i, j] >= CFG.review_threshold)]
    rng = np.random.default_rng(args.seed)
    ground_idx = rng.choice(localizing, size=min(args.ground_n, len(localizing)), replace=False)
    print(f"grounding-conflict sample: {len(ground_idx)} / {len(localizing)} eligible records")

    ground_reports, n_citations_checked = [], 0
    for k, i in enumerate(ground_idx):
        if k % 50 == 0:
            print(f"  grounding {k}/{len(ground_idx)}", flush=True)
        surfaced, confidences = surfaced_and_confidences(probs[i], label_space, CFG.review_threshold)
        territory_codes = [c for c in surfaced if leads_for(c)]
        if not territory_codes:
            continue
        n_citations_checked += sum(len(leads_for(c)) for c in territory_codes)
        saliency_by_code = ground(model, X[i], territory_codes, label_space=label_space, fs=CFG.sampling_rate)
        report, _ = build_report_for_record(str(ecg_ids[i]), surfaced, confidences, saliency_by_code)
        ground_reports.append(report)

    full_summary = summarize(reports)
    ground_summary = summarize(ground_reports)
    # the full-set summary's grounding numbers are meaningless (no saliency was computed
    # there); splice in the numbers from the dedicated grounding sample instead. Report
    # BOTH the per-record rate ("did this record have >=1 conflicting citation") and the
    # per-citation rate (conflicts / individual lead-citations checked) -- a record with
    # several cited leads compounds even a modest per-citation rate into a much higher
    # per-record one, so the per-citation number is the fairer "how often does this
    # actually happen" figure.
    full_summary["grounding_conflict_rate_per_record"] = ground_summary.get("grounding_conflict_rate")
    full_summary["n_grounding_conflicts"] = ground_summary.get("n_grounding_conflicts")
    full_summary["n_lead_citations_checked"] = n_citations_checked
    full_summary["grounding_conflict_rate_per_citation"] = (
        ground_summary.get("n_grounding_conflicts", 0) / n_citations_checked if n_citations_checked else None
    )
    full_summary["grounding_sample_n"] = ground_summary.get("n_total", 0)
    full_summary.pop("grounding_conflict_rate", None)

    # Tally which specific codes/pairs drive each flag type (for the write-up).
    from collections import Counter

    warning_codes = Counter(w.code for r in reports for w in r.consistency_warnings)
    low_conf_codes = Counter(f.code for r in reports for f in r.low_confidence)
    mutex_pairs = Counter(
        tuple(sorted((c.code_a, c.code_b))) for r in reports for c in r.mutual_exclusivity
    )
    ground_conflict_codes = Counter(c.code for r in ground_reports for c in r.grounding_conflicts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint), "n_val": n, "n_checked": len(reports),
        "n_skipped_empty": skipped_empty, "review_threshold": CFG.review_threshold,
        "low_confidence_threshold": CFG.low_confidence_threshold,
        "summary": full_summary,
        "top_consistency_warning_codes": warning_codes.most_common(10),
        "top_low_confidence_codes": low_conf_codes.most_common(10),
        "top_mutual_exclusivity_pairs": mutex_pairs.most_common(10),
        "top_grounding_conflict_codes": ground_conflict_codes.most_common(10),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))

    print("\n=== summary ===")
    for k, v in full_summary.items():
        print(f"  {k}: {v}")
    print(f"\n-> {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
