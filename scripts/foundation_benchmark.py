#!/usr/bin/env python3
"""Phase 28 — APEX against general-purpose models, on every axis that matters.

Runs each system over the same PTB-XL test records and scores discrimination, coverage,
self-consistency, latency, and cost.

**On what this can and cannot run.** The hosted arms (`gpt-4o`, `claude`) need an API key.
Without one they are skipped and recorded as *not run* rather than quietly omitted — a
comparison table with an unexplained empty column is worse than one that says why. Published
GPT-4o ECG figures are carried alongside as a cited anchor, clearly marked as cited rather
than measured.

    python scripts/foundation_benchmark.py --systems apex,apex-student
    python scripts/foundation_benchmark.py --systems local-llm --n 150
    OPENAI_API_KEY=... python scripts/foundation_benchmark.py --systems gpt-4o --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import (  # noqa: E402
    LOCAL_HOURLY_USD,
    PRICING_AS_OF,
    SUPERCLASSES,
    available_systems,
    build_system,
    summarize,
)
from src.config import PTBXL_DIR, ROOT  # noqa: E402

OUT = ROOT / "docs" / "benchmark"

# Cited, not measured. An independent evaluation of GPT-4o reading 12-lead ECG images.
PUBLISHED_ANCHOR = {
    "system": "GPT-4o (published, ECG images)",
    "multiclass_accuracy": 0.41,
    "binary_normal_abnormal": 0.53,
    "source": "Zaboli et al., JMIR AI 2025 (ai.jmir.org/2025/1/e74426)",
    "provenance": "cited",
}


def load_test(n: int | None, seed: int = 0):
    """Raw test-fold signals plus superclass ground truth."""
    import wfdb

    from src.data.labels import diagnostic_superclass_map, load_database, load_scp_statements

    db = load_database()
    test = db[db["strat_fold"] == 10]
    if n:
        test = test.sample(min(n, len(test)), random_state=seed)

    smap = diagnostic_superclass_map(load_scp_statements())
    y = np.zeros((len(test), len(SUPERCLASSES)), dtype=int)
    signals = []
    for i, (_, row) in enumerate(test.iterrows()):
        supers = {smap[c] for c in row["scp_codes"] if c in smap}
        for j, s in enumerate(SUPERCLASSES):
            y[i, j] = int(s in supers)
        signals.append(np.asarray(wfdb.rdsamp(str(PTBXL_DIR / row["filename_lr"]))[0],
                                  dtype=np.float32).T)
    return signals, y, [int(e) for e in test.index]


def key_available(key: str) -> tuple[bool, str]:
    if key == "gpt-4o":
        return bool(os.environ.get("OPENAI_API_KEY")), "OPENAI_API_KEY not set"
    if key == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY")), "ANTHROPIC_API_KEY not set"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="apex,apex-student,local-llm,gpt-4o,claude")
    ap.add_argument("--n", type=int, default=None, help="test records (default: all 2198)")
    ap.add_argument("--llm-n", type=int, default=150,
                    help="records for LLM arms, which are orders of magnitude slower")
    ap.add_argument("--hourly-usd", type=float, default=LOCAL_HOURLY_USD)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    keys = [k.strip() for k in args.systems.split(",") if k.strip()]
    unknown = [k for k in keys if k not in available_systems()]
    if unknown:
        print(f"unknown system(s): {unknown}; available: {sorted(available_systems())}")
        return 2

    print(f"loading test records (n={args.n or 'all'})...")
    signals, y, ecg_ids = load_test(args.n)
    print(f"  {len(signals)} records, superclass prevalence "
          f"{ {s: int(y[:, j].sum()) for j, s in enumerate(SUPERCLASSES)} }")

    results: dict = {"config": {"n_records": len(signals), "llm_n": args.llm_n,
                                "hourly_usd": args.hourly_usd,
                                "pricing_as_of": PRICING_AS_OF,
                                "device": args.device},
                     "systems": {}, "skipped": {}, "cited": [PUBLISHED_ANCHOR]}

    for key in keys:
        ok, why = key_available(key)
        if not ok:
            print(f"\n{key}: SKIPPED — {why}")
            results["skipped"][key] = {"reason": why,
                                       "how_to_run": f"{why.split()[0]}=... make benchmark-foundation"}
            continue

        system = build_system(key, device=args.device)
        # LLM arms are ~300x slower per record; cap them and say so rather than
        # silently running a different n and comparing the columns as if equal.
        limit = args.llm_n if system.kind == "generalist" else len(signals)
        subset = signals[:limit]
        subset_y = y[:limit]
        print(f"\n{system.name}: {len(subset)} records...", flush=True)

        outputs = []
        started = time.perf_counter()
        for i, sig in enumerate(subset):
            outputs.append(system.predict(sig, 100))
            if (i + 1) % max(1, len(subset) // 5) == 0:
                print(f"    {i + 1}/{len(subset)}  "
                      f"({time.perf_counter() - started:.0f}s elapsed)", flush=True)

        scores = summarize(system, outputs, subset_y, hourly_usd=args.hourly_usd)
        results["systems"][key] = {**scores.as_dict(), "describe": system.describe(),
                                   "n_scored": len(subset)}
        # Persist per-record outputs so CIs, examples, and re-scoring do not need a re-run.
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUT / "raw" / f"{key}.json").write_text(json.dumps(
            [{"ecg_id": ecg_ids[i], "scores": o.scores, "explanation": o.explanation,
              "latency_s": o.latency_s, "tokens_in": o.tokens_in,
              "tokens_out": o.tokens_out, "error": o.error, "raw": o.raw[:2000]}
             for i, o in enumerate(outputs)], indent=2, default=str))
        print(f"  macro AUROC {scores.macro_auroc:.4f}  coverage {scores.coverage:.1%}  "
              f"p50 {scores.latency_p50 * 1000:.0f}ms  p95 {scores.latency_p95 * 1000:.0f}ms  "
              f"self-contradiction {scores.self_contradiction:.1%} "
              f"(n={scores.self_contradiction_n})  "
              f"${scores.usd_per_1k:.3f}/1k")
        if scores.n_errors:
            print(f"  errors: {scores.n_errors}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {(OUT / 'benchmark.json').relative_to(ROOT)}")

    if results["skipped"]:
        print("\nNot run (no credentials):")
        for key, info in results["skipped"].items():
            print(f"  {key}: {info['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
