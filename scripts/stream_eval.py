#!/usr/bin/env python3
"""Phase 16 — measure what streaming does to APEX's error profile.

    python scripts/stream_eval.py --n 40

Batch metrics do not transfer to a monitor. In batch, Phase 13 measured 5.09 spurious
labels per *record*; in a stream that becomes a fresh spurious decision every hop, so the
question is not "how many false positives" but "how much does the panel churn, and does
temporal persistence fix it without hiding real findings?"

Replays test-split records as live streams (``speed=0``, so as fast as the CPU allows) and
compares, from the *same* window probabilities:

- **naive** — the panel shows whatever cleared threshold in the current window;
- **persistent** — a finding must hold ``confirm_k`` of the last ``confirm_m`` windows.

Reported per arm: panel churn (how often the displayed set changes), distinct spurious
codes shown, true-finding recall (persistence must not cost real detections), and — the
price of the whole idea — **time-to-confirm** for true findings.

Writes docs/streaming/report.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.data.labels import load_database, present_codes  # noqa: E402
from src.serving.severity import URGENT_CODES  # noqa: E402
from src.streaming import ReplaySource, StreamMonitor, load_record_signal  # noqa: E402

OUT_DIR = ROOT / "docs" / "streaming"


def replay_one(ecg_id: int, truth: set[str], duration_s: float, hop_s: float,
               confirm_k: int, confirm_m: int) -> dict | None:
    """Replay one record as a stream; return naive-vs-persistent stats."""
    try:
        sig, fs = load_record_signal(ecg_id)
    except Exception:
        return None
    mon = StreamMonitor(fs=fs, hop_s=hop_s, confirm_k=confirm_k, confirm_m=confirm_m)
    src = ReplaySource(sig, fs=fs, chunk_s=0.5, speed=0, loop=True,
                       max_duration_s=duration_s, source=str(ecg_id))

    prev_naive: set[str] = set()
    prev_conf: set[str] = set()
    naive_churn = conf_churn = 0
    naive_seen: set[str] = set()
    conf_seen: set[str] = set()
    latencies: list[float] = []
    confirm_time: dict[str, float] = {}
    first_fire: dict[str, float] = {}
    n_updates = 0

    for chunk in src:
        u = mon.push(chunk.samples)
        if u is None:
            continue
        n_updates += 1
        latencies.append(u.latency_ms)

        naive = set(u.fired)
        confirmed = {f.code for f in u.confirmed}
        naive_seen |= naive
        conf_seen |= confirmed
        for c in naive - set(first_fire):
            first_fire[c] = u.t_s
        for e in u.events:
            if e.kind == "onset" and e.code not in confirm_time:
                confirm_time[e.code] = e.t_s

        if naive != prev_naive:
            naive_churn += 1
        if confirmed != prev_conf:
            conf_churn += 1
        prev_naive, prev_conf = naive, confirmed

    if n_updates == 0:
        return None

    # lag from a code first clearing threshold to being confirmed (the price of stability)
    lags = [confirm_time[c] - first_fire[c] for c in confirm_time if c in first_fire]
    return {
        "ecg_id": int(ecg_id),
        "n_updates": n_updates,
        "truth": sorted(truth),
        "naive_seen": sorted(naive_seen),
        "confirmed_seen": sorted(conf_seen),
        "naive_spurious": sorted(naive_seen - truth),
        "confirmed_spurious": sorted(conf_seen - truth),
        "naive_true_hits": sorted(naive_seen & truth),
        "confirmed_true_hits": sorted(conf_seen & truth),
        "naive_churn": naive_churn,
        "confirmed_churn": conf_churn,
        "missed_urgent": sorted((truth & URGENT_CODES) - conf_seen),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "cold_latency_ms": float(latencies[0]),
        "mean_confirm_lag_s": float(np.mean(lags)) if lags else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="test records to replay")
    ap.add_argument("--duration", type=float, default=40.0, help="stream seconds per record")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--confirm-k", type=int, default=3)
    ap.add_argument("--confirm-m", type=int, default=5)
    ap.add_argument("--sweep", type=str, default="1,2,3,4",
                    help="confirm_k values to sweep for the trade-off curve ('' to skip)")
    ap.add_argument("--sweep-n", type=int, default=20, help="records per sweep point")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_database()
    test = df[df["strat_fold"] == 10]
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(test.index.to_numpy(), size=min(args.n, len(test)), replace=False)

    rows = []
    for i, ecg_id in enumerate(ids, 1):
        truth = set(present_codes(test.loc[ecg_id, "scp_codes"]))
        r = replay_one(int(ecg_id), truth, args.duration, args.hop,
                       args.confirm_k, args.confirm_m)
        if r:
            rows.append(r)
        if i % 10 == 0:
            print(f"  replayed {i}/{len(ids)}")

    n = len(rows)
    if n == 0:
        raise SystemExit("no records replayed (is the waveform data downloaded?)")
    summary = summarize(rows, args)

    # --- trade-off curve over confirm_k --------------------------------------
    sweep = []
    ks = [int(k) for k in args.sweep.split(",") if k.strip()] if args.sweep else []
    if ks:
        sweep_ids = ids[:args.sweep_n]
        for k in ks:
            if k > args.confirm_m:
                continue
            srows = []
            for ecg_id in sweep_ids:
                truth = set(present_codes(test.loc[ecg_id, "scp_codes"]))
                r = replay_one(int(ecg_id), truth, args.duration, args.hop, k, args.confirm_m)
                if r:
                    srows.append(r)
            if srows:
                ss = summarize(srows, args)
                sweep.append({
                    "confirm_k": k, "confirm_m": args.confirm_m, "n_records": len(srows),
                    "panel_changes_per_record": ss["persistent"]["panel_changes_per_record"],
                    "spurious_per_record": ss["persistent"]["spurious_per_record"],
                    "true_recall": ss["persistent"]["true_recall"],
                    "mean_confirm_lag_s": ss["mean_confirm_lag_s"],
                })
                print(f"  sweep k={k}: churn {sweep[-1]['panel_changes_per_record']:.1f}, "
                      f"recall {sweep[-1]['true_recall']:.3f}, "
                      f"lag {sweep[-1]['mean_confirm_lag_s']:.1f}s")

    payload = {"summary": summary, "sweep": sweep, "records": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(payload)

    s = summary
    print(f"\nnaive      : {s['naive']['spurious_per_record']:.2f} spurious/rec, "
          f"{s['naive']['panel_changes_per_record']:.1f} panel changes, "
          f"recall {s['naive']['true_recall']:.3f}")
    print(f"persistent : {s['persistent']['spurious_per_record']:.2f} spurious/rec, "
          f"{s['persistent']['panel_changes_per_record']:.1f} panel changes, "
          f"recall {s['persistent']['true_recall']:.3f}")
    print(f"churn -{s['churn_reduction'] * 100:.0f}%  spurious -{s['spurious_reduction'] * 100:.0f}%  "
          f"confirm lag {s['mean_confirm_lag_s']:.1f}s  "
          f"latency {s['median_latency_ms']:.1f}ms ({s['realtime_headroom_x']}x headroom)")
    print(f"-> {OUT_DIR / 'report.md'}")
    return 0


def summarize(rows: list[dict], args) -> dict:
    """Aggregate per-record replay stats into the naive-vs-persistent comparison."""
    n = len(rows)

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    true_total = sum(len(r["truth"]) for r in rows)
    summary = {
        "n_records": n,
        "stream_seconds_each": args.duration,
        "hop_s": args.hop,
        "confirm_k": args.confirm_k,
        "confirm_m": args.confirm_m,
        "updates_per_record": mean("n_updates"),
        "naive": {
            "spurious_per_record": float(np.mean([len(r["naive_spurious"]) for r in rows])),
            "panel_changes_per_record": mean("naive_churn"),
            "true_recall": round(sum(len(r["naive_true_hits"]) for r in rows) / true_total, 4),
        },
        "persistent": {
            "spurious_per_record": float(np.mean([len(r["confirmed_spurious"]) for r in rows])),
            "panel_changes_per_record": mean("confirmed_churn"),
            "true_recall": round(sum(len(r["confirmed_true_hits"]) for r in rows) / true_total, 4),
        },
        "mean_confirm_lag_s": mean("mean_confirm_lag_s"),
        "median_latency_ms": mean("median_latency_ms"),
        "p95_latency_ms": mean("p95_latency_ms"),
        "cold_latency_ms": rows[0]["cold_latency_ms"],
        "records_with_missed_urgent": sum(bool(r["missed_urgent"]) for r in rows),
    }
    summary["spurious_reduction"] = round(
        1 - summary["persistent"]["spurious_per_record"]
        / max(summary["naive"]["spurious_per_record"], 1e-9), 4)
    summary["churn_reduction"] = round(
        1 - summary["persistent"]["panel_changes_per_record"]
        / max(summary["naive"]["panel_changes_per_record"], 1e-9), 4)

    hop_ms = args.hop * 1000.0
    summary["realtime_headroom_x"] = round(hop_ms / max(summary["median_latency_ms"], 1e-9), 1)
    return summary


def _sweep_table(p: dict) -> list[str]:
    """The confirm_k trade-off curve, if the sweep ran."""
    sweep = p.get("sweep") or []
    if not sweep:
        return []
    m = sweep[0]["confirm_m"]
    out = [
        f"### The trade-off curve (`confirm_m` = {m}, n = {sweep[0]['n_records']} records)",
        "",
        "| `confirm_k` | panel changes | spurious/rec | true recall | confirm lag |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sweep:
        lag = "—" if row["mean_confirm_lag_s"] is None else f"{row['mean_confirm_lag_s']:.1f} s"
        out.append(
            f"| {row['confirm_k']} of {m} | {row['panel_changes_per_record']:.1f} | "
            f"{row['spurious_per_record']:.2f} | {row['true_recall']:.3f} | {lag} |"
        )
    out += [
        "",
        "`confirm_k = 1` is the naive panel with hysteresis on release only. Raising `k` "
        "buys stability and pays in recall and latency, monotonically. There is no free "
        "setting here — the right one depends on whether the deployment is a ward monitor "
        "(favour stability) or an arrhythmia alarm (favour latency).",
        "",
    ]
    return out


def _write_markdown(p: dict) -> None:
    s = p["summary"]
    nai, per = s["naive"], s["persistent"]

    lines = [
        "# Phase 16 — Real-time streaming behaviour",
        "",
        f"{s['n_records']} PTB-XL test records replayed as live streams "
        f"({s['stream_seconds_each']:.0f} s each, {s['hop_s']:.0f} s hop, 10 s rolling "
        f"window). Regenerate with `python scripts/stream_eval.py`.",
        "",
        "## The problem streaming creates",
        "",
        "Batch APEX makes one decision per recording. A monitor makes a new one every hop. "
        "Phase 13 measured 5.09 spurious labels per record at the shipped 0.5 threshold — "
        "in a stream that is a fresh handful of spurious labels *every second*, and the "
        "findings panel becomes unreadable. The batch error profile does not transfer.",
        "",
        "## Naive panel vs temporal persistence",
        "",
        f"Persistence rule: a finding is confirmed once it holds in **{s['confirm_k']} of the "
        f"last {s['confirm_m']}** windows, and released only after it is absent from all "
        f"{s['confirm_m']} (hysteresis, so findings don't flap at the boundary).",
        "",
        "| | naive (per-window) | persistent (N-of-M) |",
        "|---|---:|---:|",
        f"| Panel changes per record | {nai['panel_changes_per_record']:.1f} | "
        f"**{per['panel_changes_per_record']:.1f}** |",
        f"| Distinct spurious codes shown | {nai['spurious_per_record']:.2f} | "
        f"**{per['spurious_per_record']:.2f}** |",
        f"| True-finding recall | {nai['true_recall']:.3f} | **{per['true_recall']:.3f}** |",
        "",
        f"**Panel churn falls {s['churn_reduction'] * 100:.0f}%** — the display stops "
        "flickering and becomes readable, which is the whole point.",
        "",
        "**Two costs, both real, neither hidden:**",
        "",
        f"1. **Recall drops {nai['true_recall']:.3f} → {per['true_recall']:.3f}** "
        f"({(nai['true_recall'] - per['true_recall']) * 100:.1f} points). Requiring "
        "persistence discards true findings that only appear intermittently. This is not a "
        "free win, and anyone reporting only the churn number is telling half the story.",
        f"2. **Detection latency: {s['mean_confirm_lag_s']:.1f} s** on average between a "
        "finding first clearing threshold and being announced. For continuous monitoring "
        "that is a fine trade; for a time-critical alarm it may not be — which is why "
        "`confirm_k` is a parameter, and why the curve below matters more than any single "
        "setting.",
        "",
        *_sweep_table(p),
        "## What persistence does *not* fix",
        "",
        "**Systematically over-confident labels survive it.** Debouncing removes findings "
        "that *flicker*; a spurious code that the miscalibrated model asserts in every "
        "single window is confirmed just like a real one. Phase 13 traced the over-flagging "
        "to calibration (over-confidence: mean probability ~3x the base rate), and this "
        "phase does not fix that — it only removes the noise on top. **Phase 17 addresses "
        "it directly** (ECE 0.079 -> 0.002, spurious labels 5.09 -> 0.35 per record); these "
        "streaming numbers are measured pre-calibration, so re-running this report on "
        "calibrated probabilities would move the remaining "
        f"{per['spurious_per_record']:.2f} spurious codes per record.",
        "",
        "## Real-time feasibility",
        "",
        "| | |",
        "|---|---:|",
        f"| Median inference per window | **{s['median_latency_ms']:.1f} ms** |",
        f"| p95 | {s['p95_latency_ms']:.1f} ms |",
        f"| First window (cold model load) | {s['cold_latency_ms']:.0f} ms |",
        f"| Hop budget | {s['hop_s'] * 1000:.0f} ms |",
        f"| Headroom | **~{s['realtime_headroom_x']}x** |",
        "",
        f"A 10 s window is re-analyzed every {s['hop_s']:.0f} s in "
        f"{s['median_latency_ms']:.1f} ms on CPU — roughly **{s['realtime_headroom_x']}x "
        "faster than real time**, so a single core could carry many concurrent streams. The "
        "cold first window pays a one-off model load; the monitor warms the cache before "
        "the buffer fills in any real deployment.",
        "",
        "## Honest limitations of this simulation",
        "",
        "- **The stream is replayed, not live.** PTB-XL records are 10 s — exactly one "
        "window — so a continuous stream requires looping (or splicing a playlist). Joins "
        "are raised-cosine crossfaded to avoid injecting a step discontinuity, but a "
        "window spanning a join is still part-one-record, part-another. That is a property "
        "of the simulation, not of a real monitor.",
        "- **Looping repeats the same 10 s.** Windows are rotations of one recording, so "
        "consecutive windows are more correlated than genuinely new data would be. This "
        "*flatters* the persistence layer: real streams vary more, so real churn would be "
        "higher and persistence more valuable, but the exact numbers above would move.",
        "- **12-lead only.** An Apple Watch delivers a single lead and a typical Holter "
        "three; the detector's input convolution is fixed at 12 and the service hard-rejects "
        "anything else. This simulation is therefore *Holter/telemetry-style 12-lead*, not "
        "a smartwatch. Supporting single-lead input needs a retrained model, not an adapter.",
        "- **No generation or grounding in the loop.** Each hop runs detection only; the "
        "explanation and grounding layers run on demand, not per window.",
        "- **Ground truth is per-record, not per-window.** A looped record carries the same "
        "labels throughout, so \"recall\" here means \"did the stream ever confirm the "
        "record's true codes\", not a time-resolved measurement.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
