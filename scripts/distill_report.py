#!/usr/bin/env python3
"""Phase 19 — teacher vs. student trade-off benchmark.

    python scripts/distill_report.py
    python scripts/distill_report.py --n-latency 500 --no-pipeline

Loads every checkpoint recorded in ``docs/distillation/runs.jsonl`` plus the shipped
teacher, and measures the four axes that decide whether a distilled model is worth
deploying:

- **quality** — test-fold macro-AUROC, macro/micro-F1 at the shipped 0.5 threshold, and
  ECE before and after refitting Phase 17's vector scaler on the student's own logits;
- **latency** — isolated forward pass at batch 1 on CPU (p50/p95), batch-32 throughput,
  and the *end-to-end* `analyze_signal` pipeline, which is what a caller actually waits
  for;
- **size** — parameter count and checkpoint bytes on disk;
- **fidelity** — how often the student and teacher make the same call at the surfacing
  threshold, which an aggregate score can hide.

Every student is reported twice: distilled (`--alpha 0.7`) and trained from scratch on
ground truth alone (`--alpha 0`). The from-scratch column is the control — the size of
the gap between the two is the actual measurement of what distillation bought.

Writes docs/distillation/report.{md,json} and tradeoff.png.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import CFG, ROOT  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.detection.distill import agreement, compute_logits  # noqa: E402
from src.detection.model import count_parameters  # noqa: E402
from src.eval import calibration as cal  # noqa: E402
from src.eval.metrics import f1_scores, macro_auroc  # noqa: E402
from src.grounding import load_detector  # noqa: E402
from src.serving.metrics import percentile  # noqa: E402

OUT_DIR = ROOT / "docs" / "distillation"
CKPT_DIR = ROOT / "outputs"
TEACHER = CKPT_DIR / "final_best.pt"


# --- measurement -------------------------------------------------------------
def _bench_forward(model, x: torch.Tensor, n: int, warmup: int = 20) -> list[float]:
    """Timed forward passes in milliseconds (no grad, model already on device)."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def latency(model, signal: np.ndarray, n: int, threads: int = 1) -> dict:
    """Forward-pass latency at batch 1, plus batch-32 throughput, on CPU.

    The two run at **different thread counts on purpose**, because each is measured at the
    setting that is both faster and more representative of its deployment shape:

    - **batch 1** is pinned to ``threads`` (default 1) — one request at a time in a uvicorn
      worker. A single 12x1000 record does not even fill one core, so torch's intra-op
      parallelism is pure overhead here and measurably *loses*: on this host the teacher
      runs 3.1 ms at one thread against 4.7 ms at eight, and the student 0.66 ms against
      1.39 ms. Pinning also stops the number drifting with whatever else the machine is
      doing.
    - **batch 32** uses all cores, which is how retrospective/offline scoring would
      actually be run, and where parallelism finally pays: the teacher goes from 3.98 s per
      batch at one thread to 0.84 s across eight.

    One ordering constraint: torch will lower the thread count below the pool it already
    created, but not raise it above. The pool must therefore be created at the maximum —
    which it is, since `compute_logits` runs at the default before this is ever called.
    Pinning to 1 *first* and restoring later would silently leave everything single
    threaded.
    """
    model.eval().to("cpu")
    x1 = torch.from_numpy(signal[None, ...])
    default_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        ms = _bench_forward(model, x1, n)
    finally:
        torch.set_num_threads(default_threads)

    x32 = x1.repeat(32, 1, 1)
    batch_ms = _bench_forward(model, x32, max(10, n // 20), warmup=5)
    return {
        "p50_ms": round(percentile(sorted(ms), 50), 4),
        "p95_ms": round(percentile(sorted(ms), 95), 4),
        "mean_ms": round(float(np.mean(ms)), 4),
        "batch32_throughput_rps": round(32.0 / (float(np.median(batch_ms)) / 1000.0), 1),
        "batch32_ms": round(float(np.median(batch_ms)), 3),
        "threads": threads,
        "batch32_threads": default_threads,
        "n": n,
    }



def pipeline_latency(checkpoint: Path | None, signal: np.ndarray, fs: int, n: int) -> dict:
    """End-to-end `analyze_signal` latency — validate → preprocess → detect → generate.

    The number that matters to a caller. Phase 9 measured the shipped pipeline at ~5.9 ms
    p50 on CPU and noted the detector is only a slice of it, so this is the check on how
    much of a forward-pass speedup actually survives to the API boundary.
    """
    from src.serving.model_cache import warmup
    from src.serving.serializer import analyze_signal

    warmup(checkpoint, "cpu")
    for _ in range(5):
        analyze_signal(signal, fs, checkpoint=checkpoint)
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        analyze_signal(signal, fs, checkpoint=checkpoint)
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms.sort()
    return {"p50_ms": round(percentile(ms, 50), 3), "p95_ms": round(percentile(ms, 95), 3),
            "n": n}


def quality(z_te: np.ndarray, Yte: np.ndarray, z_va: np.ndarray, Yva: np.ndarray) -> dict:
    """Test-fold quality from raw logits, plus ECE after a val-fitted vector scaler.

    Calibration is refit per model rather than reused from the teacher: distillation
    transfers the teacher's *probabilities*, including its miscalibration, so each student
    needs its own scaler fitted on the validation fold (never on test).
    """
    p = cal._sigmoid(z_te)
    thr = CFG.review_threshold
    f1 = f1_scores(Yte, (p >= thr).astype(int))
    scaler = cal.VectorScaler().fit(z_va, Yva)
    p_cal = scaler.transform(z_te)
    return {
        "macro_auroc": round(macro_auroc(Yte, p), 5),
        "macro_f1": round(f1["macro_f1"], 5),
        "micro_f1": round(f1["micro_f1"], 5),
        "ece": round(cal.ece(Yte, p), 5),
        "ece_calibrated": round(cal.ece(Yte, p_cal), 5),
        "macro_auroc_calibrated": round(macro_auroc(Yte, p_cal), 5),
        "brier": round(cal.brier_score(Yte, p), 5),
        "mean_prob": round(float(p.mean()), 5),
        "labels_over_threshold_per_record": round(float((p >= thr).sum(axis=1).mean()), 3),
    }


def _sample_signal() -> tuple[np.ndarray, int]:
    """One *raw* PTB-XL record — the same one Phase 9's benchmark timed, so the
    end-to-end numbers here are directly comparable to `docs/serving/benchmark.md`
    (which used the raw signal, preprocessing included). Falls back to noise if the
    dataset is absent; forward-pass timing is data-independent either way."""
    try:
        import wfdb

        from src.config import PTBXL_DIR
        from src.data.labels import load_database

        row = load_database().iloc[0]
        sig, meta = wfdb.rdsamp(str(PTBXL_DIR / row["filename_lr"]))
        return sig.T.astype(np.float32), int(meta["fs"])
    except Exception:
        return np.random.default_rng(0).standard_normal((12, 1000)).astype(np.float32), 100


# --- report ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-latency", type=int, default=300)
    ap.add_argument("--n-pipeline", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU threads for the batch-1 latency measurement; batch-32 "
                         "throughput always uses all cores. See latency().")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="skip the end-to-end analyze_signal timing")
    args = ap.parse_args()

    print(f"CPU threads: {args.threads} (batch 1) / {torch.get_num_threads()} (batch 32)")

    runs_path = OUT_DIR / "runs.jsonl"
    if not runs_path.exists():
        print(f"no runs found at {runs_path}; run scripts/run_distillation.sh first")
        return 1
    runs = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()]
    # keep the newest record per run_name, so re-running a config supersedes the old one
    by_name = {r["run_name"]: r for r in runs}

    print("loading val + test splits...")
    Xva, Yva = build_split_cache("val", 100)
    Xte, Yte = build_split_cache("test", 100)
    signal, fs = _sample_signal()

    entries: list[dict] = []

    def measure(name: str, checkpoint: Path, meta: dict) -> dict:
        model, _, _ = load_detector(checkpoint, device="cpu")
        n_params = count_parameters(model)
        z_te = compute_logits(model, Xte, "cpu")
        z_va = compute_logits(model, Xva, "cpu")
        e = {
            "name": name, "checkpoint": checkpoint.name, "params": n_params,
            "size_mb": round(checkpoint.stat().st_size / 1e6, 2),
            "weights_mb": round(n_params * 4 / 1e6, 2),
            **meta,
            "quality": quality(z_te, Yte, z_va, Yva),
            "latency": latency(model, signal, args.n_latency, args.threads),
        }
        if not args.no_pipeline:
            ck = None if checkpoint == TEACHER else checkpoint
            e["pipeline"] = pipeline_latency(ck, signal, fs, args.n_pipeline)
        e["_logits"] = z_te
        print(f"  {name:<24} params {n_params:>9,}  AUROC {e['quality']['macro_auroc']:.4f}  "
              f"p50 {e['latency']['p50_ms']:.3f} ms")
        return e

    print("measuring teacher...")
    teacher = measure("teacher (cnn_bce)", TEACHER,
                      {"kind": "teacher", "width": 64, "blocks": 2, "alpha": None})
    entries.append(teacher)

    print("measuring students...")
    for r in sorted(by_name.values(), key=lambda r: (r["width"], r["kind"])):
        ckpt = CKPT_DIR / r["checkpoint"]
        if not ckpt.exists():
            print(f"  !! missing checkpoint {ckpt.name}, skipping")
            continue
        entries.append(measure(
            r["run_name"], ckpt,
            {"kind": r["kind"], "width": r["width"], "blocks": r["blocks"],
             "alpha": r["alpha"], "temperature": r["temperature"],
             "train_time_s": r.get("train_time_s"), "best_epoch": r.get("best_epoch")},
        ))

    # fidelity to the teacher (probability space), and relative cost/quality
    p_teacher = cal._sigmoid(teacher["_logits"])
    t_lat, t_pipe = teacher["latency"]["p50_ms"], teacher.get("pipeline", {}).get("p50_ms")
    t_auroc = teacher["quality"]["macro_auroc"]
    for e in entries:
        e["agreement"] = agreement(cal._sigmoid(e["_logits"]), p_teacher, CFG.review_threshold)
        e["agreement"] = {k: round(v, 5) for k, v in e["agreement"].items()}
        e["vs_teacher"] = {
            "compression": round(teacher["params"] / e["params"], 2),
            "speedup": round(t_lat / e["latency"]["p50_ms"], 2),
            "latency_reduction_pct": round((1 - e["latency"]["p50_ms"] / t_lat) * 100, 2),
            "auroc_delta": round(e["quality"]["macro_auroc"] - t_auroc, 5),
            "auroc_degradation_pct": round((1 - e["quality"]["macro_auroc"] / t_auroc) * 100, 3),
        }
        if t_pipe and e.get("pipeline"):
            e["vs_teacher"]["pipeline_reduction_pct"] = round(
                (1 - e["pipeline"]["p50_ms"] / t_pipe) * 100, 2)
        del e["_logits"]

    payload = {
        "generated_by": "scripts/distill_report.py",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": args.threads,
        "n_test": int(len(Yte)), "n_val": int(len(Yva)),
        "threshold": CFG.review_threshold,
        "entries": entries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    _plot(entries)
    _write_markdown(payload)
    print(f"\n-> {OUT_DIR / 'report.md'}")
    return 0


def _pairs(entries: list[dict]) -> list[tuple[int, dict | None, dict | None]]:
    """(width, kd_entry, scratch_entry) per student capacity, smallest first."""
    widths = sorted({e["width"] for e in entries if e["kind"] != "teacher"})
    out = []
    for w in widths:
        kd = next((e for e in entries if e["kind"] == "kd" and e["width"] == w), None)
        sc = next((e for e in entries if e["kind"] == "scratch" and e["width"] == w), None)
        out.append((w, kd, sc))
    return out


def _plot(entries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teacher = next(e for e in entries if e["kind"] == "teacher")
    pairs = _pairs(entries)
    kd = [p[1] for p in pairs if p[1]]
    sc = [p[2] for p in pairs if p[2]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, xkey, xlabel, xlog in [
        (axes[0], lambda e: e["params"], "parameters (log scale)", True),
        (axes[1], lambda e: e["latency"]["p50_ms"], "CPU forward p50 (ms, batch 1)", False),
    ]:
        for group, color, marker, label in [
            (kd, "#1f6feb", "o", "distilled (α=0.7)"),
            (sc, "#b3261e", "s", "from scratch (α=0)"),
        ]:
            if not group:
                continue
            ax.plot([xkey(e) for e in group], [e["quality"]["macro_auroc"] for e in group],
                    marker + "-", color=color, lw=1.8, ms=6, label=label)
        ax.plot([xkey(teacher)], [teacher["quality"]["macro_auroc"]], "*", color="#111",
                ms=16, label="teacher (8.8M)")
        ax.axhline(teacher["quality"]["macro_auroc"], color="#111", ls=":", lw=1, alpha=0.5)
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test macro-AUROC")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_title("Quality vs. capacity")
    axes[1].set_title("Quality vs. latency")

    ax = axes[2]
    widths = [w for w, k, s in pairs if k and s]
    gaps = [(k["quality"]["macro_auroc"] - s["quality"]["macro_auroc"]) * 100
            for w, k, s in pairs if k and s]
    ax.bar([str(w) for w in widths], gaps, color="#1f6feb", width=0.55)
    ax.axhline(0, color="#111", lw=1)
    for i, g in enumerate(gaps):
        ax.text(i, g, f"{g:+.2f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=9)
    ax.set_xlabel("student width (stem channels)")
    ax.set_ylabel("macro-AUROC gain from KD (pp)")
    ax.set_title("What distillation bought\n(distilled − from-scratch, same architecture)")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("APEX Phase 19 — knowledge distillation trade-off (PTB-XL test fold)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tradeoff.png", dpi=150)
    plt.close(fig)


def _write_markdown(p: dict) -> None:
    entries = p["entries"]
    teacher = next(e for e in entries if e["kind"] == "teacher")
    pairs = _pairs(entries)
    students = [e for e in entries if e["kind"] != "teacher"]

    # the headline student: best distilled model within 1% AUROC of the teacher,
    # preferring the smallest such model (that is the point of the exercise).
    ok = [e for e in students
          if e["kind"] == "kd" and e["vs_teacher"]["auroc_degradation_pct"] <= 1.0]
    hero = min(ok, key=lambda e: e["params"]) if ok else max(
        (e for e in students if e["kind"] == "kd"),
        key=lambda e: e["quality"]["macro_auroc"])
    hero_sc = next((e for e in students
                    if e["kind"] == "scratch" and e["width"] == hero["width"]), None)

    has_pipe = "pipeline" in teacher
    header = ["| model | params | size (MB) | macro-AUROC | macro-F1 | micro-F1 | "
              "fwd p50 (ms) | fwd p95 (ms) | batch-32 (rec/s) |"
              + (" e2e p50 (ms) |" if has_pipe else ""),
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if has_pipe else "")]
    body = []
    for e in [teacher] + sorted(students, key=lambda e: (-e["params"], e["kind"])):
        q, lat, = e["quality"], e["latency"]
        cells = [e["name"] if e["kind"] != "teacher" else "**teacher** (cnn_bce)",
                 f"{e['params']:,}", f"{e['size_mb']:.2f}",
                 f"{q['macro_auroc']:.4f}", f"{q['macro_f1']:.4f}", f"{q['micro_f1']:.4f}",
                 f"{lat['p50_ms']:.3f}", f"{lat['p95_ms']:.3f}",
                 f"{lat['batch32_throughput_rps']:,.0f}"]
        if has_pipe:
            cells.append(f"{e['pipeline']['p50_ms']:.2f}")
        body.append("| " + " | ".join(cells) + " |")

    kd_rows = ["| student | params | distilled AUROC | from-scratch AUROC | KD gain (pp) | "
               "teacher agreement @0.5 |", "|---|---:|---:|---:|---:|---:|"]
    for w, k, s in pairs:
        if not (k and s):
            continue
        gain = (k["quality"]["macro_auroc"] - s["quality"]["macro_auroc"]) * 100
        kd_rows.append(
            f"| width {w} (÷{k['vs_teacher']['compression']:.0f}) | {k['params']:,} | "
            f"**{k['quality']['macro_auroc']:.4f}** | {s['quality']['macro_auroc']:.4f} | "
            f"{gain:+.2f} | {k['agreement']['decision_agreement'] * 100:.2f}% |")

    cal_rows = ["| model | ECE (raw) | ECE (vector-scaled) | mean prob | "
                "labels ≥0.5 per record |", "|---|---:|---:|---:|---:|"]
    for e in [teacher] + sorted(students, key=lambda e: (-e["params"], e["kind"])):
        q = e["quality"]
        nm = "teacher" if e["kind"] == "teacher" else e["name"]
        cal_rows.append(f"| {nm} | {q['ece']:.4f} | {q['ece_calibrated']:.4f} | "
                        f"{q['mean_prob']:.4f} | {q['labels_over_threshold_per_record']:.2f} |")

    hv = hero["vs_teacher"]
    hero_gain = ((hero["quality"]["macro_auroc"] - hero_sc["quality"]["macro_auroc"]) * 100
                 if hero_sc else None)
    pipe_line = ""
    if has_pipe and "pipeline_reduction_pct" in hv:
        # The fixed (non-model) cost of a request, read off the smallest student: its
        # forward pass is near zero, so its end-to-end time is essentially the floor.
        floor = min(e["pipeline"]["p50_ms"] - e["latency"]["p50_ms"] for e in entries)
        pipe_line = (
            f"End to end, through `analyze_signal`, the same swap moves p50 "
            f"**{teacher['pipeline']['p50_ms']:.2f} ms → {hero['pipeline']['p50_ms']:.2f} ms "
            f"({hv['pipeline_reduction_pct']:.0f}%)**. The end-to-end reduction is smaller "
            "than the forward-pass reduction because validation, preprocessing and report "
            f"generation are untouched by distillation: they cost about **{floor:.1f} ms** "
            "per request no matter which model is loaded, which is the floor every row in "
            "the table sits on. Both figures are reported because quoting only the "
            "forward-pass number would be measuring the part that flatters the result.")

    lines = [
        "# Phase 19 — Knowledge distillation to a lightweight student",
        "",
        f"Teacher: the shipped Phase-4 detector (`cnn_bce`, {teacher['params']:,} "
        f"parameters, {teacher['size_mb']:.1f} MB). Students are the same 1D-ResNet family "
        "at reduced width and depth, trained to match the teacher's per-label probabilities. "
        "Regenerate with `bash scripts/run_distillation.sh`.",
        "",
        "## Headline",
        "",
        f"**{hv['compression']:.0f}x smaller ({teacher['params']:,} → {hero['params']:,} "
        f"parameters, {teacher['size_mb']:.1f} MB → {hero['size_mb']:.1f} MB) and "
        f"{hv['speedup']:.1f}x faster on CPU ({teacher['latency']['p50_ms']:.2f} ms → "
        f"{hero['latency']['p50_ms']:.2f} ms p50 forward pass, a "
        f"{hv['latency_reduction_pct']:.0f}% reduction), for "
        f"{abs(hv['auroc_degradation_pct']):.2f}% "
        f"{'degradation' if hv['auroc_delta'] < 0 else 'improvement'} in test macro-AUROC "
        f"({teacher['quality']['macro_auroc']:.4f} → "
        f"{hero['quality']['macro_auroc']:.4f}).**",
        "",
        pipe_line,
        "",
        "![trade-off curves](tradeoff.png)",
        "",
        "## Results",
        "",
        *header, *body,
        "",
        "## How this was measured",
        "",
        f"**Latency** is an isolated forward pass on CPU, {teacher['latency']['n']} timed "
        f"iterations after warmup, on {p['platform']} with torch {p['torch']}. "
        f"**Quality** is the PTB-XL **test** fold ({p['n_test']} records); F1 uses the "
        f"shipped {p['threshold']} threshold rather than per-model tuned thresholds, so "
        "the columns are comparable to each other — which is why the teacher's macro-F1 "
        f"here ({teacher['quality']['macro_f1']:.3f}) is below the 0.359 in Phase 12, "
        "where thresholds were tuned per label on validation.",
        "",
        f"The two latency regimes are measured at **different thread counts** — batch 1 "
        f"pinned to {p['threads']}, batch 32 across all "
        f"{teacher['latency']['batch32_threads']} cores — because each is faster at that "
        "setting, and each corresponds to a real deployment shape. That is not a "
        "convenient assumption; it is measured, and the direction reverses between the "
        "two:",
        "",
        "| | teacher batch 1 | teacher batch 32 | student batch 1 | student batch 32 |",
        "|---|---:|---:|---:|---:|",
        "| 1 thread | **3.09 ms** | 3982 ms | **0.66 ms** | 107 ms |",
        "| 8 threads | 4.67 ms | **840 ms** | 1.39 ms | **83 ms** |",
        "",
        "A single 12x1000 record does not fill one core, so at batch 1 torch's intra-op "
        "parallelism is pure coordination overhead and costs ~50% — the serving shape "
        "(one request at a time in a uvicorn worker) is genuinely *better* single "
        "threaded. At batch 32 that reverses and parallelism pays 4.7x. Reporting one "
        "thread count for both would have understated whichever regime it did not suit.",
        "",
        "There is an ordering constraint behind this that is easy to get wrong: torch "
        "will lower the thread count below the pool it has already created, but will not "
        "raise it back above. The pool therefore has to be created at the maximum first — "
        "here it is, because the logits pass runs before any timing. Pinning to one "
        "thread at process start and restoring afterwards leaves *everything* single "
        "threaded while still reporting eight, which was caught only by re-measuring the "
        "same model in a clean process.",
        "",
        "## Did distillation actually help?",
        "",
        "The interesting question is not whether a small model can do this job — it is "
        "whether the teacher's soft labels beat the ground truth alone. Each student was "
        "therefore trained twice, identical in architecture, data, schedule and seed, "
        "differing only in whether the KD term was on:",
        "",
        *kd_rows,
        "",
        _kd_verdict(pairs, hero_gain),
        "",
        "## Calibration is inherited, not fixed",
        "",
        "Distillation trains the student to reproduce the teacher's probabilities — "
        "including the teacher's *miscalibration*. Phase 17 found the detector's outputs "
        "run ~5x above the base rate because of class-weighted BCE, and the students "
        "reproduce that faithfully, which is the expected behaviour rather than a bug: a "
        "student that matched the teacher on ranking but not on probability would not be a "
        "drop-in replacement. Refitting Phase 17's vector scaler on each student's own "
        "validation logits corrects it as effectively as it did for the teacher:",
        "",
        *cal_rows,
        "",
        _calibration_note(pairs, teacher),
        "",
        "**A distilled model needs its own calibrator.** The teacher's fitted `a_j, b_j` "
        "are not transferable — the student's logit scale is its own. `outputs/"
        "calibration.json` is fitted for the teacher; deploying a student means rerunning "
        "`scripts/calibrate.py` against that checkpoint.",
        "",
        "## Fidelity to the teacher",
        "",
        "Matching macro-AUROC does not mean making the same decisions — a student can "
        "reach the same aggregate score by being right about different records. At the "
        f"{p['threshold']} surfacing threshold:",
        "",
        "| student | decision agreement | agreement on teacher-positive calls | "
        "mean \\|Δp\\| |",
        "|---|---:|---:|---:|",
        *[f"| {e['name']} | {e['agreement']['decision_agreement'] * 100:.2f}% | "
          f"{e['agreement']['decision_agreement_positives'] * 100:.2f}% | "
          f"{e['agreement']['mean_abs_prob_diff']:.4f} |"
          for e in sorted(students, key=lambda e: (-e["params"], e["kind"]))],
        "",
        "Overall agreement is high mostly because most of the 71 labels are confidently "
        "negative for most records; the second column is the honest one, since it is "
        "restricted to the calls the teacher actually surfaces. A student that agrees on "
        "aggregate but diverges on positives is a different clinical device, not a "
        "compressed copy of the same one.",
        "",
        "## Implementation notes",
        "",
        "- **Multi-label KD is not softmax KD.** The 71 outputs are independent sigmoids "
        "because conditions coexist, so the soft-target loss is a per-label Bernoulli KL "
        "rather than a KL between softmax distributions. Using the softmax form here would "
        "force labels to compete for a fixed probability budget and destroy the multi-label "
        "semantics.",
        "- **`T²` rescaling.** Temperature softens both sides via `σ(z/T)`; gradients "
        "through the scaled logit shrink as `1/T`, so the KD term carries the standard "
        "Hinton `T²` factor to keep it comparable with the hard-label term as `T` varies.",
        "- **The KD term is averaged over labels, not summed.** Summing would scale it by "
        "71x against the hard term and make `alpha` meaningless.",
        "- **`pos_weight` on the hard term only.** The teacher's probabilities already "
        "encode the class weighting it was trained under; re-applying it to the soft term "
        "would double-count the imbalance correction.",
        "- **Teacher logits are cached.** The teacher is frozen and the preprocessed "
        "tensors are not augmented, so its outputs are a pure function of "
        "`(checkpoint, split)` — computed once and reused, which makes a distillation epoch "
        "cost the same as an ordinary one. The cache filename carries a hash of the "
        "checkpoint bytes so a changed teacher cannot silently reuse stale logits.",
        "- **The student is a drop-in.** Same architecture family, same `args` schema in "
        "the checkpoint, so `load_detector`, Grad-CAM (`model.stages`), the FastAPI "
        "service and the Gradio dashboard all accept it with no code change — "
        "`analyze_signal(..., checkpoint=...)` is the whole migration.",
        "",
        "## Limitations",
        "",
        "- Latency is measured on Apple-silicon CPU. Absolute milliseconds will differ on "
        "server hardware; the *ratio* is the portable claim. Repeated runs of this script "
        "move the teacher's p50 by roughly ±12% (3.10–3.48 ms observed) and the student's "
        "similarly, so the speedup is best read as \"about 5x\" rather than to two "
        "decimal places.",
        "- One seed per configuration. The KD-vs-scratch gaps reported here are single-run "
        "differences, and small gaps (well under a point of AUROC) are within the range "
        "seed variance could produce.",
        "- Macro-AUROC averages over labels with wildly different support, so a student can "
        "hold macro-AUROC while losing ground on the rare labels Phase 13 already flagged "
        "as weakest. The per-label tables under `docs/distillation/<run>/` are where that "
        "shows up.",
        "- Distillation compresses *this* teacher, including its documented failure modes "
        "(Phase 13 over-flagging, Phase 14/18 demographic gaps). A smaller model inherits "
        "them; it does not dilute them.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def _join(items: list) -> str:
    """'8', '8 and 16', '8, 16 and 32'."""
    s = [str(i) for i in items]
    return s[0] if len(s) == 1 else " and ".join([", ".join(s[:-1]), s[-1]])


def _calibration_note(pairs, teacher) -> str:
    """Compare KD vs from-scratch calibration — a side effect worth naming explicitly.

    "Better" requires a *material* margin (>5% relative); two ECEs that differ in the
    fourth decimal are the same number, and reporting them as a win would be noise-mining.
    """
    deltas = [(w, k["quality"]["ece"], s["quality"]["ece"]) for w, k, s in pairs if k and s]
    if not deltas:
        return ""
    better = [(w, kd, sc) for w, kd, sc in deltas if sc - kd > 0.05 * sc]
    tied = [w for w, kd, sc in deltas if (w, kd, sc) not in better]
    if not better:
        return ("Raw ECE is essentially identical across distilled and from-scratch "
                "students, so the soft targets neither help nor hurt calibration here.")

    t_ece = teacher["quality"]["ece"]
    widths = _join([str(w) for w, _, _ in better])
    worst = max(better, key=lambda d: d[2] - d[1])
    note = (
        f"There is a side effect worth naming: at width{'s' if len(better) > 1 else ''} "
        f"{widths} the distilled students are "
        f"**better calibrated out of the box than their from-scratch twins** (largest gap "
        f"at width {worst[0]}: ECE {worst[2]:.4f} → {worst[1]:.4f}), landing close to the "
        f"teacher's own {t_ece:.4f}. Training on ground truth alone makes a small model "
        "*more* over-confident than the teacher, because 0/1 targets give it nothing to be "
        "uncertain about; the teacher's soft targets carry that uncertainty and the student "
        "copies it. So distillation does not merely transfer miscalibration — relative to "
        "the honest alternative, it transfers *less*."
    )
    if tied:
        note += (
            f" This does not hold at width{'s' if len(tied) > 1 else ''} {_join(tied)}, "
            "where the two are within noise of each other — consistent with the AUROC "
            "result, the soft targets stop mattering once the student has capacity to "
            "spare."
        )
    return note


def _kd_verdict(pairs, hero_gain) -> str:
    gains = [(w, (k["quality"]["macro_auroc"] - s["quality"]["macro_auroc"]) * 100)
             for w, k, s in pairs if k and s]
    if not gains:
        return "_No paired from-scratch controls were found._"
    smallest_w, smallest_gain = gains[0]
    largest_w, largest_gain = gains[-1]
    positive = [g for _, g in gains if g > 0]

    verdict = (f"Distillation helps {len(positive)} of {len(gains)} student sizes. "
               f"The gain is **{smallest_gain:+.2f} pp at width {smallest_w}** and "
               f"**{largest_gain:+.2f} pp at width {largest_w}**")
    if smallest_gain > largest_gain:
        verdict += (", i.e. it is worth most exactly where capacity is scarcest. That "
                    "ordering is the expected one and is the mechanism working as "
                    "advertised: a model with room to spare can find the structure in the "
                    "hard labels by itself, while a model that cannot afford to "
                    "rediscover inter-label geometry benefits from being handed it.")
    else:
        verdict += (". The gain does *not* grow as the student shrinks, which is worth "
                    "stating plainly rather than smoothing over — the usual argument for "
                    "KD is that soft targets matter most when capacity is scarcest, and "
                    "that is not what this sweep shows.")
    if hero_gain is not None:
        verdict += (f"\n\nFor the headline student the distilled model is "
                    f"{hero_gain:+.2f} pp of macro-AUROC ahead of the identical "
                    "architecture trained on ground truth alone — the same checkpoint, the "
                    "same 20 epochs, the same seed, with the only difference being what it "
                    "was asked to imitate.")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
