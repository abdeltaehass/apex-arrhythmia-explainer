#!/usr/bin/env python3
"""Run the Phase-5 grounding layer on real PTB-XL records.

    # one record: ground its detected labels, save an overlay figure + JSON
    python scripts/run_grounding.py --ecg-id 12

    # one record, one label
    python scripts/run_grounding.py --ecg-id 12 --label AFIB

    # sanity sweep: sample records carrying a code/superclass, grade grounding vs.
    # clinical intuition, and print an aggregate (feeds docs/grounding/sanity_check.md)
    python scripts/run_grounding.py --scan AFIB --n 40
    python scripts/run_grounding.py --scan STTC --n 40      # any STTC-superclass code

Figures + JSON land in docs/grounding/. Needs the 100 Hz waveforms (make data-100) and
a trained checkpoint (outputs/final_best.pt by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as `python scripts/run_grounding.py` (repo root not on path otherwise)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import PTBXL_DIR, ROOT  # noqa: E402
from src.data.labels import load_database, load_scp_statements  # noqa: E402
from src.grounding import (  # noqa: E402
    LEAD_NAMES,
    ground,
    load_detector,
    sanity_check,
    summarize,
)
from src.preprocessing.pipeline import preprocess  # noqa: E402

OUT_DIR = ROOT / "docs" / "grounding"


def load_record(ecg_id: int, df, sampling_rate: int = 100):
    """Return ``(clean, rpeaks, present_codes)`` for one PTB-XL record."""
    import wfdb

    row = df.loc[ecg_id]
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signal, _ = wfdb.rdsamp(str(PTBXL_DIR / row[col]))
    raw = signal.T.astype(np.float32)  # (12, T)
    clean, rpeaks = preprocess(raw, fs_in=sampling_rate, fs_out=100)
    return clean, rpeaks, sorted(row["scp_codes"].keys())


def detect(model, clean) -> np.ndarray:
    import torch

    x = torch.from_numpy(clean).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0].cpu().numpy()


def plot_grounding(clean, sal, rpeaks, ecg_id: int, code: str, result, scp, out_path: Path):
    """Overlay the per-lead saliency on the driving lead + a 12-lead saliency heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from src.grounding.regions import WAVE_WINDOWS

    fs = sal.fs
    t = np.arange(clean.shape[-1]) / fs
    top = sal.top_lead
    desc = scp.loc[code, "description"] if scp is not None and code in scp.index else code

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1.4])

    ax0.plot(t, clean[top], color="#222", lw=0.8, zorder=3)
    s = sal.per_lead[top]
    ax0.fill_between(t, clean[top].min(), clean[top].max(), where=s > 0.05, alpha=0.0)
    sc = ax0.scatter(t, clean[top], c=s, cmap="inferno", s=6, vmin=0, vmax=1, zorder=4)
    for r in rpeaks:
        ax0.axvline(r / fs, color="#3a7", lw=0.5, alpha=0.4, zorder=1)
    # shade the clinically-expected ST/T windows for the first few beats as a reference
    for r in rpeaks[:6]:
        for name in ("ST", "T"):
            lo, hi = WAVE_WINDOWS[name]
            ax0.axvspan((r + lo * fs) / fs, (r + hi * fs) / fs, color="#69c", alpha=0.08, zorder=0)
    fig.colorbar(sc, ax=ax0, label="saliency", pad=0.01)
    ax0.set_title(f"ecg_id {ecg_id} · {code} ({desc}) · p={sal.prob:.2f} · lead {LEAD_NAMES[top]} "
                  f"(top of {len(LEAD_NAMES)})\nsanity: {result.verdict.upper()} — {result.detail}",
                  fontsize=9)
    ax0.set_ylabel(f"lead {LEAD_NAMES[top]} (z)")
    ax0.set_xlim(0, t[-1])

    im = ax1.imshow(sal.per_lead, aspect="auto", cmap="inferno", vmin=0, vmax=1,
                    extent=[0, t[-1], 11.5, -0.5], interpolation="nearest")
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax1.set_ylabel("lead")
    ax1.set_xlabel("time (s)")
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.colorbar(im, ax=ax1, label="per-lead saliency", pad=0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_single(model, label_space, scp, df, ecg_id: int, label: str | None,
               threshold: float, save_fig: bool) -> list[dict]:
    clean, rpeaks, present = load_record(ecg_id, df)
    probs = detect(model, clean)

    if label:
        targets = [label]
    else:
        # ground every detected label (prob >= threshold), capped for a readable figure set
        targets = [label_space[j] for j in np.argsort(probs)[::-1]
                   if probs[j] >= threshold][:6]
    if not targets:
        top = int(np.argmax(probs))
        targets = [label_space[top]]

    sal_map = ground(model, clean, targets, label_space=label_space, fs=100)
    records = []
    for code, sal in sal_map.items():
        result = sanity_check(sal.per_lead, rpeaks, code, fs=100, scp=scp)
        rec = {
            "ecg_id": int(ecg_id), "label": code, "prob": round(float(sal.prob), 4),
            "true_label": code in present, "top_lead": LEAD_NAMES[sal.top_lead],
            "lead_importance": {LEAD_NAMES[i]: round(float(v), 3)
                                for i, v in enumerate(sal.lead_importance)},
            "verdict": result.verdict, "kind": result.kind, "detail": result.detail,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in result.metrics.items()},
        }
        records.append(rec)
        if save_fig:
            fig_path = OUT_DIR / "figures" / f"ecg{ecg_id}_{code}.png"
            plot_grounding(clean, sal, rpeaks, ecg_id, code, result, scp, fig_path)
            rec["figure"] = str(fig_path.relative_to(ROOT))
    return records


def scan(model, label_space, scp, df, target: str, n: int, threshold: float,
         seed: int) -> dict:
    """Sample records carrying ``target`` (code or superclass) and grade grounding."""
    is_superclass = target in set(scp["diagnostic_class"].dropna().unique())
    if is_superclass:
        codes = set(scp.index[scp["diagnostic_class"] == target])
        has = df["scp_codes"].apply(lambda d: bool(set(d) & codes))
    else:
        has = df["scp_codes"].apply(lambda d: target in d)
    pool = df[has].index.tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)

    from src.grounding.sanity import SanityResult

    results: list[SanityResult] = []
    rows: list[dict] = []
    detected = 0
    for ecg_id in pool:
        if detected >= n:
            break
        clean, rpeaks, present = load_record(ecg_id, df)
        probs = detect(model, clean)
        # ground the specific finding when the model actually detects it
        cand = ([target] if not is_superclass else
                [c for c in (label_space[j] for j in np.argsort(probs)[::-1]) if c in codes][:1])
        for code in cand:
            j = label_space.index(code)
            if probs[j] < threshold:
                continue
            sal = ground(model, clean, code, label_space=label_space, fs=100)[code]
            res = sanity_check(sal.per_lead, rpeaks, code, fs=100, scp=scp)
            results.append(res)
            rows.append({"ecg_id": int(ecg_id), "label": code, "prob": round(float(sal.prob), 3),
                         "verdict": res.verdict, "detail": res.detail,
                         "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in res.metrics.items()}})
            detected += 1

    agg = summarize(results)
    # mean region masses across the checked records, for the write-up
    checked = [r for r in results if r.verdict != "inconclusive"]
    region_means = {}
    if checked:
        for key in ("P", "QRS", "ST", "T", "baseline", "repolarization"):
            region_means[key] = float(np.mean([r.metrics[key] for r in checked]))
        region_means["rr_cv"] = float(np.nanmean([r.metrics["rr_cv"] for r in checked]))
    return {"target": target, "is_superclass": is_superclass, "n_detected": detected,
            "pool_size": len(pool), "summary": agg, "region_means": region_means, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs" / "final_best.pt"))
    ap.add_argument("--ecg-id", type=int, help="ground a single PTB-XL record")
    ap.add_argument("--label", default=None, help="restrict to one SCP code")
    ap.add_argument("--scan", default=None, help="sanity sweep over a code or superclass (e.g. AFIB, STTC)")
    ap.add_argument("--n", type=int, default=40, help="records to grade in --scan mode")
    ap.add_argument("--threshold", type=float, default=0.5, help="detection prob to count as surfaced")
    ap.add_argument("--no-fig", action="store_true", help="skip figure rendering (single-record mode)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, label_space, ck_args = load_detector(args.checkpoint, device=args.device)
    scp = load_scp_statements()
    df = load_database()
    print(f"loaded {ck_args.get('model') or 'cnn'} checkpoint · {len(df)} records · device={args.device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.scan:
        out = scan(model, label_space, scp, df, args.scan, args.n, args.threshold, args.seed)
        (OUT_DIR / f"scan_{args.scan}.json").write_text(json.dumps(out, indent=2))
        s = out["summary"]
        print(f"\n[{args.scan}] graded {out['n_detected']} detected records "
              f"(pool {out['pool_size']})")
        print(f"  consistent {s['n_consistent']}/{s['n_checked']} "
              f"(rate {s['consistency_rate']:.2f}), inconclusive {s['n_inconclusive']}")
        if out["region_means"]:
            rm = out["region_means"]
            print(f"  mean region mass  P={rm['P']:.2f} QRS={rm['QRS']:.2f} "
                  f"ST={rm['ST']:.2f} T={rm['T']:.2f} baseline={rm['baseline']:.2f}  "
                  f"RR_CV={rm.get('rr_cv', float('nan')):.2f}")
        print(f"  -> {OUT_DIR / f'scan_{args.scan}.json'}")
        return 0

    if args.ecg_id is None:
        ap.error("pass --ecg-id (single record) or --scan CODE (sweep)")
    records = run_single(model, label_space, scp, df, args.ecg_id, args.label,
                         args.threshold, save_fig=not args.no_fig)
    (OUT_DIR / f"ecg{args.ecg_id}.json").write_text(json.dumps(records, indent=2))
    for r in records:
        star = "*" if r["true_label"] else " "
        print(f" {star}{r['label']:6s} p={r['prob']:.2f} lead={r['top_lead']:3s} "
              f"[{r['verdict']}] {r['detail']}")
    print(f"-> {OUT_DIR / f'ecg{args.ecg_id}.json'}"
          + ("" if args.no_fig else f" · figures in {OUT_DIR / 'figures'}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
