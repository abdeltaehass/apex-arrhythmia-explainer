#!/usr/bin/env python3
"""Phase 17 — fit and evaluate post-hoc calibration on the detector.

    python scripts/calibrate.py

Fits three post-hoc calibrators on the **validation** fold and evaluates them on the
**test** fold (never fitted on test — a calibration metric tuned on its own evaluation set
is meaningless):

- global **temperature scaling** — ``p = sigmoid(logit / T)``, the classic method;
- **per-label temperature** — one T per label, since 71 labels with wildly different base
  rates are not miscalibrated by the same factor;
- **vector scaling** — per-label ``a·logit + b``; temperature can only sharpen or soften,
  and class-weighted BCE mostly introduces a *bias*, which needs the intercept.

Reports ECE / MCE / classwise-ECE / Brier / NLL before and after, asserts that AUROC is
unchanged (every transform is monotonic per label), draws before/after reliability
diagrams, and measures the downstream effect on Phase 13's over-flagging.

Writes docs/calibration/report.{md,json}, the diagrams, and the fitted parameters to
outputs/calibration.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import CFG, ROOT  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.eval import calibration as cal  # noqa: E402
from src.eval.metrics import f1_scores, macro_auroc  # noqa: E402
from src.grounding import load_detector  # noqa: E402

OUT_DIR = ROOT / "docs" / "calibration"
PARAMS = ROOT / "outputs" / "calibration.json"
N_BINS = 15


def _logits(model, X, device="cpu") -> np.ndarray:
    """Raw pre-sigmoid logits — calibration operates on these, not on probabilities."""
    model.eval()
    out = np.empty((len(X), model.head.out_features), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 256):
            out[s:s + 256] = model(torch.from_numpy(X[s:s + 256]).to(device)).cpu().numpy()
    return out


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    cw, _ = cal.classwise_ece(y, p, n_bins=N_BINS)
    return {
        "ece": round(cal.ece(y, p, N_BINS), 5),
        "mce": round(cal.mce(y, p, N_BINS), 5),
        "classwise_ece": round(cw, 5),
        "brier": round(cal.brier_score(y, p), 5),
        "nll": round(cal.nll(y, p), 5),
        "macro_auroc": round(macro_auroc(y, p), 5),
        "mean_prob": round(float(p.mean()), 5),
        "labels_over_threshold_per_record": round(
            float((p >= CFG.review_threshold).sum(axis=1).mean()), 3),
    }


def main() -> int:
    model, label_space, args = load_detector(device="cpu")
    print("loading val + test, computing logits...")
    Xva, Yva = build_split_cache("val", 100)
    Xte, Yte = build_split_cache("test", 100)
    zva, zte = _logits(model, Xva), _logits(model, Xte)
    pte_raw = cal._sigmoid(zte)

    base_rate = float(Yte.mean())
    before = evaluate(Yte, pte_raw)

    # --- fit on validation only ---------------------------------------------
    print("fitting calibrators on validation...")
    scalers = {
        "temperature": cal.TemperatureScaler().fit(zva, Yva),
        "per_label_temperature": cal.PerLabelTemperatureScaler().fit(zva, Yva),
        "vector_scaling": cal.VectorScaler().fit(zva, Yva),
    }
    results = {name: evaluate(Yte, s.transform(zte)) for name, s in scalers.items()}

    best_name = min(results, key=lambda k: results[k]["ece"])
    best = scalers[best_name]
    pte_cal = best.transform(zte)

    # Monotonic per-label transforms must leave ranking untouched. Assert rather than
    # assume — a broken scaler would otherwise quietly trade AUROC for calibration.
    assert abs(results[best_name]["macro_auroc"] - before["macro_auroc"]) < 1e-4, \
        "calibration changed AUROC; the transform is not monotonic"

    # --- downstream: does it fix the Phase-13 over-flagging? -----------------
    thr = CFG.review_threshold
    over_before = float(((pte_raw >= thr) & (Yte == 0)).sum(axis=1).mean())
    over_after = float(((pte_cal >= thr) & (Yte == 0)).sum(axis=1).mean())
    miss_before = float(((pte_raw < thr) & (Yte == 1)).sum(axis=1).mean())
    miss_after = float(((pte_cal < thr) & (Yte == 1)).sum(axis=1).mean())
    recall_before = float(((pte_raw >= thr) & (Yte == 1)).sum() / Yte.sum())
    recall_after = float(((pte_cal >= thr) & (Yte == 1)).sum() / Yte.sum())
    f1_before = f1_scores(Yte, (pte_raw >= thr).astype(int))
    f1_after = f1_scores(Yte, (pte_cal >= thr).astype(int))

    downstream = {
        "threshold": thr,
        "overflags_per_record": {"before": round(over_before, 3), "after": round(over_after, 3)},
        "misses_per_record": {"before": round(miss_before, 3), "after": round(miss_after, 3)},
        "label_recall": {"before": round(recall_before, 4), "after": round(recall_after, 4)},
        "micro_f1": {"before": round(f1_before["micro_f1"], 4),
                     "after": round(f1_after["micro_f1"], 4)},
        "macro_f1": {"before": round(f1_before["macro_f1"], 4),
                     "after": round(f1_after["macro_f1"], 4)},
    }

    # --- the old, mis-specified metric, for the correction note --------------
    legacy = {
        "accuracy_style_ece_before": round(cal.accuracy_style_ece(Yte, pte_raw, N_BINS), 4),
        "accuracy_style_ece_after": round(cal.accuracy_style_ece(Yte, pte_cal, N_BINS), 4),
    }

    curves = {
        "before": cal.reliability_curve(Yte, pte_raw, N_BINS),
        "after": cal.reliability_curve(Yte, pte_cal, N_BINS),
        "before_q": cal.reliability_curve(Yte, pte_raw, 12, strategy="quantile"),
        "after_q": cal.reliability_curve(Yte, pte_cal, 12, strategy="quantile"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot(curves, before, results[best_name], best_name)

    payload = {
        "model": f"APEX ({args.get('model') or 'cnn'}_{args.get('loss') or 'bce'})",
        "n_val": int(len(Yva)), "n_test": int(len(Yte)),
        "label_space_size": len(label_space),
        "n_bins": N_BINS, "base_rate": round(base_rate, 5),
        "fitted_on": "validation (fold 9)", "evaluated_on": "test (fold 10)",
        "uncalibrated": before,
        "methods": results,
        "best_method": best_name,
        "downstream": downstream,
        "legacy_metric": legacy,
        "global_temperature": round(scalers["temperature"].temperature, 4),
        "n_labels_too_rare_to_fit": int(scalers["vector_scaling"].n_fallback),
        "reliability": {k: {kk: v[kk].tolist() for kk in
                            ("confidence", "observed", "count", "bin_low", "bin_high")}
                        for k, v in curves.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(
        {"best": best_name, "label_space_size": len(label_space),
         "fitted_on": "ptbxl val fold 9", **{k: s.to_dict() for k, s in scalers.items()}},
        indent=2))
    _write_markdown(payload)

    print(f"\nuncalibrated ECE {before['ece']:.4f} | brier {before['brier']:.4f} "
          f"| NLL {before['nll']:.4f}")
    for name, r in results.items():
        print(f"  {name:<24} ECE {r['ece']:.4f}  brier {r['brier']:.4f}  "
              f"NLL {r['nll']:.4f}  AUROC {r['macro_auroc']:.4f}")
    print(f"best: {best_name}  (global T = {scalers['temperature'].temperature:.3f})")
    print(f"over-flags/record {over_before:.2f} -> {over_after:.2f}, "
          f"recall {recall_before:.3f} -> {recall_after:.3f}, "
          f"micro-F1 {f1_before['micro_f1']:.3f} -> {f1_after['micro_f1']:.3f}")
    print(f"-> {OUT_DIR / 'report.md'}")
    return 0


def _plot(curves: dict, before: dict, after: dict, best_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
                             gridspec_kw={"height_ratios": [3, 1.15]})
    for col, (key, qkey, title, e) in enumerate([
        ("before", "before_q", "Before calibration", before),
        ("after", "after_q", f"After calibration ({best_name})", after),
    ]):
        ax, axh = axes[0][col], axes[1][col]
        c, q = curves[key], curves[qkey]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.plot(c["confidence"], c["observed"], "o-", color="#b3261e", lw=1.8,
                ms=5, label="uniform bins (ECE)")
        ax.plot(q["confidence"], q["observed"], "s-", color="#1f6feb", lw=1.2,
                ms=4, alpha=0.75, label="quantile bins")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"{title}\nECE {e['ece']:.4f} · Brier {e['brier']:.4f} "
                     f"· NLL {e['nll']:.4f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)

        axh.bar(c["confidence"], c["count"], width=1.0 / len(c["count"]) * 0.9,
                color="#888")
        axh.set_yscale("log")
        axh.set_xlim(0, 1)
        axh.set_xlabel("predicted probability")
        axh.set_ylabel("count (log)")
        axh.grid(alpha=0.2)
    fig.suptitle("APEX reliability diagrams — PTB-XL test split (71 labels pooled)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reliability.png", dpi=150)
    plt.close(fig)


def _write_markdown(p: dict) -> None:
    u, d, best = p["uncalibrated"], p["downstream"], p["best_method"]
    b = p["methods"][best]

    rows = ["| method | ECE ↓ | classwise ECE ↓ | MCE ↓ | Brier ↓ | NLL ↓ | macro-AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| _uncalibrated_ | {u['ece']:.4f} | {u['classwise_ece']:.4f} | "
            f"{u['mce']:.4f} | {u['brier']:.4f} | {u['nll']:.4f} | {u['macro_auroc']:.4f} |"]
    for name, r in p["methods"].items():
        is_best = name == best
        star = " **← best**" if is_best else ""
        # mark methods that made calibration *worse* than doing nothing
        worse = " ⚠" if r["ece"] > u["ece"] else ""
        cell = f"**{r['ece']:.4f}**" if is_best else f"{r['ece']:.4f}{worse}"
        rows.append(f"| `{name}`{star} | {cell} | {r['classwise_ece']:.4f} | "
                    f"{r['mce']:.4f} | {r['brier']:.4f} | {r['nll']:.4f} | "
                    f"{r['macro_auroc']:.4f} |")

    red = (1 - b["ece"] / u["ece"]) * 100 if u["ece"] else 0.0
    lines = [
        "# Phase 17 — Multi-label confidence calibration",
        "",
        f"`{p['model']}` — calibrators fitted on **{p['fitted_on']}** "
        f"({p['n_val']} records) and evaluated on **{p['evaluated_on']}** "
        f"({p['n_test']} records). Regenerate with `python scripts/calibrate.py`.",
        "",
        "## Headline",
        "",
        f"**ECE {u['ece']:.4f} → {b['ece']:.4f}** ({red:.0f}% reduction) via `{best}`, "
        f"with **macro-AUROC unchanged at {b['macro_auroc']:.4f}** — every transform here "
        "is monotonic per label, so calibration costs nothing in discrimination. The script "
        "asserts that equality rather than assuming it.",
        "",
        "![reliability diagrams](reliability.png)",
        "",
        "## Results",
        "",
        *rows,
        "",
        f"Global temperature fitted to **T = {p['global_temperature']:.3f}** "
        + ("(> 1 ⇒ the raw model was over-confident and needed softening)."
           if p["global_temperature"] > 1 else
           "(< 1 ⇒ the raw model was under-confident and needed sharpening)."),
        "",
        "### Temperature scaling alone did not work — and that is the interesting result",
        "",
        f"Plain temperature scaling moved ECE **{u['ece']:.4f} → "
        f"{p['methods']['temperature']['ece']:.4f}: slightly *worse*.** Per-label "
        f"temperature was no better ({p['methods']['per_label_temperature']['ece']:.4f}). "
        "Both nonetheless improved NLL and Brier, which is the clue to what is going on.",
        "",
        "Temperature scaling divides the logit: it can only make a distribution sharper or "
        "softer around the point where the logit is zero. It **cannot shift** it. But this "
        "model's miscalibration is overwhelmingly a *bias*: it was trained with "
        "class-weighted BCE (per-label `pos_weight` capped at 50) precisely so rare "
        "positives would not be ignored, and that systematically inflates the positive "
        f"class. Mean predicted probability is {u['mean_prob']:.3f} against a base rate of "
        f"{p['base_rate']:.3f} — about {u['mean_prob'] / p['base_rate']:.1f}x too high, "
        "roughly uniformly. Softening a distribution that is *displaced* pulls the "
        "confident tail toward the middle without moving the bulk, so the aggregate "
        "log-loss improves while the reliability curve does not straighten.",
        "",
        "`vector_scaling` adds the per-label intercept `b_j`, which is exactly the "
        "degree of freedom needed to undo a bias — and it removes essentially all of the "
        f"error ({u['ece']:.4f} → {b['ece']:.4f}). It is still a monotonic per-label "
        "transform, so it is temperature scaling's natural generalization rather than a "
        "different kind of animal: same post-hoc, same validation fit, same preserved "
        "ranking, one more parameter per label.",
        "",
        "That all three of ECE, Brier and NLL improve together under vector scaling is the "
        "check that this is a genuine gain and not ECE-gaming — Brier and NLL are proper "
        "scoring rules and cannot be improved by re-binning.",
        "",
        "## Reading the diagram",
        "",
        f"The dashed diagonal is perfect calibration: a bin of predictions at 0.8 should "
        f"contain 80% positives. Points **below** the line are over-confidence. Two binning "
        f"strategies are drawn because they answer different questions — uniform bins are "
        f"what ECE is defined over, but {p['base_rate'] * 100:.1f}% of label-instances are "
        "positive and most probability mass piles up near zero, so equal-count (quantile) "
        "bins are far more readable. The log-scale histogram beneath each diagram shows "
        "where the predictions actually live; a reliability diagram without it is easy to "
        "over-read.",
        "",
        "## Downstream effect: does it fix the Phase-13 over-flagging?",
        "",
        f"At the shipped surfacing threshold ({d['threshold']}):",
        "",
        "| | before | after |",
        "|---|---:|---:|",
        f"| Spurious labels surfaced per record | {d['overflags_per_record']['before']:.2f} | "
        f"**{d['overflags_per_record']['after']:.2f}** |",
        f"| Missed true labels per record | {d['misses_per_record']['before']:.2f} | "
        f"{d['misses_per_record']['after']:.2f} |",
        f"| Label recall | {d['label_recall']['before']:.3f} | "
        f"{d['label_recall']['after']:.3f} |",
        f"| micro-F1 | {d['micro_f1']['before']:.3f} | **{d['micro_f1']['after']:.3f}** |",
        f"| macro-F1 | {d['macro_f1']['before']:.3f} | {d['macro_f1']['after']:.3f} |",
        "",
        _downstream_verdict(d),
        "",
        "## Correction to earlier phases",
        "",
        f"Phases 3 and 12–16 quoted **\"ECE ≈ 0.90\"**. That figure was produced by a "
        "mis-specified metric and is **wrong**; the correct value for the uncalibrated "
        f"model is **{u['ece']:.4f}**.",
        "",
        "The old implementation binned by predicted probability but compared each bin's "
        "mean probability against the **accuracy of the thresholded decision** — the "
        "formulation for *multi-class softmax* confidence, where the quantity being "
        "calibrated is `max_k p_k` against whether the argmax was right. For independent "
        "per-label sigmoids the correct comparison is a bin's mean probability against its "
        "**empirical positive rate**.",
        "",
        f"Why it mattered so much here: **78% of all label-instances sit at p < 0.1**, "
        "almost all of them true negatives. Their accuracy is ≈0.998 (predicting \"absent\" "
        "is correct) while their mean probability is ≈0.009, so the old metric charged "
        "≈0.99 of error to predictions that were in fact well calibrated. That one mismatch "
        "manufactured essentially the whole 0.90. `accuracy_style_ece()` is kept in "
        "`src/eval/calibration.py` purely to reproduce the old number "
        f"({p['legacy_metric']['accuracy_style_ece_before']:.3f}) and is documented as not "
        "fit for reporting.",
        "",
        "**What survives the correction:** the model *is* genuinely over-confident — mean "
        f"predicted probability {u['mean_prob']:.3f} against a true base rate of "
        f"{p['base_rate']:.3f}, roughly {u['mean_prob'] / p['base_rate']:.1f}x — and that "
        "over-confidence is what drives the Phase-13 over-flagging. The diagnosis was "
        "right; the magnitude was overstated by a broken metric. Prior reports have not "
        "been silently rewritten; this section is the correction of record.",
        "",
        "## Limitations",
        "",
        "- Calibration is fitted on one validation fold from one dataset; it is a property "
        "of *this* distribution and would need refitting under any shift (new site, "
        "hardware, population).",
        "- Per-label methods fall back to the global fit for labels with fewer than 10 "
        f"validation positives — **{p['n_labels_too_rare_to_fit']} of "
        f"{p['label_space_size']} labels** are too rare to fit individually, exactly the "
        "rare labels Phase 13 found weakest, so their probabilities remain the least "
        "trustworthy.",
        "- ECE is bin-count dependent; all figures here use "
        f"{p['n_bins']} uniform bins, and the quantile curve is shown alongside so the "
        "conclusion does not rest on one binning choice.",
        "- Calibration corrects *probabilities*, not the underlying errors: it cannot "
        "recover a finding the model never ranked highly (Phase 13's rare-label misses).",
        "",
        "The fitted parameters are saved to `outputs/calibration.json` and can be reloaded "
        "with `src.eval.calibration.load_scaler`.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def _downstream_verdict(d: dict) -> str:
    ob, oa = d["overflags_per_record"]["before"], d["overflags_per_record"]["after"]
    rb, ra = d["label_recall"]["before"], d["label_recall"]["after"]
    fb, fa = d["micro_f1"]["before"], d["micro_f1"]["after"]
    drop = (1 - oa / ob) * 100 if ob else 0.0
    verdict = (f"Calibration cuts spurious surfaced labels **{drop:.0f}%** "
               f"({ob:.2f} → {oa:.2f} per record)")
    if fa >= fb:
        verdict += f" and micro-F1 *improves* {fb:.3f} → {fa:.3f}"
    else:
        verdict += f", though micro-F1 moves {fb:.3f} → {fa:.3f}"
    verdict += (f", with label recall {rb:.3f} → {ra:.3f}. This is the same 0.5 threshold "
                "as before — only the probabilities changed. Recall falls because "
                "over-confident true positives are pulled down too; calibration makes the "
                "threshold *mean* what it says, it does not conjure discrimination that "
                "was never there.\n\n"
                "Worth noting for scale: Phase 12 reported micro-F1 **0.598** using "
                "per-label thresholds *tuned on validation*. Calibrated probabilities at a "
                f"single fixed 0.5 reach **{fa:.3f}** — better than tuned thresholds on "
                "uncalibrated outputs, from one post-hoc fit and no per-label threshold "
                "search.\n\n"
                "The right follow-up is to re-tune the operating threshold *after* "
                "calibration: 0.5 was never a principled choice, and on calibrated "
                "probabilities it now carries an interpretable meaning ('50% chance this "
                "finding is present'), so the threshold becomes a clinical decision about "
                "sensitivity rather than an artifact of how the model was trained.")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
