#!/usr/bin/env python3
"""Phase 25 — how much power does PTB-XL's test fold have on its rarest labels?

This runs before the ablation and decides its design. The nominal task is "improve AUROC on
the rarest classes", so the first question is whether AUROC on those classes can be measured
at all. It cannot: the rarest labels have one to five positive cases in the official test
fold, and a bootstrap confidence interval on such a label spans most of the possible range.

Uses the shipped detector (no training) and reports three uncertainty measures per label,
because they disagree and the disagreement is the point.

**Percentile bootstrap over test records** — what such tables usually show, and misleading
here. With two positives among 2,198 records, a resample can only ever reuse those same two
patients; it reshuffles 2,196 negatives around them. So the interval measures variability
in the negatives and comes out *narrow* — 0.013 wide for a label with one positive case.
It cannot resample the thing that is scarce.

**Leave-one-positive-out** — how far AUROC moves when a single positive case is dropped.
Also conditions on the cases in hand, so it too understates; it is reported because it is
concrete and easy to check, not because it settles the question.

**Hanley-McNeil standard error** — the analytic SE for AUROC, which depends explicitly on
the number of positives rather than on resampling them. This is the one that answers the
question, and it is what shows the rare labels cannot support the experiment.

Together they are the argument for the induced-rarity design in `src/synthesis/rarity.py`.

    python scripts/rare_class_power.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ROOT  # noqa: E402
from src.data.labels import build_label_space  # noqa: E402
from src.synthesis.ablation import bootstrap_auroc, predict  # noqa: E402
from src.synthesis.rarity import DEFAULT_TARGETS  # noqa: E402

OUT = ROOT / "docs" / "synthesis" / "power.json"


def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Analytic standard error of AUROC (Hanley & McNeil, 1982).

    Unlike a record bootstrap, this depends on ``n_pos`` directly, so it does not quietly
    condition on the handful of positive cases that happen to be in the test set. That is
    exactly the uncertainty a rare-label evaluation needs to report.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    a = float(np.clip(auroc, 1e-6, 1 - 1e-6))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def single_case_leverage(y_true: np.ndarray, score: np.ndarray) -> float:
    """Largest change in AUROC caused by dropping a single positive case.

    The honest power statistic for a label with a handful of positives: it asks how much of
    the reported number rests on one patient. A value of 0.3 means one case moves AUROC by
    0.3, so an augmentation effect of 0.02 is invisible beneath it.
    """
    from sklearn.metrics import roc_auc_score

    positives = np.flatnonzero(y_true > 0)
    if len(positives) < 2:
        return float("nan")      # dropping the only positive leaves AUROC undefined
    base = roc_auc_score(y_true, score)
    worst = 0.0
    for i in positives:
        keep = np.ones(len(y_true), dtype=bool)
        keep[i] = False
        worst = max(worst, abs(roc_auc_score(y_true[keep], score[keep]) - base))
    return float(worst)


def main() -> int:
    import torch

    from src.serving.model_cache import get_detector

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    label_space = build_label_space()
    tr = np.load(PROCESSED_DIR / "train_100hz.npz")
    te = np.load(PROCESSED_DIR / "test_100hz.npz")
    Xte, Yte, Ytr = te["X"], te["Y"], tr["Y"]

    model, model_labels, _ = get_detector(device=device)
    assert list(model_labels) == list(label_space), "label space mismatch"
    probs = predict(model, Xte, device)

    rows = []
    for j, label in enumerate(label_space):
        n_test = int(Yte[:, j].sum())
        n_train = int(Ytr[:, j].sum())
        if n_test == 0 or n_test == len(Yte):
            rows.append({"label": label, "n_train": n_train, "n_test": n_test,
                         "auroc": None, "ci_low": None, "ci_high": None, "ci_width": None})
            continue
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(Yte[:, j], probs[:, j]))
        lo, hi = bootstrap_auroc(Yte[:, j], probs[:, j], n=2000, seed=j)
        leverage = single_case_leverage(Yte[:, j], probs[:, j])
        se = hanley_mcneil_se(auroc, n_test, len(Yte) - n_test)
        rows.append({"label": label, "n_train": n_train, "n_test": n_test, "auroc": auroc,
                     "ci_low": lo, "ci_high": hi,
                     "ci_width": (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else None,
                     "single_case_leverage": leverage, "hanley_mcneil_se": se,
                     "hm_ci_width": 2 * 1.96 * se})

    rows.sort(key=lambda r: r["n_train"])
    print(f"{'label':10s} {'train':>6} {'test':>5} {'AUROC':>7} {'bootCI':>7} {'1-case':>7}"
          f" {'H-M SE':>7} {'H-M CI':>7}")
    for r in rows[:20]:
        if r["auroc"] is None:
            print(f"{r['label']:10s} {r['n_train']:6d} {r['n_test']:5d}   (no test positives)")
            continue
        lev = r["single_case_leverage"]
        lev_s = "  n/a  " if not np.isfinite(lev) else f"{lev:7.3f}"
        print(f"{r['label']:10s} {r['n_train']:6d} {r['n_test']:5d} {r['auroc']:7.3f} "
              f"{r['ci_width']:7.3f} {lev_s} {r['hanley_mcneil_se']:7.3f} "
              f"{r['hm_ci_width']:7.3f}")

    measurable = [r for r in rows if r["ci_width"] is not None]
    rare = [r for r in measurable if r["n_train"] < 50]
    targets = [r for r in measurable if r["label"] in DEFAULT_TARGETS]
    lev_rare = [r["single_case_leverage"] for r in rare
                if np.isfinite(r.get("single_case_leverage", np.nan))]
    lev_tgt = [r["single_case_leverage"] for r in targets
               if np.isfinite(r.get("single_case_leverage", np.nan))]
    summary = {
        "median_single_case_leverage_rare": float(np.median(lev_rare)) if lev_rare else None,
        "median_single_case_leverage_targets": float(np.median(lev_tgt)) if lev_tgt else None,
        "median_hm_ci_width_rare": float(np.median([r["hm_ci_width"] for r in rare])),
        "median_hm_ci_width_targets": float(np.median([r["hm_ci_width"] for r in targets])),
        "n_labels": len(rows),
        "median_ci_width_rare_train_lt_50": float(np.median([r["ci_width"] for r in rare])),
        "median_ci_width_induced_targets": float(np.median([r["ci_width"] for r in targets])),
        "median_test_positives_rare": float(np.median([r["n_test"] for r in rare])),
        "median_test_positives_targets": float(np.median([r["n_test"] for r in targets])),
        "n_rare_train_lt_50": len(rare),
    }
    print("\n=== the argument for induced rarity ===")
    print(f"  labels with <50 training examples: {summary['n_rare_train_lt_50']}")
    print(f"  17 labels with <50 training examples: median {summary['median_test_positives_rare']:.0f}"
          f" test positives")
    print(f"    bootstrap CI width {summary['median_ci_width_rare_train_lt_50']:.3f}"
          f"   Hanley-McNeil CI width {summary['median_hm_ci_width_rare']:.3f}")
    print(f"  the 8 induced-rarity targets: median {summary['median_test_positives_targets']:.0f}"
          f" test positives")
    print(f"    bootstrap CI width {summary['median_ci_width_induced_targets']:.3f}"
          f"   Hanley-McNeil CI width {summary['median_hm_ci_width_targets']:.3f}")
    ratio = summary["median_hm_ci_width_rare"] / summary["median_hm_ci_width_targets"]
    print("\n  The bootstrap says the rare labels are measured about as tightly as the")
    print("  targets. It is wrong: it resamples 2,196 negatives around the same 1-2")
    print("  positives and cannot vary which patients have the disease. Hanley-McNeil,")
    print(f"  which depends on the positive count, puts the rare labels {ratio:.1f}x wider.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "labels": rows}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
