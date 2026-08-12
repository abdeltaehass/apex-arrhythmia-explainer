#!/usr/bin/env python3
"""Phase 24 — does the feedback loop actually help?

Runs the online loop end-to-end against a simulated reviewer built from PTB-XL's labels,
in arms that switch one mitigation on at a time. Feedback streams from the **validation**
fold; every score is computed on the **test** fold, so no threshold is chosen and graded on
the same records.

The arms exist to isolate one specific failure. A reviewer can only rate what was shown,
so unaided feedback contains false positives and no false negatives — it can justify
raising a threshold and never lowering one. Arm B measures what that does. Arms C and D add
the two things that put information back into the loop.

    A  no feedback                     static global 0.5 (the shipped default)
    B  ratings only                    the naive loop — expect the ratchet
    C  + reviewers report misses       recall evidence, but no sub-threshold precision data
    D  + exploration                   sample below threshold so it can be lowered again
    E  D with a fallible reviewer      10% error, boundary uncertainty, partial miss reporting

    python scripts/feedback_sim.py                # full run
    python scripts/feedback_sim.py --limit 300    # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MANIFEST_DIR, PTBXL_DIR, ROOT  # noqa: E402
from src.feedback.policy import ThresholdSet  # noqa: E402
from src.feedback.simulate import ArmConfig, ReviewerModel, macro_scores, run_arm  # noqa: E402
from src.feedback.store import FeedbackStore  # noqa: E402

CACHE = ROOT / "data" / "processed" / "feedback"
OUT_JSON = ROOT / "docs" / "feedback" / "simulation.json"
FIG = ROOT / "docs" / "feedback" / "loop.png"


def compute_probs(split: str, checkpoint=None, device: str = "cpu"):
    """Calibrated per-label probabilities and binary truth for a manifest split."""
    import pandas as pd
    import torch
    import wfdb

    from src.data.labels import encode, load_database
    from src.preprocessing.pipeline import preprocess
    from src.serving.model_cache import get_detector

    cache = CACHE / f"{split}.npz"
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        return blob["probs"], blob["y"], list(blob["labels"])

    manifest = pd.read_csv(MANIFEST_DIR / f"{split}.csv")
    db = load_database()
    model, label_space, _ = get_detector(checkpoint, device=device)
    y = np.stack([encode(db.loc[int(e), "scp_codes"], label_space)
                  for e in manifest["ecg_id"]])

    logits = []
    batch: list[np.ndarray] = []
    for ecg_id in manifest["ecg_id"]:
        sig = np.asarray(wfdb.rdsamp(str(PTBXL_DIR / db.loc[int(ecg_id), "filename_lr"]))[0],
                         dtype=np.float32).T
        batch.append(preprocess(sig, fs_in=100, fs_out=100, detect_rpeaks=False)[0])
        if len(batch) == 64:
            with torch.no_grad():
                logits.append(model(torch.from_numpy(np.stack(batch)).to(device)).cpu().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            logits.append(model(torch.from_numpy(np.stack(batch)).to(device)).cpu().numpy())
    logits = np.concatenate(logits)

    # Phase-17 calibration matters here: the policy's prior is the model's own confidence,
    # which is only a claim about precision if the probabilities mean what they say.
    from src.longitudinal.compare import _load_calibrator

    calibrator = _load_calibrator()
    probs = (calibrator.transform(logits) if calibrator is not None
             else 1.0 / (1.0 + np.exp(-logits)))

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probs=probs, y=y, labels=np.array(label_space, dtype=object))
    return probs, y, label_space


def arms() -> list[ArmConfig]:
    honest = ReviewerModel(error_rate=0.0, uncertain_rate=0.02, miss_report_rate=0.6)
    fallible = ReviewerModel(error_rate=0.10, uncertain_rate=0.05,
                             boundary_uncertainty=0.35, miss_report_rate=0.35)
    return [
        ArmConfig("B: ratings only (naive)", collect_missed=False, explore=False,
                  reviewer=honest),
        ArmConfig("C: + reviewers report misses", collect_missed=True, explore=False,
                  reviewer=honest),
        ArmConfig("D: + exploration", collect_missed=True, explore=True, reviewer=honest),
        ArmConfig("E: D, fallible reviewer", collect_missed=True, explore=True,
                  reviewer=fallible),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="cap validation records streamed")
    ap.add_argument("--batch", type=int, default=200, help="records per threshold update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    print("computing calibrated probabilities (cached after the first run)")
    val_probs, val_y, label_space = compute_probs("val", args.checkpoint)
    test_probs, test_y, _ = compute_probs("test", args.checkpoint)
    if args.limit:
        val_probs, val_y = val_probs[: args.limit], val_y[: args.limit]
    print(f"  val {val_probs.shape}  test {test_probs.shape}  labels {len(label_space)}")

    baseline = ThresholdSet()
    p0, r0, f0 = macro_scores(test_probs, test_y, label_space, baseline)
    print(f"\nA: no feedback (static 0.5)   macro P={p0:.4f} R={r0:.4f} F1={f0:.4f}")

    results: dict = {"arm_A_static": {"macro_precision": p0, "macro_recall": r0,
                                      "macro_f1": f0},
                     "n_val": int(len(val_probs)), "n_test": int(len(test_probs)),
                     "batch": args.batch, "arms": {}}

    histories: dict[str, list] = {}
    for arm in arms():
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(Path(tmp) / "sim.db")
            history = run_arm(arm, val_probs, val_y, test_probs, test_y, label_space,
                              store, batch_size=args.batch, seed=args.seed)
            summary = store.summary()
            store.close()
        final = history[-1]
        histories[arm.name] = history
        results["arms"][arm.name] = {
            "final": {"macro_precision": final.macro_precision,
                      "macro_recall": final.macro_recall, "macro_f1": final.macro_f1,
                      "mean_threshold": final.mean_threshold,
                      "n_ratings": final.n_ratings, "n_moved": final.n_moved,
                      "moved_up": final.moved_up, "moved_down": final.moved_down},
            "delta_f1": final.macro_f1 - f0,
            "delta_precision": final.macro_precision - p0,
            "delta_recall": final.macro_recall - r0,
            "store": summary,
            "history": [vars(s) for s in history],
        }
        print(f"\n{arm.name}")
        print(f"  ratings {final.n_ratings:5d}  moved {final.n_moved:3d} "
              f"(up {final.moved_up}, down {final.moved_down})  "
              f"mean threshold {final.mean_threshold:.3f}")
        print(f"  macro P={final.macro_precision:.4f} ({final.macro_precision - p0:+.4f})  "
              f"R={final.macro_recall:.4f} ({final.macro_recall - r0:+.4f})  "
              f"F1={final.macro_f1:.4f} ({final.macro_f1 - f0:+.4f})")

    # --- figure ---------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
        for name, hist in histories.items():
            x = [s.n_reports for s in hist]
            axes[0].plot(x, [s.macro_precision for s in hist], marker="o", ms=3, label=name)
            axes[1].plot(x, [s.macro_recall for s in hist], marker="o", ms=3, label=name)
            axes[2].plot(x, [s.macro_f1 for s in hist], marker="o", ms=3, label=name)
        for ax, (title, base) in zip(axes, [("macro precision", p0), ("macro recall", r0),
                                            ("macro F1", f0)], strict=True):
            ax.axhline(base, ls="--", c="#888", lw=1, label="A: no feedback")
            ax.set_title(title)
            ax.set_xlabel("validation reports reviewed")
            ax.grid(alpha=0.3)
        axes[2].legend(fontsize=7, loc="best")
        fig.suptitle("Phase 24 — online feedback loop, held-out test fold", y=1.02)
        fig.tight_layout()
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {FIG.relative_to(ROOT)}")
    except Exception as e:                                  # noqa: BLE001
        print(f"\n(figure skipped: {e})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
