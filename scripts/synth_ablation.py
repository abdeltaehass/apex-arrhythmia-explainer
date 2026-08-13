#!/usr/bin/env python3
"""Phase 25 — synthetic ECG augmentation for rare classes: the ablation.

Induces rarity on labels that have enough test support to measure (see
`src/synthesis/rarity.py` for why the genuinely rarest labels cannot be used), trains a
conditional diffusion model on the *masked* training labels, and compares five ways of
adding rare-class examples.

Per seed: mask -> train generator -> sample -> train five classifiers -> score on test.
The generator is retrained per seed on that seed's masked labels; reusing one across seeds
would leak the withheld annotations into every arm that samples from it.

    python scripts/synth_ablation.py                     # full run (hours)
    python scripts/synth_ablation.py --smoke             # tiny end-to-end check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, ROOT  # noqa: E402
from src.data.labels import build_label_space  # noqa: E402
from src.synthesis.ablation import (  # noqa: E402
    ARMS,
    TrainConfig,
    build_arm_data,
    evaluate,
    paired_bootstrap_delta,
    predict,
    sample_for_targets,
    train_classifier,
)
from src.synthesis.diffusion import DiffusionConfig, train_diffusion  # noqa: E402
from src.synthesis.quality import assess  # noqa: E402
from src.synthesis.rarity import DEFAULT_TARGETS, make_rare  # noqa: E402

OUT = ROOT / "docs" / "synthesis"


def load(split: str):
    d = np.load(PROCESSED_DIR / f"{split}_100hz.npz")
    return d["X"], d["Y"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-keep", type=int, default=50, help="real examples kept per target")
    ap.add_argument("--gen-epochs", type=int, default=25)
    ap.add_argument("--clf-epochs", type=int, default=10)
    ap.add_argument("--n-synth", type=int, default=200, help="unique samples per target")
    ap.add_argument("--n-added", type=int, default=350, help="rows each arm adds per target")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.gen_epochs, args.clf_epochs = 1, 1, 1
        args.n_synth, args.n_added = 8, 16

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    label_space = build_label_space()
    Xtr, Ytr = load("train")
    Xte, Yte = load("test")
    Xva, Yva = load("val")
    if args.smoke:
        Xtr, Ytr = Xtr[:600], Ytr[:600]
        Xte, Yte = Xte[:300], Yte[:300]
    targets = tuple(DEFAULT_TARGETS)
    print(f"device={device}  train={Xtr.shape}  test={Xte.shape}  targets={len(targets)}")

    tcfg = TrainConfig(epochs=args.clf_epochs, n_added=args.n_added)
    dcfg = DiffusionConfig(epochs=args.gen_epochs)

    results: dict = {"config": {"seeds": args.seeds, "n_keep": args.n_keep,
                                "gen_epochs": args.gen_epochs, "clf_epochs": args.clf_epochs,
                                "n_synth": args.n_synth, "n_added": args.n_added,
                                "targets": list(targets), "device": device},
                     "seeds": [], "quality": []}
    # arm -> label -> list of test-probability vectors, for the paired bootstrap
    probs_by_arm: dict[str, dict[int, list[np.ndarray]]] = {a: {} for a in ARMS}

    for seed in range(args.seeds):
        t0 = time.perf_counter()
        print(f"\n=== seed {seed} ===", flush=True)
        scenario = make_rare(Ytr, label_space, targets, n_keep=args.n_keep, seed=seed)
        print("  masked: " + ", ".join(f"{k} {len(v)} kept" for k, v in scenario.kept.items()),
              flush=True)

        print("  training diffusion generator on MASKED labels...", flush=True)
        gen = train_diffusion(Xtr, scenario.y_masked, dcfg, device=device, seed=seed,
                              verbose=True)
        print("  sampling...", flush=True)
        synthetic = sample_for_targets(gen, scenario, label_space, args.n_synth, dcfg,
                                       device, seed)

        for label, (batch, _cond) in synthetic.items():
            j = label_space.index(label)
            real = Xtr[scenario.positive_rows(label)]
            heldout = Xva[Yva[:, j] > 0]
            if len(heldout) < 3:
                continue
            q = assess(batch, real, heldout)
            results["quality"].append({"seed": seed, "label": label, **q.as_dict()})
            print(f"    {label:8s} memorization={q.memorization_ratio:.2f} "
                  f"diversity={q.diversity_ratio:.2f} "
                  f"P-wave {q.p_detected_rate:.0%} measurable {q.measurable_rate:.0%}",
                  flush=True)

        seed_result: dict = {"seed": seed, "arms": {}}
        rng = np.random.default_rng(1000 + seed)
        for arm in ARMS:
            Xa, Ya = build_arm_data(arm, Xtr, scenario, label_space, synthetic, tcfg, rng)
            model = train_classifier(Xa, Ya, tcfg, device, seed)
            probs = predict(model, Xte, device)
            per, common = evaluate(probs, Yte, label_space, targets)
            seed_result["arms"][arm] = {"auroc": per, "auroc_common": common,
                                        "n_rows": int(len(Xa))}
            for label in per:
                probs_by_arm[arm].setdefault(label_space.index(label), []).append(
                    probs[:, label_space.index(label)])
            print(f"  {arm:22s} rows={len(Xa):6d}  targets={np.mean(list(per.values())):.4f}"
                  f"  others={common:.4f}", flush=True)
        results["seeds"].append(seed_result)
        print(f"  seed done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    # --- aggregate ------------------------------------------------------------
    summary: dict = {}
    for arm in ARMS:
        per_label: dict[str, float] = {}
        for label in targets:
            vals = [s["arms"][arm]["auroc"].get(label) for s in results["seeds"]
                    if label in s["arms"][arm]["auroc"]]
            vals = [v for v in vals if v is not None]
            if vals:
                per_label[label] = float(np.mean(vals))
        commons = [s["arms"][arm]["auroc_common"] for s in results["seeds"]]
        summary[arm] = {
            "auroc_by_target": per_label,
            "macro_targets": float(np.mean(list(per_label.values()))) if per_label else float("nan"),
            "macro_common": float(np.mean(commons)),
            "macro_common_sd": float(np.std(commons, ddof=1)) if len(commons) > 1 else 0.0,
        }

    # Paired bootstrap against baseline, on seed-averaged scores.
    deltas: dict = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        rows = {}
        for label in targets:
            j = label_space.index(label)
            if j not in probs_by_arm["baseline"] or j not in probs_by_arm[arm]:
                continue
            base = np.mean(probs_by_arm["baseline"][j], axis=0)
            cand = np.mean(probs_by_arm[arm][j], axis=0)
            mean, lo, hi = paired_bootstrap_delta(Yte[:, j], base, cand, n=1000, seed=j)
            rows[label] = {"delta": mean, "ci_low": lo, "ci_high": hi,
                           "significant": bool(lo > 0 or hi < 0)}
        deltas[arm] = rows
    results["summary"] = summary
    results["deltas_vs_baseline"] = deltas

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n=== macro AUROC over the 8 induced-rare targets (mean of seeds) ===")
    for arm in ARMS:
        s = summary[arm]
        print(f"  {arm:22s} targets={s['macro_targets']:.4f}   "
              f"others={s['macro_common']:.4f} (sd {s['macro_common_sd']:.4f})")
    print("\n=== per-target delta vs baseline, paired bootstrap 95% CI ===")
    for arm, rows in deltas.items():
        sig = sum(1 for r in rows.values() if r["significant"])
        print(f"  {arm:22s} {sig}/{len(rows)} labels with a CI excluding zero")
    print(f"\nwrote {(OUT / 'ablation.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
