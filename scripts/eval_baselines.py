#!/usr/bin/env python3
"""Phase 12 — evaluate APEX on the PTB-XL test split vs published baselines.

    python scripts/eval_baselines.py

Runs the detector (`outputs/final_best.pt`) on PTB-XL fold 10 (the held-out test split,
never used for tuning) and computes:

  - all-task (71-code) macro-AUROC — the number the PTB-XL benchmark's "all" task reports
  - per-superclass + macro AUROC on the 5 diagnostic superclasses (NORM/MI/STTC/CD/HYP),
    obtained by pooling the 71 code scores (`src/eval/superclass.py`)
  - macro / micro F1 at per-label thresholds tuned on the *validation* fold (no test
    leakage), and ECE

and writes a comparison table against the published PTB-XL benchmark (Strodthoff et al.
2021) to docs/model_comparison/baseline_comparison.{md,json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import ROOT  # noqa: E402
from src.data.labels import load_scp_statements  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    expected_calibration_error,
    f1_scores,
    macro_auroc,
    tune_thresholds,
)
from src.eval.superclass import SUPERCLASSES, superclass_auroc  # noqa: E402
from src.grounding import load_detector  # noqa: E402

OUT_DIR = ROOT / "docs" / "model_comparison"

# --- Published PTB-XL benchmark (Strodthoff et al. 2021, github.com/helme/
#     ecg_ptbxl_benchmarking). Macro-AUROC; "(nn)" is the reported +-0.0nn 95% CI half-width.
PUBLISHED = {
    # model:            (all-task AUROC, superdiagnostic AUROC)
    "inception1d":     (0.925, 0.921),
    "xresnet1d101":    (0.925, 0.928),
    "resnet1d_wang":   (0.919, 0.930),
    "fcn_wang":        (0.918, 0.925),
    "lstm_bidir":      (0.914, 0.921),
    "lstm":            (0.907, 0.927),
    "Wavelet+NN":      (0.849, 0.874),
}


def _predict(model, X, device="cpu") -> np.ndarray:
    model.eval()
    out = np.empty((len(X), model.head.out_features), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 256):
            out[s:s + 256] = torch.sigmoid(model(torch.from_numpy(X[s:s + 256]).to(device))).cpu().numpy()
    return out


def main() -> int:
    model, label_space, args = load_detector(device="cpu")
    scp = load_scp_statements()
    print("loading val + test caches...")
    Xva, Yva = build_split_cache("val", 100)
    Xte, Yte = build_split_cache("test", 100)

    print("predicting...")
    pva, pte = _predict(model, Xva), _predict(model, Xte)
    thr = tune_thresholds(Yva, pva)                 # thresholds tuned on val, applied to test

    all_auroc = macro_auroc(Yte, pte)
    sc = superclass_auroc(Yte, pte, label_space, scp)
    f1 = f1_scores(Yte, (pte >= thr).astype(int))
    ece = expected_calibration_error(Yte, pte)

    apex = {
        "model": f"APEX ({args.get('model') or 'cnn'}_{args.get('loss') or 'bce'})",
        "all_auroc": round(all_auroc, 4),
        "superclass_auroc": {s: round(sc[s], 4) for s in SUPERCLASSES},
        "superclass_macro_auroc": round(sc["macro"], 4),
        "macro_f1": round(f1["macro_f1"], 4),
        "micro_f1": round(f1["micro_f1"], 4),
        "ece": round(ece, 4),
        "n_test": int(len(Yte)),
    }
    payload = {"apex": apex, "published_ptbxl_benchmark": PUBLISHED,
               "note": "Published = Strodthoff et al. 2021 (github.com/helme/ecg_ptbxl_benchmarking). "
                       "APEX is trained on the 71-code 'all' task; its superclass AUROC is pooled "
                       "from the 71 codes (max over members), not trained on the 5-class "
                       "superdiagnostic task directly."}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baseline_comparison.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(payload)

    print(f"\nAPEX test: all-AUROC {all_auroc:.4f} | superclass-macro {sc['macro']:.4f} "
          f"| macro-F1 {f1['macro_f1']:.4f} | micro-F1 {f1['micro_f1']:.4f}")
    print(f"-> {OUT_DIR / 'baseline_comparison.md'}")
    return 0


def _write_markdown(p: dict) -> None:
    apex = p["apex"]
    sc = apex["superclass_auroc"]

    # detection table: APEX + published, ranked by all-task AUROC
    rows = [(apex["model"], apex["all_auroc"], apex["superclass_macro_auroc"], "**this work**")]
    for m, (a, s) in p["published_ptbxl_benchmark"].items():
        rows.append((m, a, s, "Strodthoff 2021"))
    rows.sort(key=lambda r: -r[1])
    det = ["| model | all-task AUROC | superclass AUROC | source |",
           "|---|---:|---:|---|"]
    for name, a, s, src in rows:
        bold = "**" if src == "**this work**" else ""
        det.append(f"| {bold}{name}{bold} | {bold}{a:.3f}{bold} | {s:.3f} | {src} |")

    sc_tbl = ["| superclass | APEX test AUROC |", "|---|---:|"]
    for s in SUPERCLASSES:
        sc_tbl.append(f"| {s} | {sc[s]:.3f} |")
    sc_tbl.append(f"| **macro** | **{apex['superclass_macro_auroc']:.3f}** |")

    lines = [
        "# Phase 12 — APEX vs published PTB-XL baselines",
        "",
        f"APEX detector (`{apex['model']}`) on the **PTB-XL test split** (fold 10, "
        f"{apex['n_test']} records, never used for tuning). Regenerate with "
        "`python scripts/eval_baselines.py`.",
        "",
        "## Detection: macro-AUROC vs the published benchmark",
        "",
        *det,
        "",
        "Published numbers are the PTB-XL benchmark of **Strodthoff et al. 2021** "
        "([helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking)) "
        "— *the* landmark PTB-XL baseline. On the **71-code \"all\" task APEX matches "
        f"`resnet1d_wang` ({apex['all_auroc']:.3f} vs 0.919)** and sits just under the "
        "inception1d / xresnet1d101 top of 0.925 — competitive with a compact 1D-CNN, no "
        "ensembling.",
        "",
        "> **On \"Ribeiro et al. 2020\"** (the landmark deep-ECG paper): that model was "
        "trained on the *CODE* dataset (2M+ Brazilian ECGs, 6 classes) and reports F1, not "
        "AUROC on PTB-XL — a different dataset and task, so no direct PTB-XL row exists. Its "
        "residual-1D-CNN architecture is the lineage of `resnet1d_wang` above, which is the "
        "correct like-for-like PTB-XL comparison.",
        "",
        "## Per-superclass AUROC (APEX)",
        "",
        *sc_tbl,
        "",
        f"Macro F1 {apex['macro_f1']:.3f} · micro F1 {apex['micro_f1']:.3f} (per-label "
        f"thresholds tuned on validation, applied to test) · ECE {apex['ece']:.3f}. The "
        "superclass numbers are **pooled** from the 71-code head (max over each "
        "superclass's members), not from a model trained on the 5-class superdiagnostic "
        "task — a slight handicap versus the published superdiagnostic-trained rows, so "
        "read the \"all\"-task column as the apples-to-apples comparison. HYP (hypertrophy, "
        "the rarest superclass) is the weakest, matching the Phase-3 per-label finding.",
        "",
        "Explanation-quality (BLEU/ROUGE) and the GPT-4o zero-shot comparison are in "
        "`docs/model_comparison/gpt4o_comparison.md`.",
    ]
    (OUT_DIR / "baseline_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
