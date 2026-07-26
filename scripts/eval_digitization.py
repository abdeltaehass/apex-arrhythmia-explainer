#!/usr/bin/env python3
"""Evaluate the Phase-10 digitizer: render real PTB-XL signals, digitize, score fidelity.

    python scripts/eval_digitization.py                 # 50 records, clean + photo levels
    python scripts/eval_digitization.py --n 200

For each sampled record: render it to a paper-ECG image (`digitization.render`), digitize
it back (`digitization.digitize`), and measure per-lead reconstruction fidelity —
Pearson correlation (resampled to a common length + best small-lag alignment, so a
global calibration offset doesn't masquerade as low fidelity) and normalized RMSE.
Repeated at several `augment.photograph` levels to show graceful degradation. Writes
docs/digitization/report.md + report.json and a couple of example images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import PTBXL_DIR, ROOT  # noqa: E402
from src.digitization import digitize_image, render_ecg  # noqa: E402
from src.digitization.augment import photograph  # noqa: E402

OUT_DIR = ROOT / "docs" / "digitization"
LEVELS = (0.0, 0.5, 1.0)


def _fidelity(orig: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    """(mean per-lead correlation, mean per-lead normalized RMSE)."""
    T = orig.shape[1]
    corrs, nrmses = [], []
    for i in range(12):
        ri = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, recon.shape[1]), recon[i])
        best_c, best_shift = -1.0, ri
        for lag in range(-4, 5):
            b = np.roll(ri, lag)
            a, bb = orig[i][6:T - 6] - orig[i][6:T - 6].mean(), b[6:T - 6] - b[6:T - 6].mean()
            c = float((a @ bb) / (np.linalg.norm(a) * np.linalg.norm(bb) + 1e-9))
            if c > best_c:
                best_c, best_shift = c, b
        corrs.append(best_c)
        denom = orig[i].std() or 1.0
        nrmses.append(float(np.sqrt(np.mean((orig[i] - best_shift) ** 2)) / denom))
    return float(np.mean(corrs)), float(np.mean(nrmses))


def _load_signals(n: int, seed: int):
    import wfdb

    from src.data.labels import load_database

    df = load_database()
    rng = np.random.default_rng(seed)
    ids = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
    for ecg_id in ids:
        row = df.loc[ecg_id]
        sig, _ = wfdb.rdsamp(str(PTBXL_DIR / row["filename_lr"]))
        yield int(ecg_id), sig.T.astype(np.float32)


def _save_examples(rng) -> dict[str, str]:
    """One clean + one photographed example image + a digitized-overlay figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    for ecg_id, orig in _load_signals(1, seed=7):
        clean = render_ecg(orig, fs=100)
        photo = photograph(clean, level=0.6, rng=rng)
        (OUT_DIR / "examples").mkdir(parents=True, exist_ok=True)
        clean.save(OUT_DIR / "examples" / f"ecg{ecg_id}_render.png")
        photo.convert("RGB").save(OUT_DIR / "examples" / f"ecg{ecg_id}_photo.jpg", quality=75)
        recon = digitize_image(photo, fs_out=100)
        T = min(orig.shape[1], recon.shape[1])
        fig, axes = plt.subplots(3, 1, figsize=(11, 6))
        for ax, i, name in zip(axes, (1, 6, 10), ("II", "V1", "V5"), strict=True):
            ax.plot(orig[i, :T], color="#333", lw=0.8, label="original")
            ax.plot(recon[i, :T], color="#c33", lw=0.8, alpha=0.8, label="digitized")
            ax.set_ylabel(name)
            ax.set_xticks([])
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"ecg_id {ecg_id}: original vs digitized (from a photographed render)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "examples" / f"ecg{ecg_id}_overlay.png", dpi=110)
        plt.close(fig)
        saved = {"render": f"examples/ecg{ecg_id}_render.png",
                 "photo": f"examples/ecg{ecg_id}_photo.jpg",
                 "overlay": f"examples/ecg{ecg_id}_overlay.png"}
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    signals = list(_load_signals(args.n, args.seed))
    print(f"evaluating digitization on {len(signals)} records at photo levels {LEVELS}")

    results = {}
    for level in LEVELS:
        corrs, nrmses = [], []
        for _, orig in signals:
            img = render_ecg(orig, fs=100)
            if level > 0:
                img = photograph(img, level=level, rng=rng)
            c, nr = _fidelity(orig, digitize_image(img, fs_out=100))
            corrs.append(c)
            nrmses.append(nr)
        results[f"level_{level}"] = {
            "photo_level": level, "n": len(signals),
            "corr_mean": round(float(np.mean(corrs)), 4), "corr_std": round(float(np.std(corrs)), 4),
            "corr_p10": round(float(np.percentile(corrs, 10)), 4),
            "nrmse_mean": round(float(np.mean(nrmses)), 4),
        }
        print(f"  level {level}: corr {results[f'level_{level}']['corr_mean']:.3f} "
              f"+- {results[f'level_{level}']['corr_std']:.3f}, "
              f"nRMSE {results[f'level_{level}']['nrmse_mean']:.3f}")

    examples = _save_examples(rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps({"results": results, "examples": examples}, indent=2))
    _write_markdown(results, examples, args.n)
    print(f"-> {OUT_DIR / 'report.md'}")
    return 0


def _write_markdown(results: dict, examples: dict[str, str], n: int) -> None:
    rows = ["| photo level | corr (mean) | corr (std) | corr p10 | nRMSE (mean) |",
            "|---|---:|---:|---:|---:|"]
    labels = {0.0: "0.0 — clean render", 0.5: "0.5 — mild phone-photo", 1.0: "1.0 — heavy phone-photo"}
    for r in results.values():
        rows.append(f"| {labels.get(r['photo_level'], r['photo_level'])} | {r['corr_mean']} | "
                    f"{r['corr_std']} | {r['corr_p10']} | {r['nrmse_mean']} |")
    lines = [
        "# Phase 10 — ECG image digitization: fidelity report",
        "",
        f"Round-trip reconstruction fidelity over **{n} random PTB-XL records**, measured by "
        "`scripts/eval_digitization.py`: each signal is rendered to a paper-ECG image, "
        "digitized back to a `(12, T)` signal, and compared per lead.",
        "",
        "**Metric**: mean per-lead Pearson correlation between the original and the "
        "digitized signal (resampled to a common length + best small-lag alignment, so a "
        "global time/gain calibration offset isn't scored as lost shape), plus normalized "
        "RMSE (RMSE / original std). `corr p10` is the 10th-percentile record — the "
        "typical worst case, not the average.",
        "",
        "## Results",
        "",
        *rows,
        "",
        "## Examples",
        "",
    ]
    if examples:
        lines += [
            f"Rendered paper ECG (the digitizer's input): ![render]({examples['render']})",
            "",
            f"Same, photographed (blur + noise + JPEG): ![photo]({examples['photo']})",
            "",
            f"Original vs digitized, three leads: ![overlay]({examples['overlay']})",
            "",
        ]
    lines += [
        "## How it works",
        "",
        "Classical computer vision (no training data — no dataset of real paper-ECG photos "
        "paired with signals exists), in `src/digitization/`:",
        "",
        "1. **grid** — the pink grid's bounding box gives the plotting rectangle; the median "
        "spacing between detected grid lines gives the px-per-mm calibration.",
        "2. **trace** — an adaptive luminance threshold isolates the dark trace ink from the "
        "lighter grid; each of the 12 stacked lead bands is read column-by-column as the "
        "darkness-weighted centroid row.",
        "3. **calibrate** — pixels -> mm via the grid pitch, mm -> mV / seconds via the "
        "standard 10 mm/mV and 25 mm/s; the per-lead baseline is removed and each lead is "
        "resampled to the target rate.",
        "",
        "## Honest limitations",
        "",
        "- **Validated on rendered images**, not real-world phone photos. The `photo` levels "
        "approximate blur/noise/JPEG but not perspective skew, folds, or shadows; a learned "
        "trace/grid segmentation model (trainable on this renderer's paired output) is the "
        "natural upgrade for those, and the reason to keep the renderer around.",
        "- **Sharp QRS peaks are the main fidelity loss** — raster digitization smooths "
        "narrow, near-vertical strokes, so clean fidelity sits around 0.9 rather than ~1.0 "
        "(a clean sine round-trips at 0.998).",
        "- The **standard clinical 3x4 mosaic carries only 2.5 s per lead**, so full 12-lead "
        "x 10 s reconstruction uses the stacked full-width layout; the mosaic is rendered "
        "for display only.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
