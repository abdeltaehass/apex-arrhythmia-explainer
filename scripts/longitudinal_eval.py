#!/usr/bin/env python3
"""Phase 22 evaluation — fit the change thresholds, then test what the module can detect.

Six blocks, run in order because each depends on the last:

1. **Cohort.** What the longitudinal slice of PTB-XL actually contains.
2. **Repeatability.** Within-record split-half: pure measurement noise, no disease change
   possible. The floor everything else is compared against.
3. **The same-day trap.** Why the obvious null cohort is not one, with the severity numbers
   that explain it. This is the block that changed the design.
4. **Threshold fit.** Robust minimum detectable change from *training-fold* label-stable
   pairs, Bonferroni-corrected for the eight-lead ST family -> outputs/.
5. **Absolute accuracy.** Do the intervals mean what they claim? Scored on the held-out
   fold against labels the delineator never sees (1AVB implies PR > 200 ms, and so on).
6. **Change detection.** The actual task, both channels, on held-out pairs — including the
   contrast against static single-ECG detection that shows how much harder change is.

Splits are respected throughout: thresholds are fitted on folds 1-8 and every number that
is quoted as a result comes from fold 10.

    python scripts/longitudinal_eval.py            # full run
    python scripts/longitudinal_eval.py --limit 60 # quick smoke
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PTBXL_DIR, ROOT  # noqa: E402
from src.longitudinal.delta import (  # noqa: E402
    DEFAULT_PROB_MDC,
    INTERVAL_META,
    ST_LEADS_TESTED,
    Z_95,
    Z_FAMILYWISE,
    compare_findings,
)
from src.longitudinal.intervals import measure, split_half  # noqa: E402
from src.longitudinal.pairs import (  # noqa: E402
    build_pairs,
    cohort_summary,
    load_longitudinal_db,
)

CACHE = ROOT / "data" / "processed" / "longitudinal"
OUT_JSON = ROOT / "docs" / "longitudinal" / "eval.json"
THRESHOLDS = ROOT / "outputs" / "longitudinal_thresholds.json"

FIELDS = list(INTERVAL_META)
TRAIN_FOLDS = tuple(range(1, 9))
TEST_FOLD = 10

# Codes whose definition pins an interval, so the label validates the measurement.
INTERVAL_LABEL_CHECKS = [
    ("pr", "1AVB", "PR -> first-degree AV block"),
    ("qrs", "CRBBB", "QRS -> complete RBBB"),
    ("qrs", "CLBBB", "QRS -> complete LBBB"),
    ("qrs", "IVCD", "QRS -> intraventricular conduction delay"),
    ("qtc_fridericia", "LNGQT", "QTcF -> long QT"),
    ("heart_rate", "STACH", "HR -> sinus tachycardia"),
    ("heart_rate", "SBRAD", "HR -> sinus bradycardia (inverted)"),
]
# Same idea for *change*: gaining one of these labels should move the measurement.
CHANGE_LABEL_CHECKS = [
    ("pr", "1AVB", "increase", "DeltaPR -> newly labelled 1AVB"),
    ("qrs", "CRBBB", "increase", "DeltaQRS -> newly labelled CRBBB"),
    ("qrs", "CLBBB", "increase", "DeltaQRS -> newly labelled CLBBB"),
    ("heart_rate", "STACH", "increase", "DeltaHR -> newly labelled sinus tachycardia"),
    ("heart_rate", "SBRAD", "decrease", "DeltaHR -> newly labelled sinus bradycardia"),
]


def robust_rc(x: np.ndarray, z: float = Z_95) -> float:
    """Repeatability coefficient from a robust spread estimate: ``z * 1.4826 * MAD``.

    Robust rather than ``z * SD`` because a few percent of records suffer gross delineation
    failure and those outliers inflate the SD by an order of magnitude (split-half QRS:
    SD 31.3 ms, robust SD 2.5 ms). An SD-based bar would be so high that only bundle branch
    block could clear it.
    """
    x = np.asarray([v for v in np.ravel(x) if np.isfinite(v)], dtype=float)
    if len(x) < 20:
        return float("nan")
    return float(z * 1.4826 * np.median(np.abs(x - np.median(x))))


def _auroc(y_true, score) -> float | None:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 20 or len(set(y_true[ok].tolist())) < 2 or y_true[ok].sum() < 8:
        return None
    return float(roc_auc_score(y_true[ok], score[ok]))


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


# --- measurement cache -------------------------------------------------------
def measure_cohort(ecg_ids, df, do_split_half: bool) -> tuple[dict, dict]:
    import wfdb

    fn = df.set_index("ecg_id")["filename_lr"].to_dict()
    iv, sh = {}, {}
    for i, e in enumerate(ecg_ids):
        signal = np.asarray(wfdb.rdsamp(str(PTBXL_DIR / fn[e]))[0], dtype=np.float32).T
        iv[e] = measure(signal, 100)
        if do_split_half:
            sh[e] = split_half(signal, 100)
        if i and i % 1000 == 0:
            print(f"    ... {i}/{len(ecg_ids)}", flush=True)
    return iv, sh


def load_or_measure(ecg_ids, df, refresh: bool) -> tuple[dict, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "intervals.pkl"
    if path.exists() and not refresh:
        iv, sh = pickle.loads(path.read_bytes())
        if set(ecg_ids) <= set(iv):
            print(f"  [cache] {len(iv)} records from {path.relative_to(ROOT)}")
            return iv, sh
    print(f"  measuring {len(ecg_ids)} records (~3 ms each)...")
    iv, sh = measure_cohort(ecg_ids, df, do_split_half=True)
    path.write_bytes(pickle.dumps((iv, sh)))
    return iv, sh


def diffs(pairs, iv, field) -> np.ndarray:
    out = []
    for p in pairs:
        a, b = iv.get(p.prior_id), iv.get(p.current_id)
        if a is None or b is None or not (a.measurable and b.measurable):
            continue
        va, vb = getattr(a, field), getattr(b, field)
        if va is not None and vb is not None:
            out.append(vb - va)
    return np.asarray(out, dtype=float)


def st_diffs(pairs, iv) -> np.ndarray:
    out = []
    for p in pairs:
        a, b = iv.get(p.prior_id), iv.get(p.current_id)
        if a is None or b is None:
            continue
        out.extend(b.st_level[k] - a.st_level[k] for k in a.st_level if k in b.st_level)
    return np.asarray(out, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="cap pairs per cohort (smoke test)")
    ap.add_argument("--refresh", action="store_true", help="recompute the interval cache")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip block 6's diagnostic channel (no torch / checkpoint needed)")
    args = ap.parse_args()

    report: dict = {}
    df = load_longitudinal_db()
    pairs = build_pairs(df)
    if args.limit:
        pairs = pairs[: args.limit * 4]

    # --- 1. cohort ----------------------------------------------------------
    print("\n[1/6] Cohort")
    summary = cohort_summary(pairs)
    report["cohort"] = summary
    print(f"  {summary['n_pairs']} pairs from {summary['n_patients']} patients; "
          f"median gap {summary['median_gap_days']:.0f} d")
    print(f"  gap buckets: {summary['by_gap_bucket']}")
    print(f"  label-set changed in {100 * summary['label_change_rate']:.1f}% of pairs")

    ids = sorted({p.prior_id for p in pairs} | {p.current_id for p in pairs})
    iv, sh = load_or_measure(ids, df, args.refresh)
    report["measurable"] = {
        "n_records": len(ids),
        "unmeasurable": sum(not iv[e].measurable for e in ids),
        "noisy": sum(iv[e].quality == "noisy" for e in ids),
        "p_detected_rate": round(float(np.mean([iv[e].p_detected for e in ids])), 4),
    }

    # --- 2. split-half repeatability ----------------------------------------
    print("\n[2/6] Within-record split-half repeatability (instrument noise floor)")
    rep = {}
    for f in FIELDS:
        d = np.asarray([getattr(b, f) - getattr(a, f)
                        for a, b in sh.values()
                        if a.measurable and b.measurable
                        and getattr(a, f) is not None and getattr(b, f) is not None])
        if len(d) < 20:
            continue
        rep[f] = {"n": len(d), "rc_robust": round(robust_rc(d), 2),
                  "rc_sd": round(float(1.96 * np.std(d, ddof=1)), 2)}
        print(f"  {f:16s} n={len(d):5d}  RC(robust)={rep[f]['rc_robust']:7.2f}  "
              f"RC(SD)={rep[f]['rc_sd']:7.2f}")
    d = np.asarray([b.st_level[k] - a.st_level[k] for a, b in sh.values()
                    if a.measurable and b.measurable
                    for k in a.st_level if k in b.st_level])
    if len(d) >= 20:
        rep["st_level"] = {"n": len(d), "rc_robust": round(robust_rc(d), 4),
                           "rc_sd": round(float(1.96 * np.std(d, ddof=1)), 4)}
        print(f"  {'st_level':16s} n={len(d):5d}  RC(robust)={rep['st_level']['rc_robust']:7.4f}  "
              f"RC(SD)={rep['st_level']['rc_sd']:7.4f}")
    report["split_half_repeatability"] = rep

    # --- 3. the same-day trap ------------------------------------------------
    print("\n[3/6] Why same-day pairs are NOT a null cohort")
    acute = {"AMI", "IMI", "ASMI", "ALMI", "ILMI", "IPMI", "IPLMI", "INJAS", "INJAL",
             "INJIN", "INJIL", "INJLA", "STE_", "ISCAL", "ISCIN", "ISCIL", "ISCAS",
             "ISCLA", "ISCAN", "AFIB", "AFLT", "SVTAC", "PSVT", "VTAC", "STACH"}
    buckets = {
        "<24h": [p for p in pairs if p.same_day],
        "1-30d": [p for p in pairs if 1 <= p.interval_days <= 30],
        "31-365d": [p for p in pairs if 31 <= p.interval_days <= 365],
        ">1y": [p for p in pairs if p.interval_days > 365],
    }
    trap = {}
    for name, ps in buckets.items():
        stable = [p for p in ps if p.label_stable]
        trap[name] = {
            "n_pairs": len(ps),
            "pct_acute": round(100 * float(np.mean([bool((p.prior_codes | p.current_codes) & acute)
                                                    for p in ps])), 1) if ps else None,
            "pct_both_normal": round(100 * float(np.mean([("NORM" in p.prior_codes) and
                                                          ("NORM" in p.current_codes)
                                                          for p in ps])), 1) if ps else None,
            "pct_label_changed": (round(100 * float(np.mean([not p.label_stable for p in ps])), 1)
                                  if ps else None),
            "rc_qrs_all": round(robust_rc(diffs(ps, iv, "qrs")), 2),
            "rc_qrs_label_stable": round(robust_rc(diffs(stable, iv, "qrs")), 2),
            "sd_qrs_all": (round(float(np.std(diffs(ps, iv, "qrs"), ddof=1)), 2)
                           if len(diffs(ps, iv, "qrs")) > 2 else None),
            "n_label_stable": len(stable),
        }
        t = trap[name]
        print(f"  {name:8s} n={t['n_pairs']:4d}  acute {t['pct_acute']:5.1f}%  "
              f"both-NORM {t['pct_both_normal']:5.1f}%  changed {t['pct_label_changed']:5.1f}%  "
              f"| QRS SD(all) {t['sd_qrs_all']}  "
              f"RC-robust(label-stable) {t['rc_qrs_label_stable']}")
    report["same_day_trap"] = trap
    print("  -> the unconditioned spread is LARGER at <24h than at >1y (noise cannot shrink")
    print("     with time); conditioned on label stability it is flat. Selection, not noise.")

    # --- 4. threshold fit (training folds only) ------------------------------
    print(f"\n[4/6] Fitting MDC on label-stable pairs, folds {TRAIN_FOLDS[0]}-{TRAIN_FOLDS[-1]}")
    fit_pairs = [p for p in pairs if p.fold in TRAIN_FOLDS and p.label_stable]
    mdc, fit_detail = {}, {}
    for f in FIELDS:
        d = diffs(fit_pairs, iv, f)
        rc = robust_rc(d)
        if not np.isfinite(rc):
            continue
        mdc[f] = round(float(np.ceil(rc / 5.0) * 5.0), 1)   # round up to a reportable 5-unit step
        fit_detail[f] = {"n": len(d), "rc_robust": round(rc, 2), "threshold": mdc[f]}
        print(f"  {f:16s} n={len(d):5d}  RC={rc:6.2f}  -> MDC {mdc[f]:6.1f}")
    d_st = st_diffs(fit_pairs, iv)
    rc_st = robust_rc(d_st, z=Z_FAMILYWISE)
    mdc["st_level"] = round(float(np.ceil(rc_st * 100) / 100), 3)
    fit_detail["st_level"] = {
        "n": len(d_st), "rc_robust_95": round(robust_rc(d_st), 4),
        "rc_robust_familywise": round(rc_st, 4), "threshold": mdc["st_level"],
        "leads_tested": ST_LEADS_TESTED, "z_95": Z_95, "z_familywise": Z_FAMILYWISE,
        "note": (f"per-lead bar widened from z={Z_95:.2f} to z={Z_FAMILYWISE:.2f} so the "
                 f"family-wise false-positive rate across {ST_LEADS_TESTED} leads stays at "
                 f"5%; uncorrected it would be {100 * (1 - 0.95 ** ST_LEADS_TESTED):.0f}%"),
    }
    rc95 = fit_detail["st_level"]["rc_robust_95"]
    print(f"  {'st_level':16s} n={len(d_st):5d}  RC95={rc95:.4f}"
          f"  RC-familywise={rc_st:.4f}  -> MDC {mdc['st_level']}")
    prob_mdc = DEFAULT_PROB_MDC
    report["mdc_fit"] = {"fitted_on": f"folds {TRAIN_FOLDS[0]}-{TRAIN_FOLDS[-1]}, label-stable pairs",
                         "n_pairs": len(fit_pairs), "mdc": mdc, "detail": fit_detail}

    THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS.write_text(json.dumps(
        {"mdc": mdc, "prob_mdc": prob_mdc,
         "fitted_on": f"PTB-XL folds {TRAIN_FOLDS[0]}-{TRAIN_FOLDS[-1]}, label-stable pairs",
         "n_pairs": len(fit_pairs), "estimator": "1.96 * 1.4826 * MAD (robust RC)",
         "st_correction": "Bonferroni across 8 leads"}, indent=2))
    print(f"  wrote {THRESHOLDS.relative_to(ROOT)}")

    # --- 5. absolute accuracy on the held-out fold ---------------------------
    print(f"\n[5/6] Absolute interval accuracy, held-out fold {TEST_FOLD} (labels never seen)")
    test_ids = sorted({p.prior_id for p in pairs if p.fold == TEST_FOLD}
                      | {p.current_id for p in pairs if p.fold == TEST_FOLD})
    codes = df.set_index("ecg_id")["code_set"].to_dict()
    acc = {}
    for field, code, label in INTERVAL_LABEL_CHECKS:
        y = np.array([code in codes[e] for e in test_ids])
        v = np.array([getattr(iv[e], field) if getattr(iv[e], field) is not None else np.nan
                      for e in test_ids])
        sign = -1.0 if code == "SBRAD" else 1.0
        a = _auroc(y, sign * v)
        if a is None:
            print(f"  {label:42s} skipped (n_pos={int(y.sum())})")
            continue
        pos = v[y & np.isfinite(v)]
        neg = v[~y & np.isfinite(v)]
        acc[f"{field}:{code}"] = {"auroc": round(a, 4), "n_pos": int(y.sum()),
                                  "median_pos": round(float(np.median(pos)), 1),
                                  "median_neg": round(float(np.median(neg)), 1)}
        print(f"  {label:42s} AUROC={a:.3f}  median {np.median(pos):6.1f} vs {np.median(neg):6.1f}"
              f"  (n_pos={int(y.sum())})")
    report["absolute_accuracy_fold10"] = acc

    # --- 6. change detection -------------------------------------------------
    print(f"\n[6/6] Change detection, held-out fold {TEST_FOLD}")
    test_pairs = [p for p in pairs if p.fold == TEST_FOLD]
    if args.limit:
        test_pairs = test_pairs[: args.limit]
    print(f"  {len(test_pairs)} held-out pairs")

    # 6a. measurement channel vs label-derived change
    chg = {}
    for field, code, direction, label in CHANGE_LABEL_CHECKS:
        y, v = [], []
        for p in test_pairs:
            a, b = iv.get(p.prior_id), iv.get(p.current_id)
            if a is None or b is None:
                continue
            va, vb = getattr(a, field), getattr(b, field)
            if va is None or vb is None:
                continue
            y.append(code in p.new_codes)
            v.append(vb - va)
        sign = 1.0 if direction == "increase" else -1.0
        a = _auroc(np.array(y), sign * np.array(v))
        n_pos = int(np.sum(y))
        if a is None:
            print(f"  {label:42s} skipped (n_pos={n_pos}, underpowered)")
            chg[f"{field}:{code}"] = {"auroc": None, "n_pos": n_pos, "underpowered": True}
            continue
        chg[f"{field}:{code}"] = {"auroc": round(a, 4), "n_pos": n_pos, "n": len(y)}
        print(f"  {label:42s} AUROC={a:.3f}  (n_pos={n_pos}/{len(y)})")
    report["change_channel_fold10"] = chg

    # 6b. diagnostic channel: new/resolved against the annotated label change
    if not args.no_detector:
        print("\n  Diagnostic channel (detector), new-onset detection vs annotated change:")
        from src.longitudinal.compare import _predict
        from src.longitudinal.pairs import load_signal

        probs: dict[int, dict[str, float]] = {}
        need = sorted({p.prior_id for p in test_pairs} | {p.current_id for p in test_pairs})
        batch, batch_ids = [], []
        for e in need:
            batch.append(load_signal(e, df)[0])
            batch_ids.append(e)
            if len(batch) == 64:
                got, _, calibrated = _predict(batch, 100)
                probs.update(dict(zip(batch_ids, got, strict=True)))
                batch, batch_ids = [], []
        if batch:
            got, _, calibrated = _predict(batch, 100)
            probs.update(dict(zip(batch_ids, got, strict=True)))
        print(f"    calibrated probabilities: {calibrated}")

        variants = {}
        for name, pmdc in (("naive (independent thresholding)", 0.0),
                           (f"delta-p gated (>{prob_mdc})", prob_mdc)):
            tp = fp = fn = 0
            rtp = rfp = rfn = 0
            for p in test_pairs:
                fc = compare_findings(probs[p.prior_id], probs[p.current_id],
                                      prob_mdc=pmdc)
                pred_new = {f.code for f in fc if f.status == "new"}
                pred_res = {f.code for f in fc if f.status == "resolved"}
                tp += len(pred_new & p.new_codes)
                fp += len(pred_new - p.new_codes)
                fn += len(p.new_codes - pred_new)
                rtp += len(pred_res & p.resolved_codes)
                rfp += len(pred_res - p.resolved_codes)
                rfn += len(p.resolved_codes - pred_res)
            variants[name] = {"new_onset": _prf(tp, fp, fn), "resolved": _prf(rtp, rfp, rfn)}
            n, r = variants[name]["new_onset"], variants[name]["resolved"]
            print(f"    {name:34s} new: P={n['precision']:.3f} R={n['recall']:.3f} F1={n['f1']:.3f}"
                  f" | resolved: F1={r['f1']:.3f}")

        # static contrast: same detector, same records, but the ordinary "is it present"
        # question rather than "did it change"
        stp = sfp = sfn = 0
        for p in test_pairs:
            for eid, truth in ((p.prior_id, p.prior_codes), (p.current_id, p.current_codes)):
                pred = {c for c, v in probs[eid].items() if v >= 0.5}
                stp += len(pred & truth)
                sfp += len(pred - truth)
                sfn += len(truth - pred)
        static = _prf(stp, sfp, sfn)
        print(f"    {'STATIC single-ECG detection':34s} P={static['precision']:.3f} "
              f"R={static['recall']:.3f} F1={static['f1']:.3f}   <- the easier task, same data")
        report["diagnostic_channel_fold10"] = {
            "calibrated": calibrated, "variants": variants, "static_contrast": static,
            "n_pairs": len(test_pairs),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
