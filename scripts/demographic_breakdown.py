#!/usr/bin/env python3
"""Phase 14 — demographic performance breakdown for the model card.

    python scripts/demographic_breakdown.py

Runs the detector on the PTB-XL **test split** (fold 10) and asks whether macro-AUROC
differs by patient **sex** or **age band** — the model card documents the answer either
way, including "no detectable difference".

Two methodological guards (see `src/eval/fairness.py`):

- every subgroup is scored on the **same** label set (the labels evaluable in all of
  them), so the macro numbers are comparable rather than averaging over different labels;
- each subgroup AUROC and each between-group gap carries a **bootstrap CI**, because the
  subgroups differ hugely in size — a gap whose CI straddles zero is reported as noise,
  not as a finding.

Also counts the coverage of the populations the model card declares out of scope
(pediatric records, pacemaker rhythms), so those claims cite measured numbers.

Writes docs/model_card/demographics.{md,json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import ROOT, TRAIN_FOLDS  # noqa: E402
from src.data.labels import encode, load_database, load_scp_statements  # noqa: E402
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.eval.fairness import (  # noqa: E402
    AGE_BANDS,
    ANON_AGE,
    SEX_LABELS,
    age_band,
    bootstrap_gap_ci,
    common_evaluable_labels,
    max_gap,
    subgroup_breakdown,
)
from src.eval.superclass import SUPERCLASSES, superclass_auroc  # noqa: E402
from src.grounding import load_detector  # noqa: E402

OUT_DIR = ROOT / "docs" / "model_card"
N_BOOT = 200


def _predict(model, X, device="cpu") -> np.ndarray:
    model.eval()
    out = np.empty((len(X), model.head.out_features), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 256):
            out[s:s + 256] = torch.sigmoid(model(torch.from_numpy(X[s:s + 256]).to(device))).cpu().numpy()
    return out


def _res_dict(r):
    return {"group": r.name, "n": r.n, "macro_auroc": round(r.macro_auroc, 4),
            "ci_low": round(r.ci_low, 4), "ci_high": round(r.ci_high, 4),
            "n_labels": r.n_labels, "reliable": r.reliable}


def main() -> int:
    model, label_space, args = load_detector(device="cpu")
    scp = load_scp_statements()
    df = load_database()
    df_test = df[df["strat_fold"] == 10]

    print("loading test cache + predicting...")
    X, Y = build_split_cache("test", 100)
    # alignment guard — the demographic rows must line up with the cached prediction rows
    Y_ref = np.stack([encode(c, label_space) for c in df_test["scp_codes"]])
    assert Y_ref.shape == Y.shape and np.array_equal(Y_ref, Y), \
        "test cache is not aligned with df fold-10 order"
    probs = _predict(model, X)

    ages = df_test["age"].to_numpy(dtype=float)
    sexes = df_test["sex"].to_numpy()

    # --- sex ------------------------------------------------------------------
    sex_masks = {name: (sexes == code) for code, name in SEX_LABELS.items()}
    sex_results, sex_labels = subgroup_breakdown(Y, probs, sex_masks, n_boot=N_BOOT)
    gap, glo, ghi = bootstrap_gap_ci(Y, probs, sex_masks["male"], sex_masks["female"],
                                     sex_labels, n_boot=N_BOOT)
    sex_gap = {"gap_male_minus_female": round(gap, 4),
               "ci_low": round(glo, 4), "ci_high": round(ghi, 4),
               "significant": bool(not (glo <= 0 <= ghi))}

    # per-superclass by sex — coarser (5 classes), so more stable on subgroups
    sex_super = {}
    for name, m in sex_masks.items():
        sc = superclass_auroc(Y[m], probs[m], label_space, scp)
        sex_super[name] = {s: round(sc[s], 4) for s in SUPERCLASSES}
        sex_super[name]["macro"] = round(sc["macro"], 4)

    # --- age ------------------------------------------------------------------
    bands = [np.array([age_band(a) == name for a in ages]) for name, _, _ in AGE_BANDS]
    age_masks = {name: m for (name, _, _), m in zip(AGE_BANDS, bands, strict=True)}
    # Derive the shared label set from the *adult* bands only. The <18 band holds 13
    # records, so including it in the intersection would collapse the comparison to a
    # handful of labels and make the adult bands mutually incomparable. All bands —
    # pediatric included — are then scored on that same set, so the numbers stay
    # like-for-like; the small band is flagged as unreliable rather than dropped.
    adult_masks = {k: v for k, v in age_masks.items() if k != "<18"}
    age_labels = common_evaluable_labels(Y, adult_masks)
    age_results, _ = subgroup_breakdown(Y, probs, age_masks, n_boot=N_BOOT, labels=age_labels)

    # widest reliable-vs-reliable age gap, with a CI
    reliable = [r for r in age_results if r.reliable and not np.isnan(r.macro_auroc)]
    age_gap = None
    if len(reliable) >= 2:
        best = max(reliable, key=lambda r: r.macro_auroc)
        worst = min(reliable, key=lambda r: r.macro_auroc)
        g, lo, hi = bootstrap_gap_ci(Y, probs, age_masks[best.name], age_masks[worst.name],
                                     age_labels, n_boot=N_BOOT)
        age_gap = {"best": best.name, "worst": worst.name, "gap": round(g, 4),
                   "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                   "significant": bool(not (lo <= 0 <= hi))}

    # --- out-of-scope population coverage -------------------------------------
    train = df[df["strat_fold"].isin(TRAIN_FOLDS)]
    n_all = len(df)
    ped_all = int((df["age"] < 18).sum())
    ped_train = int((train["age"] < 18).sum())
    ped_test = int((df_test["age"] < 18).sum())
    pace_all = int(df["scp_codes"].apply(lambda c: "PACE" in c).sum())
    pace_train = int(train["scp_codes"].apply(lambda c: "PACE" in c).sum())
    pace_test = int(df_test["scp_codes"].apply(lambda c: "PACE" in c).sum())
    min_age = float(df.loc[df["age"] != ANON_AGE, "age"].min())

    coverage = {
        "n_records_total": n_all,
        "pediatric_under_18": {"total": ped_all, "share": round(ped_all / n_all, 5),
                               "train": ped_train, "test": ped_test,
                               "youngest_age_in_dataset": min_age},
        "pacemaker_PACE": {"total": pace_all, "share": round(pace_all / n_all, 5),
                           "train": pace_train, "test": pace_test},
        "age_sentinel_300_over_89": int((df["age"] == ANON_AGE).sum()),
    }

    payload = {
        "model": f"APEX ({args.get('model') or 'cnn'}_{args.get('loss') or 'bce'})",
        "split": "PTB-XL test (fold 10)",
        "n_test": int(len(df_test)),
        "n_bootstrap": N_BOOT,
        "common_label_count_sex": len(sex_labels),
        "common_label_count_age": len(age_labels),
        "sex": {"subgroups": [_res_dict(r) for r in sex_results], "gap": sex_gap,
                "superclass_auroc": sex_super},
        "age": {"subgroups": [_res_dict(r) for r in age_results], "gap": age_gap,
                "max_gap_reliable": round(max_gap(age_results), 4)},
        "out_of_scope_coverage": coverage,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "demographics.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(payload)

    print("\nsex: " + " | ".join(f"{r.name} n={r.n} AUROC {r.macro_auroc:.4f}" for r in sex_results))
    print(f"  gap {sex_gap['gap_male_minus_female']:+.4f} "
          f"[{sex_gap['ci_low']:+.4f},{sex_gap['ci_high']:+.4f}] "
          f"significant={sex_gap['significant']}")
    for r in age_results:
        flag = "" if r.reliable else "  (small n — wide CI)"
        print(f"age {r.name:>6}: n={r.n:<5} AUROC {r.macro_auroc:.4f} "
              f"[{r.ci_low:.4f},{r.ci_high:.4f}]{flag}")
    print(f"-> {OUT_DIR / 'demographics.md'}")
    return 0


def _write_markdown(p: dict) -> None:
    def rows(subs):
        out = ["| group | n | macro-AUROC | 95% CI (bootstrap) | |", "|---|---:|---:|---|---|"]
        for s in subs:
            note = "" if s["reliable"] else "⚠ small n"
            ci = ("—" if np.isnan(s["ci_low"])
                  else f"{s['ci_low']:.3f} – {s['ci_high']:.3f}")
            out.append(f"| {s['group']} | {s['n']} | {s['macro_auroc']:.4f} | {ci} | {note} |")
        return out

    sex, age = p["sex"], p["age"]
    g = sex["gap"]
    cov = p["out_of_scope_coverage"]
    ped, pace = cov["pediatric_under_18"], cov["pacemaker_PACE"]

    sup = ["| superclass | male | female |", "|---|---:|---:|"]
    for s in [*SUPERCLASSES, "macro"]:
        sup.append(f"| {s} | {sex['superclass_auroc']['male'][s]:.3f} | "
                   f"{sex['superclass_auroc']['female'][s]:.3f} |")

    ag = age["gap"]
    age_verdict = (
        f"The widest gap between two adequately-sized age bands is **{ag['best']} vs "
        f"{ag['worst']}: {ag['gap']:+.4f}** (95% CI {ag['ci_low']:+.4f} – {ag['ci_high']:+.4f}), "
        + ("which **excludes zero** — a real, if modest, age effect."
           if ag["significant"] else
           "whose CI **includes zero**, so it is not distinguishable from sampling noise.")
    ) if ag else "Too few adequately-sized age bands to compare."

    lines = [
        "# Demographic performance breakdown",
        "",
        f"`{p['model']}` on the **{p['split']}** ({p['n_test']} records). Regenerate with "
        "`python scripts/demographic_breakdown.py`.",
        "",
        "**Method.** Every subgroup is scored on the *same* label set — the labels evaluable "
        "in all subgroups of a comparison "
        f"({p['common_label_count_sex']} labels for sex, {p['common_label_count_age']} for "
        "age) — because macro-AUROC silently skips labels with only one class present, and "
        "averaging over different label sets would not be a like-for-like comparison. For the "
        "age comparison that set is derived from the **adult** bands only: the 13-record "
        "`<18` band makes almost no label evaluable, so including it in the intersection "
        "would have collapsed the whole comparison to a dozen labels. Every band, pediatric "
        "included, is then scored on that same set. Each "
        f"figure carries a percentile bootstrap CI ({p['n_bootstrap']} resamples). Subgroups "
        "under 50 records are marked; their point estimates should not be quoted without the "
        "interval.",
        "",
        "## By sex",
        "",
        *rows(sex["subgroups"]),
        "",
        f"**Gap (male − female): {g['gap_male_minus_female']:+.4f}** "
        f"(95% CI {g['ci_low']:+.4f} – {g['ci_high']:+.4f}). "
        + ("The interval **excludes zero**: a real difference at this sample size."
           if g["significant"] else
           "The interval **includes zero**, so there is **no detectable difference** in "
           "macro-AUROC between male and female patients in this test set."),
        "",
        "Per-superclass, by sex:",
        "",
        *sup,
        "",
        "## By age band",
        "",
        *rows(age["subgroups"]),
        "",
        age_verdict,
        "",
        "The `<18` band is reported to *measure* pediatric coverage, not because the model is "
        "intended for it. PTB-XL's age sentinel (300 = \"older than 89\", "
        f"{cov['age_sentinel_300_over_89']} records dataset-wide) is excluded rather than "
        "bucketed into `75+`, since treating it as a real age would invent precision the "
        "dataset deliberately removed.",
        "",
        "## Coverage of the out-of-scope populations",
        "",
        "The model card declares pediatric ECGs and pacemaker rhythms out of scope. Those "
        "claims are measured, not assumed:",
        "",
        "| population | records | share of dataset | train | test |",
        "|---|---:|---:|---:|---:|",
        f"| pediatric (age < 18) | {ped['total']} | {ped['share'] * 100:.2f}% | "
        f"{ped['train']} | {ped['test']} |",
        f"| pacemaker rhythm (`PACE`) | {pace['total']} | {pace['share'] * 100:.2f}% | "
        f"{pace['train']} | {pace['test']} |",
        "",
        f"> **Correction to a common assumption:** PTB-XL is often described as adult-only. It "
        f"is not — the youngest patient in the dataset is **{ped['youngest_age_in_dataset']:.0f} "
        f"years old**, and {ped['total']} records ({ped['share'] * 100:.2f}%) are under 18. The "
        "out-of-scope conclusion is unchanged, but the *reason* is different and stronger: "
        f"pediatric ECGs are present yet vanishingly rare ({ped['train']} training records), far "
        "too few to train or validate on. Pediatric ECGs also differ physiologically from adult "
        "ones (faster rates, right-dominant axis, benign T-wave inversion patterns), so adult "
        "norms misread them.",
    ]
    (OUT_DIR / "demographics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
