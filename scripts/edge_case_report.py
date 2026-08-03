#!/usr/bin/env python3
"""Phase 13 — adversarial & edge-case testing.

    python scripts/edge_case_report.py

Curates five hard-case cohorts from the PTB-XL **test split** (fold 10) and measures how
the deployed APEX pipeline behaves on each, at the shipped surfacing rule (probability ≥
``review_threshold`` = 0.5 — what the system actually says, not a post-hoc tuned
threshold):

  1. noisy recordings with significant artifact (PTB-XL's own signal-quality annotations)
  2. borderline cases where a present label sits within 0.1 of the threshold
  3. rare labels with few training examples
  4. recordings carrying several conditions at once
  5. completely normal ECGs — does the system correctly stay quiet?

For each cohort it reports the two failure axes — **misses** (silent false negatives, with
dangerous *urgent* misses called out) and **over-flags** (false positives / alarm
fatigue) — then runs the *full* report pipeline (`analyze_signal`, with grounding) on a
handful of concrete example records so the write-up cites real reports, not just rates.

Writes docs/edge_cases/report.{md,json}: cohort table, failure-mode taxonomy with
specific example records, and recommended deployment guardrails.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import ROOT, TRAIN_FOLDS  # noqa: E402
from src.data.labels import (  # noqa: E402
    aggregate_superclasses,
    diagnostic_superclass_map,
    load_database,
    load_scp_statements,
    present_codes,
)
from src.detection.data_cache import build_split_cache  # noqa: E402
from src.eval import edge_cases as ec  # noqa: E402
from src.eval.reliability import check_low_confidence, check_mutual_exclusivity  # noqa: E402
from src.grounding import load_detector  # noqa: E402
from src.serving.severity import URGENT_CODES, severity, urgent_findings  # noqa: E402

OUT_DIR = ROOT / "docs" / "edge_cases"
THRESHOLD = 0.5
LOW_CONF = 0.7


def _predict(model, X, device="cpu") -> np.ndarray:
    model.eval()
    out = np.empty((len(X), model.head.out_features), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 256):
            out[s:s + 256] = torch.sigmoid(model(torch.from_numpy(X[s:s + 256]).to(device))).cpu().numpy()
    return out


def build_cohort(rows, ecg_ids, present_lists, probs, label_space):
    """Score a set of test-row indices; return (outcomes, review_proxy_rate)."""
    outcomes, reviewed = [], 0
    for r in rows:
        o = ec.evaluate_record(ecg_ids[r], present_lists[r], probs[r], label_space,
                               urgent=URGENT_CODES, threshold=THRESHOLD)
        outcomes.append(o)
        _, conf = ec.surfaced_from_probs(probs[r], label_space, THRESHOLD)
        low = check_low_confidence(conf, LOW_CONF)
        mutex = check_mutual_exclusivity(set(o.surfaced))
        reviewed += bool(low or mutex)
    rate = round(reviewed / len(rows), 4) if rows else None
    return outcomes, rate


def run_example(ecg_id, df, present, supmap):
    """Run the full deployed pipeline on one record and summarise what it said."""
    import wfdb

    from src.config import PTBXL_DIR
    from src.serving.serializer import analyze_signal

    raw = wfdb.rdsamp(str(PTBXL_DIR / df.loc[ecg_id, "filename_lr"]))[0].T.astype(np.float32)
    report = analyze_signal(raw, sampling_rate=100, with_grounding=True)
    surfaced = {f.label for f in report.findings}
    present_set = set(present)
    present_supers = aggregate_superclasses(df.loc[ecg_id, "scp_codes"], supmap)
    return {
        "ecg_id": int(ecg_id),
        "present": sorted(present_set),
        "present_superclasses": present_supers,
        "surfaced": [
            {"label": f.label, "confidence": f.confidence, "needs_review": f.needs_review,
             "flags": [fl.type.value for fl in f.flags]}
            for f in sorted(report.findings, key=lambda f: -f.confidence)
        ],
        "misses": sorted(present_set - surfaced),
        "missed_urgent": sorted((present_set - surfaced) & URGENT_CODES),
        "false_positives": sorted(surfaced - present_set),
        "review_recommended": report.review_recommended,
        "severity": severity(report),
        "urgent_findings": urgent_findings(report),
        "impression": report.impression,
    }


def main() -> int:
    model, label_space, args = load_detector(device="cpu")
    label_set = set(label_space)
    scp = load_scp_statements()
    supmap = diagnostic_superclass_map(scp)
    df = load_database()
    df_test = df[df["strat_fold"] == 10]
    ecg_ids = df_test.index.to_numpy()

    print("loading test cache + predicting...")
    Xte, Yte = build_split_cache("test", 100)
    # alignment guard: cache rows must line up with df_test order, or every ecg_id below
    # would be mislabelled. Recompute the label matrix from metadata and require a match.
    from src.data.labels import encode
    Y_ref = np.stack([encode(c, label_space) for c in df_test["scp_codes"]])
    assert Y_ref.shape == Yte.shape and np.array_equal(Y_ref, Yte), \
        "test cache is not aligned with df fold-10 order"
    probs = _predict(model, Xte)

    present_lists = [[c for c in present_codes(codes) if c in label_set]
                     for codes in df_test["scp_codes"]]

    # --- training support per label (for the rare-label cohort) ---------------
    train = df[df["strat_fold"].isin(TRAIN_FOLDS)]
    counts = Counter()
    for codes in train["scp_codes"]:
        for c in codes:
            if c in label_set:
                counts[c] += 1
    train_counts = {c: counts.get(c, 0) for c in label_space}

    # --- cohorts --------------------------------------------------------------
    profiles = [ec.artifact_profile(df_test.iloc[r]["baseline_drift"],
                                    df_test.iloc[r]["static_noise"],
                                    df_test.iloc[r]["burst_noise"],
                                    df_test.iloc[r]["electrodes_problems"])
                for r in range(len(df_test))]
    noisy_rows = [r for r, p in enumerate(profiles) if p.significant]
    whole_rows = [r for r, p in enumerate(profiles) if p.whole_record]
    clean_rows = [r for r, p in enumerate(profiles) if not p.any_noise]

    borderline_rows = ec.select_borderline(probs, present_lists, label_space, THRESHOLD, band=0.1)
    multi_rows = ec.select_multicondition(present_lists, min_codes=5)
    rare_labels = ec.rarest_labels(train_counts, k=12, min_count=1)
    rare_rows = ec.select_carrying(present_lists, set(rare_labels))
    normal_rows = [r for r, codes in enumerate(df_test["scp_codes"])
                   if aggregate_superclasses(codes, supmap) == ["NORM"]]

    all_rows = list(range(len(df_test)))
    cohort_rows = {
        "overall (all test)": all_rows,
        "noisy — significant artifact": noisy_rows,
        "noisy — whole-record": whole_rows,
        "borderline (present @ ±0.1)": borderline_rows,
        "rare labels (12 rarest)": rare_rows,
        "multi-condition (≥5 codes)": multi_rows,
        "normal (NORM-only)": normal_rows,
        "clean (no annotated noise)": clean_rows,
    }

    cohorts = {}
    outcomes_by_cohort = {}
    for name, rows in cohort_rows.items():
        outcomes, review_rate = build_cohort(rows, ecg_ids, present_lists, probs, label_space)
        outcomes_by_cohort[name] = outcomes
        m = ec.cohort_metrics(outcomes)
        m["review_proxy_rate"] = review_rate
        cohorts[name] = m

    # normal-ECG specific behaviour: does it stay quiet?
    normal_out = outcomes_by_cohort["normal (NORM-only)"]
    normal_surfaces_norm = sum("NORM" in o.surfaced for o in normal_out)
    normal_overflags_path = sum(
        bool(set(o.false_positives) & (set(label_space) & set(supmap))) for o in normal_out
    )
    normal_summary = {
        "n": len(normal_out),
        "surfaces_norm": normal_surfaces_norm,
        "surfaces_norm_rate": round(normal_surfaces_norm / len(normal_out), 4) if normal_out else None,
        "overflags_pathology": normal_overflags_path,
        "overflag_rate": round(normal_overflags_path / len(normal_out), 4) if normal_out else None,
    }

    # near-miss prevalence across the whole test set (silent sub-threshold findings)
    all_out = outcomes_by_cohort["overall (all test)"]
    total_misses = sum(len(o.misses) for o in all_out)
    total_near = sum(len(o.near_miss) for o in all_out)
    near_miss_share = round(total_near / total_misses, 4) if total_misses else None

    # --- concrete examples: run the real pipeline -----------------------------
    # per-row outcome for every candidate row, so example selection can be by failure mode
    cand_rows = sorted(set(sum(cohort_rows.values(), [])))
    outcome_of = {r: ec.evaluate_record(ecg_ids[r], present_lists[r], probs[r], label_space,
                                        urgent=URGENT_CODES, threshold=THRESHOLD)
                  for r in cand_rows}

    def first_match(rows, pred):
        """First ecg_id in ``rows`` whose outcome satisfies ``pred``; else the first row."""
        for r in rows:
            if pred(outcome_of[r]):
                return int(ecg_ids[r])
        return int(ecg_ids[rows[0]]) if rows else None

    picks = {
        # a whole-record-noise record that suffered a miss (artifact -> miss)
        "noisy": first_match(whole_rows, lambda o: bool(o.misses)),
        # a record with a genuine near-miss (present code dropped just below 0.5)
        "borderline": first_match(borderline_rows, lambda o: bool(o.near_miss)),
        # a rare-label record where the rare label itself was missed
        "rare": first_match(rare_rows, lambda o: bool(set(o.misses) & set(rare_labels))),
        # the record carrying the most present codes at once
        "multi_condition": int(ecg_ids[max(multi_rows, key=lambda r: len(present_lists[r]))])
        if multi_rows else None,
        # a NORM-only record the system handled cleanly (success case)
        "normal_clean": first_match(normal_rows, lambda o: o.correct_silent),
    }
    # a NORM-only record the system over-flagged (failure case), only if one exists
    over_rows = [r for r in normal_rows if outcome_of[r].false_positives]
    picks["normal_overflag"] = int(ecg_ids[over_rows[0]]) if over_rows else None

    print("running full pipeline on concrete examples...")
    examples = {}
    seen = set()
    for key, ecg_id in picks.items():
        if ecg_id is None or ecg_id in seen:
            continue
        seen.add(ecg_id)
        row = int(np.where(ecg_ids == ecg_id)[0][0])
        examples[key] = run_example(ecg_id, df, present_lists[row], supmap)
        print(f"  {key}: ecg {ecg_id} -> severity {examples[key]['severity']}, "
              f"{len(examples[key]['misses'])} miss / {len(examples[key]['false_positives'])} over-flag")

    payload = {
        "model": f"APEX ({args.get('model') or 'cnn'}_{args.get('loss') or 'bce'})",
        "threshold": THRESHOLD,
        "low_confidence_threshold": LOW_CONF,
        "n_test": int(len(df_test)),
        "urgent_codes": sorted(URGENT_CODES),
        "rare_labels": rare_labels,
        "rare_label_train_counts": {c: train_counts[c] for c in rare_labels},
        "cohorts": cohorts,
        "normal_behaviour": normal_summary,
        "near_miss_share_of_all_misses": near_miss_share,
        "examples": examples,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(payload)
    print(f"\n-> {OUT_DIR / 'report.md'}")
    return 0


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _write_markdown(p: dict) -> None:
    c = p["cohorts"]
    order = ["overall (all test)", "normal (NORM-only)", "clean (no annotated noise)",
             "noisy — significant artifact", "noisy — whole-record",
             "borderline (present @ ±0.1)", "rare labels (12 rarest)",
             "multi-condition (≥5 codes)"]

    tbl = ["| cohort | n | label recall | over-flag / rec | dangerous misses | "
           "routed to review | clean & silent |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name in order:
        m = c[name]
        tbl.append(
            f"| {name} | {m['n']} | {_pct(m.get('label_recall'))} | "
            f"{m.get('overflag_per_record', 0):.2f} | "
            f"{m.get('dangerous_miss_records', 0)} ({_pct(m.get('dangerous_miss_rate'))}) | "
            f"{_pct(m.get('review_proxy_rate'))} | {_pct(m.get('clean_silent_rate'))} |"
        )

    nb = p["normal_behaviour"]
    sig = c["noisy — significant artifact"]
    clean = c["clean (no annotated noise)"]
    multi = c["multi-condition (≥5 codes)"]
    rare = c["rare labels (12 rarest)"]

    ex = p["examples"]

    def ex_block(key, title):
        e = ex.get(key)
        if not e:
            return []
        rows = [f"| {s['label']} | {s['confidence']:.2f} | "
                f"{'review' if s['needs_review'] else 'ok'} | {', '.join(s['flags']) or '—'} |"
                for s in e["surfaced"]]
        out = [
            f"**{title} — ECG {e['ecg_id']}.**",
            "",
            f"Present (ground truth): `{', '.join(e['present']) or 'none'}` "
            f"· superclasses: {', '.join(e['present_superclasses']) or 'none'}.",
            "",
            "| surfaced | conf | gate | flags |",
            "|---|---:|---|---|",
            *(rows or ["| _nothing surfaced_ | — | — | — |"]),
            "",
            f"Severity banner: **{e['severity']}**"
            + (f" (urgent: {', '.join(e['urgent_findings'])})" if e["urgent_findings"] else "")
            + f" · review recommended: **{e['review_recommended']}**.",
        ]
        if e["misses"]:
            danger = (" ⚠ **dangerous (urgent):** " + ", ".join(e["missed_urgent"])
                      if e["missed_urgent"] else "")
            out.append("")
            out.append(f"Missed: `{', '.join(e['misses'])}`.{danger}")
        if e["false_positives"]:
            out.append("")
            out.append(f"Over-flagged (not in ground truth): `{', '.join(e['false_positives'])}`.")
        out.append("")
        return out

    lines = [
        "# Phase 13 — Adversarial & Edge-Case Report",
        "",
        f"Deployed pipeline `{p['model']}` on the PTB-XL **test split** ({p['n_test']} records, "
        f"fold 10). Everything is measured at the **shipped surfacing rule** — a finding is "
        f"surfaced when its probability ≥ {p['threshold']} — so these are the system's real "
        "outputs, not a post-hoc tuned optimum. Regenerate with "
        "`python scripts/edge_case_report.py`.",
        "",
        "Two failure axes, because they cost differently:",
        "",
        "- a **miss** is a present label the system did *not* surface — a silent false "
        "negative. A missed **urgent** code (ST-elevation / injury) is a *dangerous miss*.",
        "- an **over-flag** is a surfaced label that is not present — a false positive, and "
        "the driver of alarm fatigue.",
        "",
        "## Cohorts",
        "",
        *tbl,
        "",
        "\"Routed to review\" is a proxy: a record trips review if any surfaced finding is "
        f"below the low-confidence bar ({p['low_confidence_threshold']}) or two surfaced "
        "labels are mutually exclusive (the text-dependent consistency and grounding checks "
        "aren't included here). \"Label recall\" is per-label: of all present labels in the "
        "cohort, the share the system surfaced.",
        "",
        "## What the numbers say",
        "",
        f"**The normal ECG — does it stay quiet?** Of {nb['n']} diagnostically-normal "
        f"(NORM-only) records, the system surfaces NORM on **{_pct(nb['surfaces_norm_rate'])}** "
        f"and over-flags a diagnostic pathology on **{_pct(nb['overflag_rate'])}**. So it *mostly* "
        "says the right thing on a clean normal — but a non-trivial minority pick up a "
        "spurious diagnostic label, each of which would (correctly, given the flag) route an "
        "otherwise-normal patient to review. That is the alarm-fatigue tax.",
        "",
        f"**Why so many over-flags? The operating point, not the ranking.** The deployed rule "
        f"surfaces every label at probability ≥ {p['threshold']}, but the detector was trained "
        "with heavy class weighting (Phase 3) that deliberately inflates probabilities — mean "
        "predicted probability is 0.118 against a true base rate of 0.039, about 3x too high "
        "(pooled ECE 0.079), so 0.5 is far too low a bar. That is "
        f"why the system averages **{c['overall (all test)']['overflag_per_record']:.1f} "
        "surfaced-but-absent labels per record** and tags a spurious diagnostic code on nearly "
        "half of normal ECGs. The Phase-12 per-label F1-tuned thresholds already cut this "
        "sharply (micro-F1 0.60 at tuned thresholds vs the flood at 0.5) — and the *same* model "
        "still scores 0.92 AUROC, which is threshold-free. So most of the over-flagging here is "
        "a **calibration** problem, not a discrimination one. **Phase 17 confirmed this and "
        "fixed it**: per-label vector scaling cut ECE 0.079 -> 0.002 and spurious surfaced "
        "labels 5.09 -> 0.35 per record at this same threshold, with AUROC unchanged "
        "(`docs/calibration/report.md`). Numbers on this page are measured pre-calibration.",
        "",
        f"**Noise degrades it, as expected.** Records with significant artifact drop to "
        f"**{_pct(sig.get('label_recall'))}** label recall against **{_pct(clean.get('label_recall'))}** "
        f"on clean records, and over-flag more per record ({sig.get('overflag_per_record'):.2f} vs "
        f"{clean.get('overflag_per_record'):.2f}). Whole-record (\"alles\") noise is the worst "
        f"bucket at **{_pct(c['noisy — whole-record'].get('label_recall'))}** recall.",
        "",
        f"**Rare labels are the biggest blind spot.** Across the 12 rarest labels "
        f"(training support {min(p['rare_label_train_counts'].values())}–"
        f"{max(p['rare_label_train_counts'].values())} examples), cohort label recall is "
        f"**{_pct(rare.get('label_recall'))}** — the system most often says *nothing* rather "
        "than flag an uncertain rare finding.",
        "",
        f"**Multi-condition records lose the secondary findings.** On records carrying ≥5 codes "
        f"at once, label recall is **{_pct(multi.get('label_recall'))}**: the dominant "
        "abnormality surfaces, the co-morbid ones often don't.",
        "",
        f"**Silent near-misses are common.** **{_pct(p['near_miss_share_of_all_misses'])}** of all "
        "missed labels across the test set had a probability in 0.35–0.5 — findings the model "
        "nearly surfaced and then dropped with no trace. To a user, a 0.49 miss is "
        "indistinguishable from a confident negative.",
        "",
        "## Failure-mode taxonomy",
        "",
        _tax(ex, "F1", "Co-morbid under-call",
             "On multi-condition records the detector surfaces the dominant abnormality and "
             f"misses secondary findings (cohort recall {_pct(multi.get('label_recall'))}). "
             "Subtle STTC/HYP alongside a loud rhythm or infarct are the usual casualties.",
             "multi_condition"),
        _tax(ex, "F2", "Silent miss on rare labels",
             f"Labels with little training support have low recall ({_pct(rare.get('label_recall'))} "
             "on the 12 rarest); the system emits a confident-looking negative instead of "
             "abstaining.", "rare"),
        _tax(ex, "F3", "Borderline suppression / near-miss",
             f"{_pct(p['near_miss_share_of_all_misses'])} of misses sat in 0.35–0.5. A finding "
             "just under threshold is dropped silently — no surfacing and no flag — so a "
             "near-miss is invisible to the reader.", "borderline"),
        _tax(ex, "F4", "Artifact-driven error",
             f"Significant artifact cuts recall to {_pct(sig.get('label_recall'))} (whole-record "
             f"{_pct(c['noisy — whole-record'].get('label_recall'))}) and raises over-flagging. "
             "The system does not itself refuse a corrupted trace.", "noisy"),
        _tax(ex, "F5", "Over-flagging / alarm fatigue",
             f"Even on clean normals, {_pct(nb['overflag_rate'])} pick up a spurious diagnostic "
             "label, and the low-confidence gate routes a large share of all records to review "
             "(Phase 7 measured 75.9% on validation). High sensitivity is bought with reviewer "
             "load.", "normal_overflag"),
        "",
        "**F6 — Dangerous miss (missed urgent).** The highest-severity mode: an urgent "
        f"ST-elevation / injury code ({', '.join(p['urgent_codes'])}) present but not surfaced, "
        "so no red banner fires. Per-cohort counts are in the \"dangerous misses\" column above; "
        "this is the number to watch in production.",
        "",
        "## Concrete examples",
        "",
        "Each was run through the *full* deployed pipeline (`analyze_signal`, grounding on).",
        "",
        *ex_block("normal_clean", "Normal, handled correctly"),
        *ex_block("multi_condition", "Multi-condition record"),
        *ex_block("noisy", "Whole-record noise"),
        *ex_block("borderline", "Borderline / near-miss"),
        *ex_block("rare", "Rare label"),
        *ex_block("normal_overflag", "Normal, over-flagged"),
        "## Recommended deployment guardrails",
        "",
        "1. **Human over-read stays mandatory.** APEX is decision *support*; a green banner is "
        "never a clearance. The disclaimer and review gate apply at every severity level.",
        "2. **Surface a sub-threshold tier for high-consequence codes.** For MI / ST-change / "
        "injury labels, show a \"possible — below confidence\" note when the probability lands "
        "in ~0.35–0.5 instead of dropping it silently (mitigates F3, F6).",
        "3. **Refuse or hard-flag corrupted input.** Route records with whole-record or "
        "multi-type artifact (auto-detected SNR, or annotations where available) to mandatory "
        "review; never emit a green banner on a trace the system can't trust (F4).",
        "4. **Abstain, don't assert, on rare labels.** Below a training-support floor, express "
        "low confidence / defer rather than implying a confident negative (F2).",
        "5. **Report the full surfaced set, not just the top finding, and state that a missing "
        "secondary finding is not its exclusion** (F1).",
        "6. **Tune the confidence gate to the setting.** A screening deployment should bias "
        "toward sensitivity (accept more review load); a confirmatory one toward precision. "
        "This is the calibration lever (F5) — see the calibration follow-up.",
        "7. **Monitor red-banner (urgent) recall as the primary safety metric in production**, "
        "and audit every dangerous miss (F6).",
        "",
        "## Honest limitations",
        "",
        "- Ground truth is PTB-XL's human cardiologist labels, which themselves carry "
        "annotation noise; a \"miss\" or \"over-flag\" against them is not always a true error.",
        "- The artifact cohort uses PTB-XL's own signal-quality annotations, which are "
        "incomplete — some noisy records are unlabelled, so the clean cohort is a mild "
        "over-estimate of clean performance.",
        "- Text-dependent checks (consistency, grounding-conflict) are exercised only on the "
        "concrete examples, not across whole cohorts, because they require running generation "
        "on every record. The cohort \"routed to review\" proxy therefore under-counts.",
        "- All numbers are for the single shipped checkpoint at threshold "
        f"{p['threshold']}; a different operating point trades misses against over-flags.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def _tax(examples, tag, title, body, example_key):
    e = examples.get(example_key)
    ref = f" _Example: ECG {e['ecg_id']}._" if e else ""
    return f"**{tag} — {title}.** {body}{ref}\n"


if __name__ == "__main__":
    raise SystemExit(main())
