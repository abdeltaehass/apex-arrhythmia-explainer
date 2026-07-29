"""Phase 13 — tests for the edge-case curation / failure-analysis logic.

All data-independent: synthetic probability rows and label lists, no torch, no dataset.
"""

from __future__ import annotations

import numpy as np
from pytest import approx

from src.eval.edge_cases import (
    artifact_profile,
    cohort_metrics,
    evaluate_record,
    parse_affected,
    rarest_labels,
    select_borderline,
    select_carrying,
    select_multicondition,
    surfaced_from_probs,
)

LABELS = ["NORM", "SR", "AFIB", "STE_", "IMI", "LVH"]


def _probs(**kw) -> np.ndarray:
    row = np.zeros(len(LABELS), dtype=np.float32)
    for code, p in kw.items():
        row[LABELS.index(code)] = p
    return row


# --- noise-annotation parsing ------------------------------------------------
def test_parse_affected_whole_record():
    assert parse_affected(" , alles,  ") == ["ALL"]


def test_parse_affected_leads_and_empty():
    assert parse_affected(" , V1") == ["V1"]
    assert parse_affected("I-V1") == ["I-V1"]
    assert parse_affected("") == []
    assert parse_affected(None) == []
    assert parse_affected(float("nan")) == []  # pandas NaN cell


def test_artifact_profile_significance():
    # whole-record noise is significant on its own
    whole = artifact_profile(static_noise=" , alles,  ")
    assert whole.whole_record and whole.significant and whole.any_noise
    # a single mild baseline drift on one lead is not "significant"
    mild = artifact_profile(baseline_drift=" , V6")
    assert mild.any_noise and not mild.significant
    # two noise types at once *is* significant
    two = artifact_profile(baseline_drift=" , V6", static_noise=" , V1")
    assert two.n_types == 2 and two.significant
    # a burst / electrode problem is significant even alone
    burst = artifact_profile(burst_noise="V5")
    assert burst.burst_or_electrode and burst.significant
    # no annotations -> nothing
    clean = artifact_profile()
    assert not clean.any_noise and not clean.significant


# --- surfacing + per-record outcome ------------------------------------------
def test_surfaced_from_probs_threshold():
    surfaced, conf = surfaced_from_probs(_probs(NORM=0.9, AFIB=0.5, IMI=0.49), LABELS, 0.5)
    assert surfaced == ["NORM", "AFIB"]           # 0.49 is below threshold, excluded
    assert conf["AFIB"] == 0.5 and "IMI" not in conf


def test_evaluate_record_miss_and_overflag():
    # present AFIB + IMI; model surfaces NORM (fp) and AFIB (hit), misses IMI
    o = evaluate_record(1, ["AFIB", "IMI"], _probs(NORM=0.8, AFIB=0.9, IMI=0.2), LABELS)
    assert o.surfaced == ["NORM", "AFIB"]
    assert o.misses == ["IMI"]
    assert o.false_positives == ["NORM"]
    assert not o.correct_silent


def test_evaluate_record_dangerous_and_near_miss():
    urgent = frozenset({"STE_"})
    # STE_ present at 0.42 (near miss, and urgent) -> missed + dangerous + near
    o = evaluate_record(2, ["STE_"], _probs(STE_=0.42), LABELS, urgent=urgent,
                        threshold=0.5, near_band=0.15)
    assert o.misses == ["STE_"]
    assert o.missed_urgent == ["STE_"]
    assert o.near_miss == ["STE_"]          # 0.35 <= 0.42 < 0.5
    assert o.top_miss_prob == approx(0.42, abs=1e-5)


def test_evaluate_record_present_outside_label_space_ignored():
    o = evaluate_record(3, ["AFIB", "NOTACODE"], _probs(AFIB=0.9), LABELS)
    assert o.present == ["AFIB"] and not o.misses


def test_correct_silent_when_exact():
    o = evaluate_record(4, ["NORM", "SR"], _probs(NORM=0.9, SR=0.8), LABELS)
    assert o.correct_silent and not o.misses and not o.false_positives


# --- cohort selection --------------------------------------------------------
def test_select_multicondition():
    present = [["NORM"], ["AFIB", "IMI", "LVH", "STE_"], ["SR", "AFIB"]]
    assert select_multicondition(present, min_codes=4) == [1]


def test_select_borderline_only_present_labels_near_threshold():
    probs = np.stack([
        _probs(AFIB=0.55),   # present AFIB near 0.5 -> borderline
        _probs(AFIB=0.95),   # present AFIB far from 0.5 -> not
        _probs(IMI=0.52),    # IMI near 0.5 but NOT present here -> not borderline
    ])
    present = [["AFIB"], ["AFIB"], ["NORM"]]
    assert select_borderline(probs, present, LABELS, threshold=0.5, band=0.1) == [0]


def test_select_carrying():
    present = [["NORM"], ["STE_", "SR"], ["IMI"]]
    assert select_carrying(present, {"STE_", "IMI"}) == [1, 2]


def test_rarest_labels_orders_and_respects_min_count():
    counts = {"A": 5, "B": 1, "C": 100, "D": 0}
    # D has 0 examples -> excluded by min_count; rarest eligible are B(1), A(5)
    assert rarest_labels(counts, k=2, min_count=1) == ["B", "A"]


# --- cohort aggregation ------------------------------------------------------
def test_cohort_metrics_aggregate():
    urgent = frozenset({"STE_"})
    outcomes = [
        evaluate_record(1, ["AFIB", "IMI"], _probs(AFIB=0.9, IMI=0.2), LABELS, urgent),  # 1 miss
        evaluate_record(2, ["STE_"], _probs(STE_=0.3), LABELS, urgent),                  # dangerous miss
        evaluate_record(3, ["NORM"], _probs(NORM=0.9), LABELS, urgent),                  # clean
    ]
    m = cohort_metrics(outcomes)
    assert m["n"] == 3
    assert m["n_present_labels"] == 4 and m["n_missed_labels"] == 2
    assert m["label_recall"] == 0.5
    assert m["dangerous_miss_records"] == 1
    assert m["records_with_any_miss"] == 2
    assert 0.0 <= m["clean_silent_rate"] <= 1.0


def test_cohort_metrics_empty():
    assert cohort_metrics([]) == {"n": 0}
