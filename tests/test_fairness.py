"""Phases 14 & 18 — tests for the demographic subgroup analysis.

Data-independent: synthetic label/probability matrices, no torch, no dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.fairness import (
    AGE_BANDS,
    ANON_AGE,
    SEX_LABELS,
    age_band,
    benjamini_hochberg,
    bootstrap_gap_ci,
    bootstrap_label_gap,
    common_evaluable_labels,
    evaluable_labels,
    label_auroc,
    macro_auroc_on,
    max_gap,
    per_label_gaps,
    subgroup_breakdown,
)


# --- age banding -------------------------------------------------------------
def test_age_band_boundaries_are_half_open():
    assert age_band(17.9) == "<18"
    assert age_band(18) == "18-39"      # lower bound inclusive
    assert age_band(39.9) == "18-39"
    assert age_band(40) == "40-59"
    assert age_band(74.9) == "60-74"
    assert age_band(75) == "75+"
    assert age_band(89) == "75+"


def test_age_band_rejects_sentinel_and_unusable():
    # 300 is PTB-XL's ">89" anonymization sentinel — must not be bucketed as a real age
    assert age_band(ANON_AGE) is None
    assert age_band(None) is None
    assert age_band(float("nan")) is None
    assert age_band(0) is None
    assert age_band(-5) is None


def test_age_bands_cover_contiguously():
    names = [n for n, _, _ in AGE_BANDS]
    assert names == ["<18", "18-39", "40-59", "60-74", "75+"]
    # no gaps: each band's upper bound is the next band's lower bound
    for (_, _, hi), (_, lo2, _) in zip(AGE_BANDS, AGE_BANDS[1:], strict=False):
        assert hi == lo2


def test_sex_labels_match_ptbxl_encoding():
    assert SEX_LABELS == {0: "male", 1: "female"}


# --- evaluable-label logic ---------------------------------------------------
def test_evaluable_labels_needs_both_classes():
    y = np.array([[1, 0, 1],
                  [0, 0, 1],
                  [1, 0, 1]])
    # col 0 mixed -> evaluable; col 1 all-zero -> not; col 2 all-one -> not
    assert evaluable_labels(y) == {0}


def test_common_evaluable_labels_is_intersection():
    y = np.array([[1, 1],
                  [0, 1],
                  [1, 0],
                  [1, 0]])
    masks = {"a": np.array([True, True, False, False]),
             "b": np.array([False, False, True, True])}
    # group a: col0 mixed, col1 all-one -> {0}; group b: col0 all-one, col1 mixed -> {1}
    assert common_evaluable_labels(y, masks) == set()


def test_common_evaluable_labels_skips_empty_groups():
    y = np.array([[1, 0], [0, 1]])
    masks = {"a": np.array([True, True]), "empty": np.array([False, False])}
    assert common_evaluable_labels(y, masks) == {0, 1}


# --- macro AUROC on an explicit label set ------------------------------------
def test_macro_auroc_on_perfect_and_inverted():
    y = np.array([[0], [0], [1], [1]])
    perfect = np.array([[0.1], [0.2], [0.8], [0.9]])
    assert macro_auroc_on(y, perfect, {0}) == pytest.approx(1.0)
    assert macro_auroc_on(y, 1 - perfect, {0}) == pytest.approx(0.0)


def test_macro_auroc_on_empty_label_set_is_nan():
    y = np.array([[0], [1]])
    assert np.isnan(macro_auroc_on(y, np.array([[0.1], [0.9]]), set()))


def test_macro_auroc_on_ignores_single_class_columns():
    # col 1 is all-ones inside this slice: contributes nothing, doesn't crash
    y = np.array([[0, 1], [1, 1]])
    p = np.array([[0.1, 0.5], [0.9, 0.5]])
    assert macro_auroc_on(y, p, {0, 1}) == pytest.approx(1.0)


# --- subgroup breakdown ------------------------------------------------------
def _two_group_data():
    rng = np.random.default_rng(0)
    n = 120
    y = rng.integers(0, 2, size=(n, 3))
    p = rng.random((n, 3))
    masks = {"a": np.arange(n) < 60, "b": np.arange(n) >= 60}
    return y, p, masks


def test_subgroup_breakdown_shapes_and_shared_labels():
    y, p, masks = _two_group_data()
    results, common = subgroup_breakdown(y, p, masks, n_boot=20)
    assert {r.name for r in results} == {"a", "b"}
    assert all(r.n == 60 for r in results)
    # every subgroup scored on the SAME label set — that is the point of the method
    assert all(r.n_labels == len(common) for r in results)
    assert all(r.ci_low <= r.macro_auroc <= r.ci_high or np.isnan(r.ci_low) for r in results)


def test_subgroup_breakdown_accepts_explicit_label_set():
    y, p, masks = _two_group_data()
    results, common = subgroup_breakdown(y, p, masks, n_boot=10, labels={0})
    assert common == {0}
    assert all(r.n_labels == 1 for r in results)


def test_subgroup_breakdown_skips_empty_group():
    y, p, masks = _two_group_data()
    masks["empty"] = np.zeros(len(y), dtype=bool)
    results, _ = subgroup_breakdown(y, p, masks, n_boot=10)
    assert "empty" not in {r.name for r in results}


def test_reliable_flag_uses_50_record_floor():
    y, p, _ = _two_group_data()
    masks = {"small": np.arange(len(y)) < 10, "big": np.arange(len(y)) >= 10}
    results, _ = subgroup_breakdown(y, p, masks, n_boot=10)
    by = {r.name: r for r in results}
    assert not by["small"].reliable
    assert by["big"].reliable


# --- gap CI ------------------------------------------------------------------
def test_bootstrap_gap_ci_brackets_point_estimate():
    y, p, masks = _two_group_data()
    common = common_evaluable_labels(y, masks)
    gap, lo, hi = bootstrap_gap_ci(y, p, masks["a"], masks["b"], common, n_boot=50)
    assert lo <= hi
    # random data -> the gap should not be a confident non-zero difference
    assert lo <= 0 <= hi


def test_bootstrap_gap_ci_detects_a_real_gap():
    n = 200
    y = np.zeros((n, 1), dtype=int)
    y[n // 2:] = 1                                   # balanced single label
    order = np.arange(n)
    rng = np.random.default_rng(1)
    p = np.zeros((n, 1))
    # group A: probability tracks the label perfectly -> AUROC 1.0
    # group B: probability is pure noise            -> AUROC ~0.5
    mask_a = order % 2 == 0
    p[mask_a, 0] = y[mask_a, 0]
    p[~mask_a, 0] = rng.random((~mask_a).sum())
    gap, lo, hi = bootstrap_gap_ci(y, p, mask_a, ~mask_a, {0}, n_boot=100)
    assert gap > 0.2
    assert lo > 0          # CI excludes zero -> a detected difference


def test_max_gap_respects_reliable_filter():
    y, p, _ = _two_group_data()
    masks = {"tiny": np.arange(len(y)) < 5,
             "big1": (np.arange(len(y)) >= 5) & (np.arange(len(y)) < 60),
             "big2": np.arange(len(y)) >= 60}
    results, _ = subgroup_breakdown(y, p, masks, n_boot=10)
    # the tiny group can be an outlier; reliable_only must exclude it
    assert max_gap(results, reliable_only=True) <= max_gap(results, reliable_only=False) + 1e-9


def test_max_gap_nan_when_too_few_groups():
    y, p, _ = _two_group_data()
    results, _ = subgroup_breakdown(y, p, {"only": np.ones(len(y), dtype=bool)}, n_boot=5)
    assert np.isnan(max_gap(results))


# --- per-label subgroup analysis (Phase 18) ----------------------------------
LABELS_3 = ["A", "B", "C"]


def test_label_auroc_nan_when_single_class():
    assert np.isnan(label_auroc(np.zeros(10), np.random.default_rng(0).random(10)))
    assert label_auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_benjamini_hochberg_is_monotone_and_bounded():
    q = benjamini_hochberg([0.001, 0.01, 0.04, 0.5])
    assert all(0 <= v <= 1 for v in q)
    # q-values must be non-decreasing in p
    assert q == sorted(q)
    # BH always inflates (or keeps) the raw p-value
    assert all(qq >= pp - 1e-12 for qq, pp in zip(q, [0.001, 0.01, 0.04, 0.5], strict=True))


def test_benjamini_hochberg_controls_a_family_of_nulls():
    """20 uniform p-values (all null) should yield essentially no discoveries."""
    rng = np.random.default_rng(4)
    q = benjamini_hochberg(list(rng.uniform(size=20)))
    assert sum(v < 0.05 for v in q) == 0


def test_benjamini_hochberg_preserves_nans_and_handles_empty():
    q = benjamini_hochberg([0.01, float("nan"), 0.02])
    assert np.isnan(q[1]) and np.isfinite(q[0]) and np.isfinite(q[2])
    assert all(np.isnan(v) for v in benjamini_hochberg([float("nan")]))


def test_bootstrap_label_gap_detects_a_real_difference():
    n = 400
    y = np.zeros((n, 1))
    y[::2, 0] = 1
    p = np.zeros((n, 1))
    a = np.arange(n) < n // 2
    p[a, 0] = y[a, 0]                                  # group A: perfect
    p[~a, 0] = np.random.default_rng(2).random((~a).sum())   # group B: noise
    gap, lo, hi, pv = bootstrap_label_gap(y, p, a, ~a, 0, n_boot=200)
    assert gap > 0.2 and lo > 0 and pv < 0.05


def test_bootstrap_label_gap_finds_nothing_when_groups_match():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=(600, 1))
    p = rng.random((600, 1))
    a = np.arange(600) < 300
    _, lo, hi, pv = bootstrap_label_gap(y, p, a, ~a, 0, n_boot=200)
    assert lo <= 0 <= hi and pv > 0.05


def _split_data(n=300, seed=6):
    """Positives interleaved so *both* subgroups carry them (label B deliberately rare)."""
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 3))
    y[::4, 0] = 1                       # ~75 positives, spread across both halves
    y[:4, 1] = 1                        # 4 positives, both halves starved -> underpowered
    y[1::5, 2] = 1                      # ~60 positives, spread
    a = np.arange(n) < n // 2
    return y, rng.random((n, 3)), a


def test_per_label_gaps_marks_underpowered_labels():
    y, p, a = _split_data()
    gaps = {g.label: g for g in per_label_gaps(y, p, LABELS_3, a, ~a, n_boot=50)}
    assert gaps["B"].powered is False
    assert np.isnan(gaps["B"].gap) and not gaps["B"].significant
    assert gaps["A"].powered is True and np.isfinite(gaps["A"].gap)
    assert gaps["A"].n_pos_a >= 10 and gaps["A"].n_pos_b >= 10


def test_per_label_gaps_assigns_q_values_only_to_powered_labels():
    y, p, a = _split_data(seed=7)
    gaps = per_label_gaps(y, p, LABELS_3, a, ~a, n_boot=50)
    assert any(g.powered for g in gaps) and any(not g.powered for g in gaps)
    for g in gaps:
        assert np.isfinite(g.q_value) if g.powered else np.isnan(g.q_value)


def test_significant_requires_both_power_and_fdr():
    from src.eval.fairness import LabelGap

    assert not LabelGap("X", 0.9, 0.5, 3, 3, 0.4, 0.1, 0.7, 0.001,
                        q_value=0.001, powered=False).significant
    assert LabelGap("X", 0.9, 0.5, 50, 50, 0.4, 0.1, 0.7, 0.001,
                    q_value=0.01, powered=True).significant
    assert not LabelGap("X", 0.9, 0.5, 50, 50, 0.4, -0.1, 0.7, 0.4,
                        q_value=0.4, powered=True).significant
