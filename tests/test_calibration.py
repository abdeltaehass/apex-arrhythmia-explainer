"""Phase 17 — tests for multi-label calibration metrics and scalers.

Data-independent: synthetic logits/labels, no torch, no dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.calibration import (
    PerLabelTemperatureScaler,
    TemperatureScaler,
    VectorScaler,
    _sigmoid,
    accuracy_style_ece,
    brier_score,
    classwise_ece,
    ece,
    load_scaler,
    mce,
    nll,
    reliability_curve,
)


def _perfectly_calibrated(n=20000, seed=0):
    """Labels drawn *from* the predicted probabilities — ECE must be ~0."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=(n, 1))
    y = (rng.uniform(0, 1, size=(n, 1)) < p).astype(float)
    return y, p


# --- metric definitions ------------------------------------------------------
def test_ece_near_zero_when_perfectly_calibrated():
    y, p = _perfectly_calibrated()
    assert ece(y, p, n_bins=15) < 0.02


def test_ece_large_when_systematically_overconfident():
    y, p = _perfectly_calibrated()
    # inflate every probability toward 1 -> over-confident, ECE must rise sharply
    assert ece(y, np.clip(p * 3, 0, 1), n_bins=15) > 0.15


def test_ece_is_symmetric_in_direction():
    """Under- and over-confidence both count as error."""
    y = np.array([[1.0]] * 50 + [[0.0]] * 50)
    over = np.full((100, 1), 0.9)
    under = np.full((100, 1), 0.1)
    assert ece(y, over) == pytest.approx(ece(y, under), abs=1e-9)


def test_ece_handles_a_constant_prediction():
    y = np.array([[1.0]] * 30 + [[0.0]] * 70)
    p = np.full((100, 1), 0.3)          # says 0.3, truth is 0.3 -> calibrated
    assert ece(y, p) < 1e-6


def test_reliability_curve_bins_and_counts():
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    p = np.array([[0.9], [0.9], [0.1], [0.1]])
    c = reliability_curve(y, p, n_bins=10)
    assert c["count"].sum() == 4
    # the 0.9 bin holds one positive of two -> observed 0.5 against confidence 0.9
    i = int(np.argmax(c["confidence"]))
    assert c["observed"][i] == pytest.approx(0.5)


def test_reliability_curve_quantile_bins_are_balanced():
    rng = np.random.default_rng(1)
    p = rng.beta(0.4, 8, size=(4000, 1))       # heavily skewed toward zero
    y = (rng.uniform(size=p.shape) < p).astype(float)
    uni = reliability_curve(y, p, 10, strategy="uniform")
    qua = reliability_curve(y, p, 10, strategy="quantile")
    # quantile binning spreads the mass far more evenly than uniform binning
    assert qua["count"].std() < uni["count"].std()


def test_reliability_curve_includes_probability_of_one():
    y = np.array([[1.0], [1.0]])
    p = np.array([[1.0], [1.0]])
    assert reliability_curve(y, p, n_bins=5)["count"].sum() == 2


def test_mce_is_at_least_ece():
    y, p = _perfectly_calibrated(n=5000, seed=3)
    p2 = np.clip(p * 1.5, 0, 1)
    assert mce(y, p2) >= ece(y, p2) - 1e-9


def test_classwise_ece_skips_labels_without_enough_positives():
    y = np.zeros((200, 3))
    y[:60, 0] = 1                       # enough positives
    y[:2, 1] = 1                        # too few -> skipped
    p = np.full((200, 3), 0.3)
    mean, per = classwise_ece(y, p, min_positives=10)
    assert 0 in per and 1 not in per and 2 not in per
    assert not np.isnan(mean)


def test_classwise_ece_nan_when_nothing_qualifies():
    y = np.zeros((50, 2))
    mean, per = classwise_ece(y, np.full((50, 2), 0.5), min_positives=10)
    assert per == {} and np.isnan(mean)


def test_accuracy_style_ece_reproduces_the_old_bug():
    """The legacy metric charges huge error to correct, well-calibrated negatives."""
    n = 5000
    y = np.zeros((n, 1))
    p = np.full((n, 1), 0.01)           # says 1%, truth is 0% -> essentially calibrated
    assert ece(y, p) < 0.02                    # correct metric: tiny
    assert accuracy_style_ece(y, p) > 0.9      # legacy metric: near-maximal


# --- proper scoring rules ----------------------------------------------------
def test_brier_and_nll_reward_the_truth():
    y = np.array([[1.0], [0.0]])
    good = np.array([[0.9], [0.1]])
    bad = np.array([[0.1], [0.9]])
    assert brier_score(y, good) < brier_score(y, bad)
    assert nll(y, good) < nll(y, bad)


def test_nll_is_finite_at_the_extremes():
    assert np.isfinite(nll(np.array([[1.0]]), np.array([[0.0]])))


# --- scalers -----------------------------------------------------------------
def _overconfident_data(n=4000, seed=7):
    """Logits whose true probability is sigmoid(z) but which are reported too sharply."""
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0, 1.5, size=(n, 2))
    y = (rng.uniform(size=z_true.shape) < _sigmoid(z_true)).astype(float)
    return z_true * 2.5, y            # inflate -> needs T ~ 2.5 to undo


def test_temperature_scaler_recovers_the_inflation():
    z, y = _overconfident_data()
    s = TemperatureScaler().fit(z, y)
    assert 1.8 < s.temperature < 3.4
    assert ece(y, s.transform(z)) < ece(y, _sigmoid(z))


def test_temperature_scaling_preserves_ranking():
    z, y = _overconfident_data()
    p = TemperatureScaler().fit(z, y).transform(z)
    # a positive monotonic map must leave the ordering of every column intact
    for j in range(z.shape[1]):
        assert np.array_equal(np.argsort(z[:, j]), np.argsort(p[:, j]))


def test_per_label_temperature_falls_back_for_rare_labels():
    z, y = _overconfident_data()
    y[:, 1] = 0.0
    y[:3, 1] = 1.0                     # only 3 positives -> below min_positives
    s = PerLabelTemperatureScaler().fit(z, y)
    assert s.n_fallback == 1
    assert s.temperatures[1] == pytest.approx(s.fallback)


def test_vector_scaling_fixes_a_bias_that_temperature_cannot():
    """The core Phase-17 finding, as a test: a shifted logit needs an intercept."""
    rng = np.random.default_rng(11)
    z_true = rng.normal(0, 1.5, size=(6000, 1))
    y = (rng.uniform(size=z_true.shape) < _sigmoid(z_true)).astype(float)
    z_bias = z_true + 2.0              # pure bias, no change in sharpness
    temp = ece(y, TemperatureScaler().fit(z_bias, y).transform(z_bias))
    vec = ece(y, VectorScaler().fit(z_bias, y).transform(z_bias))
    assert vec < temp / 2


def test_vector_scaling_preserves_ranking():
    z, y = _overconfident_data()
    p = VectorScaler().fit(z, y).transform(z)
    for j in range(z.shape[1]):
        assert np.array_equal(np.argsort(z[:, j]), np.argsort(p[:, j]))


def test_scalers_raise_before_fit():
    with pytest.raises(RuntimeError):
        PerLabelTemperatureScaler().transform(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        VectorScaler().transform(np.zeros((2, 2)))


# --- round-trip --------------------------------------------------------------
def test_load_scaler_round_trips_every_method():
    z, y = _overconfident_data(n=800)
    for scaler in (TemperatureScaler().fit(z, y),
                   PerLabelTemperatureScaler().fit(z, y),
                   VectorScaler().fit(z, y)):
        restored = load_scaler(scaler.to_dict())
        assert np.allclose(restored.transform(z), scaler.transform(z), atol=1e-6)


def test_load_scaler_rejects_unknown_method():
    with pytest.raises(ValueError):
        load_scaler({"method": "nope"})
