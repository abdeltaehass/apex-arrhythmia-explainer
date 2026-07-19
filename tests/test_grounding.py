"""Unit tests for the Phase-5 grounding layer (synthetic signals + tiny models)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip module if torch absent

import torch.nn as nn  # noqa: E402

from src.config import NUM_LABELS  # noqa: E402
from src.detection.model import ECGResNet1d  # noqa: E402
from src.grounding.regions import (  # noqa: E402
    rr_coefficient_of_variation,
    saliency_by_region,
)
from src.grounding.saliency import (  # noqa: E402
    LEAD_NAMES,
    ground,
    is_grounded,
    lead_saliency,
)
from src.grounding.sanity import expectation_for, sanity_check  # noqa: E402

FS = 100
T = 1000


class _RegionModel(nn.Module):
    """Toy detector whose target logit is exactly the sum of one lead over a window.

    Makes input-gradient saliency deterministic: d(logit)/d(x) is 1 on
    ``(lead0, t0:t1)`` and 0 everywhere else, so the trace must localize there.
    """

    def __init__(self, lead0: int, t0: int, t1: int, label: int, num_labels: int = NUM_LABELS):
        super().__init__()
        self.lead0, self.t0, self.t1, self.label = lead0, t0, t1, label
        self._p = nn.Parameter(torch.zeros(1))  # so next(model.parameters()) works

    def forward(self, x):
        out = torch.zeros(x.shape[0], NUM_LABELS, device=x.device) + self._p
        out = out.clone()
        out[:, self.label] = x[:, self.lead0, self.t0:self.t1].sum(dim=1)
        return out


# --- lead_saliency shapes / normalization -----------------------------------
def test_lead_saliency_shapes_and_ranges():
    model = ECGResNet1d(width=8, blocks=1).eval()
    sig = np.random.default_rng(0).standard_normal((12, T)).astype(np.float32)
    sal = lead_saliency(model, sig, target_label=3, fs=FS)
    assert sal.per_lead.shape == (12, T)
    assert sal.temporal.shape == (T,)
    assert sal.lead_importance.shape == (12,)
    assert 0.0 <= sal.per_lead.min() and sal.per_lead.max() <= 1.0 + 1e-6
    assert sal.lead_importance.sum() == pytest.approx(1.0, abs=1e-5)
    assert 0 <= sal.top_lead < 12


@pytest.mark.parametrize("method", ["guided_gradcam", "input_grad", "gradcam"])
def test_methods_run(method):
    model = ECGResNet1d(width=8, blocks=1).eval()
    sig = np.random.default_rng(1).standard_normal((12, T)).astype(np.float32)
    sal = lead_saliency(model, sig, target_label=0, method=method, fs=FS)
    assert np.isfinite(sal.per_lead).all()
    assert sal.per_lead.shape == (12, T)


def test_input_grad_localizes_to_the_driving_lead():
    lead0, t0, t1, label = 7, 400, 460, 12
    model = _RegionModel(lead0, t0, t1, label).eval()
    sig = np.ones((12, T), dtype=np.float32)
    sal = lead_saliency(model, sig, target_label=label, method="input_grad", smooth_s=0.0, fs=FS)
    assert sal.top_lead == lead0
    # essentially all saliency mass sits inside the driving window of that lead
    in_window = sal.per_lead[lead0, t0:t1].sum()
    assert in_window / sal.per_lead.sum() > 0.95
    assert sal.per_lead[lead0, :t0].sum() == pytest.approx(0.0, abs=1e-5)


def test_ground_accepts_codes_and_indices():
    model = ECGResNet1d(width=8, blocks=1).eval()
    sig = np.random.default_rng(2).standard_normal((12, T)).astype(np.float32)
    space = [f"C{i}" for i in range(NUM_LABELS)]
    by_code = ground(model, sig, ["C0", "C5"], label_space=space, fs=FS)
    assert set(by_code) == {"C0", "C5"}
    by_idx = ground(model, sig, 5, fs=FS)
    assert "5" in by_idx
    with pytest.raises(KeyError):
        ground(model, sig, "NOPE", label_space=space)


def test_is_grounded_accepts_1d_and_2d():
    peak = np.zeros((12, T), dtype=np.float32)
    peak[3, 100:400] = 1.0  # 30% of one lead lit up
    assert is_grounded(peak, rel_threshold=0.5, min_frac=0.02)
    assert not is_grounded(np.zeros(T), rel_threshold=0.5)


# --- regions ----------------------------------------------------------------
def _delta_saliency_at(offset_s: float, rpeaks: np.ndarray, width: int = 3) -> np.ndarray:
    s = np.zeros(T, dtype=np.float32)
    for r in rpeaks:
        c = int(r + offset_s * FS)
        if 0 <= c < T:
            s[max(0, c - width): c + width] = 1.0
    return s


def test_saliency_by_region_localizes_st_and_p():
    rpeaks = np.arange(100, T, 100)  # 1 s spacing, regular
    st = saliency_by_region(_delta_saliency_at(0.14, rpeaks), rpeaks, fs=FS)
    assert st.dominant() == "ST"
    p = saliency_by_region(_delta_saliency_at(-0.14, rpeaks), rpeaks, fs=FS)
    assert p.dominant() == "P"


def test_rr_cv_regular_vs_irregular():
    regular = np.arange(100, T, 100)
    assert rr_coefficient_of_variation(regular, FS) == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    irregular = np.cumsum(rng.integers(50, 160, size=12)) + 50
    assert rr_coefficient_of_variation(irregular, FS) > 0.15


# --- sanity checks ----------------------------------------------------------
def test_expectations():
    assert expectation_for("AFIB").kind == "rhythm_irregular"
    assert expectation_for("AMI").kind == "repolarization"
    assert expectation_for("INJAS").kind == "repolarization"
    assert expectation_for("ZZZ").kind == "unknown"


def test_sanity_repolarization_consistent_and_inconsistent():
    rpeaks = np.arange(100, T, 100)
    good = _delta_saliency_at(0.14, rpeaks)   # saliency on ST segment
    res = sanity_check(good, rpeaks, "AMI", fs=FS)
    assert res.verdict == "consistent"
    bad = _delta_saliency_at(-0.14, rpeaks)   # saliency on P wave
    res2 = sanity_check(bad, rpeaks, "AMI", fs=FS)
    assert res2.verdict == "inconsistent"


def test_sanity_afib_consistent_when_avoiding_p_wave():
    rng = np.random.default_rng(0)
    rpeaks = np.cumsum(rng.integers(50, 150, size=12)) + 50
    rpeaks = rpeaks[rpeaks < T]
    on_qrs = _delta_saliency_at(0.0, rpeaks)      # saliency on QRS, not the P window
    res = sanity_check(on_qrs, rpeaks, "AFIB", fs=FS)
    assert res.verdict == "consistent"
    on_p = _delta_saliency_at(-0.14, rpeaks)      # saliency on the (absent) P wave
    res2 = sanity_check(on_p, rpeaks, "AFIB", fs=FS)
    assert res2.verdict == "inconsistent"


def test_sanity_inconclusive_with_too_few_beats():
    rpeaks = np.array([200, 500])
    res = sanity_check(np.ones(T, dtype=np.float32), rpeaks, "AMI", fs=FS)
    assert res.verdict == "inconclusive"


def test_lead_names_cover_twelve_leads():
    assert len(LEAD_NAMES) == 12
