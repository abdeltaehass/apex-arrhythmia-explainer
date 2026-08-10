"""Phase 22 — tests for serial ECG comparison.

Mostly data-independent: intervals are measured on a synthetic ECG built to order, so the
delineator is checked against boundaries that are known rather than assumed. The handful of
tests that need PTB-XL skip cleanly when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import PTBXL_DIR
from src.longitudinal.delta import (
    DEFAULT_MDC,
    ST_LEADS_TESTED,
    Z_95,
    Z_FAMILYWISE,
    build_delta,
    compare_findings,
    compare_intervals,
    group_leads,
)
from src.longitudinal.intervals import (
    LEAD_NAMES,
    PR_PLAUSIBLE_MS,
    IntervalSet,
    measure,
    remove_baseline,
    split_half,
)
from src.longitudinal.pairs import ECGPair, build_pairs, load_longitudinal_db, same_day_null
from src.longitudinal.report import check_change_consistency, render_change_report

HAS_PTBXL = (PTBXL_DIR / "ptbxl_database.csv").exists()
HAS_WAVEFORMS = (PTBXL_DIR / "records100").exists()
needs_ptbxl = pytest.mark.skipif(not HAS_PTBXL, reason="PTB-XL metadata not downloaded")
needs_waveforms = pytest.mark.skipif(not HAS_WAVEFORMS, reason="PTB-XL waveforms not downloaded")


# --- synthetic ECG -----------------------------------------------------------
def synth_ecg(fs: int = 100, seconds: int = 10, hr: float = 60.0, pr_ms: float = 160.0,
              qrs_ms: float = 90.0, qt_ms: float = 380.0, st_mv: float = 0.0,
              with_p: bool = True, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A crude but geometrically honest 12-lead ECG with known intervals.

    Waves are Gaussian bumps placed so that the *boundaries* land where the arguments say:
    PR is measured P-onset to QRS-onset, so the P bump's centre is offset by both half the
    QRS width and the ~2-sigma lead-in of its own Gaussian. Getting that geometry wrong is
    easy and makes the generator, not the delineator, the thing under test. Good enough to
    check that boundaries are recovered and that the plausibility guards fire — not a
    physiological simulator.
    """
    p_sigma = 0.022
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    rr = 60.0 / hr
    sig = np.zeros((12, n), dtype=float)

    def bump(centre_s: float, width_s: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((t - centre_s) / width_s) ** 2)

    beat = 0
    while (r_time := 0.5 + beat * rr) < seconds - 0.6:
        wave = np.zeros(n)
        qrs_onset_s = r_time - qrs_ms / 2000.0
        if with_p:
            # QRS onset sits half a QRS width before the R peak; the P wave's onset is one
            # PR interval before that, and a Gaussian's visible onset leads its centre by
            # roughly 2 sigma.
            wave += bump(qrs_onset_s - pr_ms / 1000.0 + 2 * p_sigma, p_sigma, 0.15)
        wave += bump(r_time, qrs_ms / 1000.0 / 5.0, 1.2)             # R
        wave -= bump(r_time - qrs_ms / 1000.0 / 2.2, 0.010, 0.15)    # Q
        wave -= bump(r_time + qrs_ms / 1000.0 / 2.2, 0.012, 0.20)    # S
        # QT runs QRS-onset to T-offset, and a Gaussian's visible offset trails its centre
        # by roughly 2 sigma — so the T centre is placed accordingly rather than at some
        # fraction of QT after the R peak.
        t_sigma = 0.055
        wave += bump(qrs_onset_s + qt_ms / 1000.0 - 2 * t_sigma, t_sigma, 0.32)   # T
        if st_mv:
            j = r_time + qrs_ms / 1000.0 / 2.0
            wave += st_mv * ((t >= j) & (t <= j + 0.12))
        sig += wave[None, :]
        beat += 1

    sig *= np.linspace(0.6, 1.0, 12)[:, None]
    if noise:
        sig += rng.normal(0, noise, sig.shape)
    return sig.astype(np.float32)


# --- intervals ---------------------------------------------------------------
def test_measures_synthetic_intervals_within_tolerance():
    m = measure(synth_ecg(hr=60, pr_ms=160, qrs_ms=90), 100)
    assert m.measurable and m.p_detected
    assert m.heart_rate == pytest.approx(60, abs=3)
    assert m.pr == pytest.approx(160, abs=35)
    assert m.qrs == pytest.approx(90, abs=35)
    assert m.qt == pytest.approx(380, abs=60)


def test_no_p_wave_reports_no_pr_rather_than_zero():
    """The AF case: absent P must yield None, never a number."""
    m = measure(synth_ecg(with_p=False), 100)
    assert m.pr is None and not m.p_detected
    assert any("no P wave" in n for n in m.notes)


def test_pr_outside_physiologic_range_is_rejected():
    lo, hi = PR_PLAUSIBLE_MS
    assert lo > 0 and hi < 1000
    m = measure(synth_ecg(pr_ms=40), 100)      # impossibly short
    assert m.pr is None or lo <= m.pr <= hi


def test_implausible_qt_is_not_reported():
    m = IntervalSet(qt=120.0)
    assert m.qt == 120.0          # the dataclass holds whatever it is given...
    real = measure(synth_ecg(hr=180, qt_ms=150), 100)
    assert real.qt is None or real.qt >= 200      # ...but `measure` refuses to emit it


def test_flat_signal_is_unmeasurable_not_silently_zero():
    m = measure(np.zeros((12, 1000), dtype=np.float32), 100)
    assert not m.measurable and m.pr is None and m.qrs is None


def test_wrong_shape_is_rejected():
    assert not measure(np.zeros((3, 1000), dtype=np.float32), 100).measurable


def test_baseline_removal_kills_wander_but_keeps_the_beat():
    sig = synth_ecg()
    t = np.arange(sig.shape[1]) / 100.0
    drifted = sig + 2.0 * np.sin(2 * np.pi * 0.05 * t)[None, :]
    cleaned = remove_baseline(drifted, 100)
    assert np.abs(cleaned.mean()) < np.abs(drifted.mean())
    assert cleaned.std() < drifted.std()
    # the QRS peak survives
    assert cleaned.max() > 0.5 * sig.max()


def test_split_half_measures_the_same_recording_twice():
    a, b = split_half(synth_ecg(seconds=10, hr=75), 100)
    assert a.measurable and b.measurable
    assert abs(a.heart_rate - b.heart_rate) < 10


def test_st_levels_track_an_injected_shift():
    flat = measure(synth_ecg(st_mv=0.0), 100)
    shifted = measure(synth_ecg(st_mv=-0.3), 100)
    lead = "V6"
    assert flat.st_level[lead] > shifted.st_level[lead]


def test_lead_names_match_ptbxl_order():
    assert LEAD_NAMES[0] == "I" and LEAD_NAMES[1] == "II" and LEAD_NAMES[-1] == "V6"
    assert len(LEAD_NAMES) == 12


# --- delta -------------------------------------------------------------------
def _iv(**kw) -> IntervalSet:
    base = {"heart_rate": 70.0, "pr": 160.0, "qrs": 90.0, "qt": 380.0,
            "qtc_bazett": 400.0, "qtc_fridericia": 395.0, "p_detected": True,
            "st_level": {name: 0.0 for name in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")},
            "n_beats": 9}
    base.update(kw)
    return IntervalSet(**base)


def test_sub_threshold_change_is_not_reported():
    changes = {c.key: c for c in compare_intervals(_iv(pr=160.0), _iv(pr=160.0 + DEFAULT_MDC["pr"] - 1))}
    assert not changes["pr"].significant


def test_supra_threshold_change_is_reported_with_direction():
    changes = {c.key: c for c in compare_intervals(_iv(pr=160.0), _iv(pr=160.0 + DEFAULT_MDC["pr"] + 20))}
    assert changes["pr"].significant and changes["pr"].direction == "increased"


def test_pr_suppressed_when_no_p_wave():
    changes = {c.key: c for c in compare_intervals(_iv(), _iv(pr=None, p_detected=False))}
    assert not changes["pr"].significant
    assert "undefined" in changes["pr"].suppressed_reason


def test_pr_suppressed_in_av_dissociation_even_though_measurable():
    """Complete AV block: a PR can be measured and is still meaningless."""
    changes = {c.key: c for c in compare_intervals(
        _iv(pr=278.0), _iv(pr=180.0), rhythm_codes={"3AVB"})}
    assert not changes["pr"].significant
    assert "dissociated" in changes["pr"].suppressed_reason


def test_rate_driven_qt_change_is_suppressed():
    changes = {c.key: c for c in compare_intervals(_iv(heart_rate=70, qt=380),
                                                   _iv(heart_rate=130, qt=290))}
    assert not changes["qt"].significant and changes["qt"].suppressed_reason
    assert not changes["qtc_bazett"].significant
    # Fridericia still carries a verdict
    assert changes["qtc_fridericia"].suppressed_reason is None


def test_st_threshold_carries_the_familywise_correction():
    assert Z_FAMILYWISE > Z_95
    assert DEFAULT_MDC["st_level"] == pytest.approx(0.07, abs=1e-9)
    assert ST_LEADS_TESTED == 8


def test_new_finding_needs_the_probability_to_actually_move():
    """A label drifting 0.49 -> 0.51 is a threshold flicker, not a new onset."""
    flicker = compare_findings({"AFIB": 0.49}, {"AFIB": 0.51}, prob_mdc=0.25)
    assert not [f for f in flicker if f.status == "new"]
    real = compare_findings({"AFIB": 0.05}, {"AFIB": 0.92}, prob_mdc=0.25)
    assert [f.status for f in real] == ["new"]


def test_resolved_and_persistent_findings():
    out = {f.code: f.status for f in compare_findings(
        {"AFIB": 0.95, "SR": 0.9}, {"AFIB": 0.02, "SR": 0.93}, prob_mdc=0.25)}
    assert out["AFIB"] == "resolved" and out["SR"] == "persistent"


@pytest.mark.parametrize("leads,expected", [
    (["V4", "V5", "V6"], "V4-V6"),
    (["V2", "V3"], "V2-V3"),
    (["II"], "II"),
    (["V1", "V2", "V3", "V5"], "V1-V3 and V5"),
])
def test_group_leads(leads, expected):
    assert group_leads(leads) == expected


# --- report ------------------------------------------------------------------
def test_no_change_is_stated_explicitly():
    d = build_delta(_iv(), _iv())
    text = render_change_report(d).comparison
    assert "no significant change" in text.lower()


def test_change_report_quotes_both_values():
    d = build_delta(_iv(pr=160.0), _iv(pr=215.0))
    text = render_change_report(d).comparison
    assert "160" in text and "215" in text and "PR interval" in text


def test_rhythm_transition_reads_as_a_transition():
    d = build_delta(_iv(), _iv(),
                    prior_probs={"AFIB": 0.95, "SR": 0.01},
                    current_probs={"AFIB": 0.02, "SR": 0.97})
    text = render_change_report(d).comparison
    assert "reverted to sinus rhythm" in text.lower()


def test_uncomparable_intervals_are_declared_not_omitted():
    d = build_delta(_iv(), _iv(pr=None, p_detected=False))
    rep = render_change_report(d)
    assert "PR interval not compared" in rep.not_compared


def test_consistency_gate_passes_its_own_output():
    d = build_delta(_iv(pr=160.0), _iv(pr=215.0),
                    prior_probs={"AFIB": 0.9}, current_probs={"AFIB": 0.02})
    rep = render_change_report(d)
    assert check_change_consistency(rep.text, d).consistent


def test_consistency_gate_catches_a_fabricated_finding():
    d = build_delta(_iv(), _iv())
    fake = "New since the prior study: atrial fibrillation."
    result = check_change_consistency(fake, d)
    assert not result.consistent and "new:AFIB" in result.unsupported


def test_consistency_gate_catches_a_reversed_direction():
    """Saying the QT shortened when it lengthened is the opposite clinical message."""
    d = build_delta(_iv(qt=380.0), _iv(qt=460.0))
    lie = "The QT interval has decreased from 380 ms to 460 ms."
    result = check_change_consistency(lie, d)
    assert not result.consistent
    assert any(u.startswith("interval-direction") for u in result.unsupported)


def test_consistency_gate_tolerates_nested_vocabulary_phrases():
    """PRC(S) 'premature complexes' is a substring of PAC 'atrial premature complexes'."""
    d = build_delta(_iv(), _iv(), prior_probs={"PAC": 0.9}, current_probs={"PAC": 0.01})
    rep = render_change_report(d)
    result = check_change_consistency(rep.text, d)
    assert result.consistent, f"false positive: {result.unsupported}"


# --- cohort ------------------------------------------------------------------
def _pair(days: int = 0, hours: float = 0.0, prior=(), current=()) -> ECGPair:
    import pandas as pd

    start = pd.Timestamp("1990-01-01 08:00:00")
    return ECGPair(patient_id=1, prior_id=1, current_id=2, prior_date=start,
                   current_date=start + pd.Timedelta(days=days, hours=hours),
                   fold=1, prior_codes=frozenset(prior), current_codes=frozenset(current))


def test_same_day_uses_elapsed_hours_not_calendar_date():
    """A 21.9 h overnight repeat is a rapid repeat even though the date rolled over."""
    overnight = _pair(hours=21.9)
    assert overnight.same_day and overnight.gap_bucket() == "<24h"
    assert not _pair(hours=30).same_day


def test_gap_phrases_are_singular_where_they_should_be():
    assert _pair(days=1).describe_gap() == "1 day earlier"
    assert _pair(days=3).describe_gap() == "3 days earlier"
    assert _pair(hours=1).describe_gap() == "1 hour earlier"


def test_new_and_resolved_codes():
    p = _pair(days=5, prior=("AFIB", "SR"), current=("SR", "1AVB"))
    assert p.new_codes == frozenset({"1AVB"})
    assert p.resolved_codes == frozenset({"AFIB"})
    assert p.persistent_codes == frozenset({"SR"})
    assert not p.label_stable


@needs_ptbxl
def test_cohort_shape_and_no_fold_leakage():
    df = load_longitudinal_db()
    pairs = build_pairs(df)                    # raises if a patient spans folds
    assert len(pairs) == 2930
    assert len({p.patient_id for p in pairs}) == 2111
    assert len(build_pairs(df, folds=(10,))) == 294
    assert len(same_day_null(pairs)) == 327


@needs_ptbxl
def test_gold_pairs_recoverable():
    from src.longitudinal.pairs import gold_comparison_pairs

    gold = gold_comparison_pairs(load_longitudinal_db())
    assert len(gold) == 12
    assert all("compared" in g.statement.lower() for g in gold)
    assert all(g.pair.prior_date < g.pair.current_date for g in gold)


@needs_waveforms
def test_end_to_end_comparison_on_a_real_pair():
    """16404 -> 16408: atrial fibrillation reverting to sinus rhythm, 53 minutes apart."""
    from src.longitudinal.compare import compare_signals
    from src.longitudinal.pairs import load_signal

    prior, fs = load_signal(16404)
    current, _ = load_signal(16408)
    result = compare_signals(prior, current, fs, with_detector=False)
    assert result.prior_intervals.pr is None          # AF: no P wave
    assert result.current_intervals.pr is not None    # sinus: PR measurable
    assert result.current_intervals.pr > 200          # and prolonged, per the annotation
    assert result.consistency.consistent
