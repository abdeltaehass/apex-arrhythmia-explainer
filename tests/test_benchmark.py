"""Phase 28 — tests for the foundation-model benchmark harness.

Data-independent: synthetic scores and stub systems. Nothing here calls an API, loads a
language model, or touches PTB-XL. The load-bearing tests are the ones about *abstention*
and *provenance* — a benchmark that scores a non-answer as a wrong answer, or that lets a
cited number sit in the same column as a measured one, produces a table that reads as
authoritative and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import (
    SUPERCLASSES,
    BenchOutput,
    System,
    available_systems,
    build_prompt,
    build_system,
    compliance,
    cost_per_1k,
    coverage,
    describe,
    latency_percentiles,
    parse_interpretation,
    parse_scores,
    per_superclass_auroc,
    self_contradiction_rate,
    summarize,
)
from src.longitudinal.intervals import IntervalSet


def intervals(**kw) -> IntervalSet:
    base = {"heart_rate": 72.0, "pr": 160.0, "qrs": 90.0, "qt": 380.0,
            "qtc_fridericia": 396.0, "p_detected": True,
            "st_level": {"I": 0.01, "V5": -0.12}}
    base.update(kw)
    return IntervalSet(**base)


class StubSystem(System):
    name = "stub"

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._i = 0

    def predict(self, signal, fs=100):
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out


# --- the feature protocol -----------------------------------------------------
def test_prompt_contains_the_measurements_and_the_answer_format():
    prompt = build_prompt(intervals())
    assert "72 bpm" in prompt and "160 ms" in prompt
    for label in SUPERCLASSES:
        assert f"{label}:" in prompt


def test_unmeasurable_values_are_stated_not_omitted():
    """A missing PR in atrial fibrillation is informative; dropping the line reads as normal."""
    text = describe(intervals(pr=None, p_detected=False))
    assert "PR interval: not measurable" in text
    assert "P waves: not detectable" in text


def test_st_levels_are_signed():
    assert "V5 -0.12" in describe(intervals())


# --- score parsing ------------------------------------------------------------
def test_parses_the_requested_format():
    text = "NORM: 7\nMI: 2\nSTTC: 5\nCD: 1\nHYP: 0\nINTERPRETATION: Sinus rhythm."
    assert parse_scores(text) == {"NORM": 0.7, "MI": 0.2, "STTC": 0.5, "CD": 0.1, "HYP": 0.0}
    assert parse_interpretation(text) == "Sinus rhythm."


@pytest.mark.parametrize("text,expected", [
    ("MI: 7/10", 0.7),
    ("**MI**: 7", 0.7),
    ("- MI — 7", 0.7),
    ("MI = 6.5", 0.65),
])
def test_tolerates_formatting_variants(text, expected):
    """Scoring a compliant answer as missing would understate the model being tested."""
    assert parse_scores(text)["MI"] == pytest.approx(expected)


def test_missing_label_is_none_not_zero():
    """A non-answer and a confident 'absent' are different claims."""
    assert parse_scores("NORM: 5")["HYP"] is None


def test_out_of_range_is_clipped_not_dropped():
    assert parse_scores("MI: 12")["MI"] == 1.0


def test_compliance_counts_supplied_scores():
    assert compliance("NORM: 5\nMI: 5") == pytest.approx(2 / 5)
    assert compliance("I cannot read this.") == 0.0


# --- metrics ------------------------------------------------------------------
def test_auroc_is_perfect_on_separable_scores():
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]] * 20)
    scores = [{s: float(row[j]) for j, s in enumerate(SUPERCLASSES)} for row in y]
    _, macro = per_superclass_auroc(y, scores)
    assert macro == pytest.approx(1.0)


def test_abstentions_are_excluded_from_auroc_not_scored_as_wrong():
    """Scoring a non-answer as 0.5 silently rewards hedging."""
    y = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]] * 20)
    good = [{s: float(row[0] if s == "NORM" else 0.0) for s in SUPERCLASSES} for row in y]
    partial = [dict(d) for d in good]
    for d in partial[:10]:
        d["NORM"] = None
    assert per_superclass_auroc(y, partial)[0]["NORM"] == pytest.approx(
        per_superclass_auroc(y, good)[0]["NORM"])


def test_coverage_measures_answered_cells():
    full = [{s: 0.5 for s in SUPERCLASSES}] * 10
    assert coverage(full) == 1.0
    holed = [dict(d) for d in full]
    holed[0]["HYP"] = None
    assert coverage(holed) == pytest.approx(1 - 1 / 50)


def test_self_contradiction_detects_text_disagreeing_with_scores():
    contradicting = BenchOutput(
        scores={"NORM": 0.9, "MI": 0.05, "STTC": 0.1, "CD": 0.1, "HYP": 0.1},
        explanation="Findings consistent with anteroseptal myocardial infarction.")
    agreeing = BenchOutput(
        scores={"NORM": 0.1, "MI": 0.95, "STTC": 0.1, "CD": 0.1, "HYP": 0.1},
        explanation="Findings consistent with anteroseptal myocardial infarction.")
    rate, n = self_contradiction_rate([contradicting])
    assert rate == 1.0 and n == 1
    rate, n = self_contradiction_rate([agreeing])
    assert rate == 0.0 and n == 1


def test_silent_text_is_skipped_not_counted_as_clean():
    silent = BenchOutput(scores={s: 0.1 for s in SUPERCLASSES}, explanation="Unremarkable.")
    rate, n = self_contradiction_rate([silent])
    assert np.isnan(rate) and n == 0


def test_self_contradiction_reports_its_denominator():
    """The rate is meaningless alone — most replies assert nothing, so n is usually small."""
    asserting = BenchOutput(
        scores={"NORM": 0.9, "MI": 0.05, "STTC": 0.1, "CD": 0.1, "HYP": 0.1},
        explanation="Findings consistent with anteroseptal myocardial infarction.")
    silent = BenchOutput(scores={s: 0.1 for s in SUPERCLASSES}, explanation="Unremarkable.")
    rate, n = self_contradiction_rate([asserting] + [silent] * 99)
    assert rate == 1.0, "rate alone reads as 100% of everything"
    assert n == 1, "which is why the denominator must travel with it"


def test_latency_percentiles_ignore_failed_calls():
    outs = [BenchOutput(latency_s=1.0), BenchOutput(latency_s=3.0),
            BenchOutput(error="boom", latency_s=99.0)]
    p50, p95 = latency_percentiles(outs)
    assert p50 == pytest.approx(2.0) and p95 < 3.1


def test_local_cost_scales_with_measured_latency():
    cheap = cost_per_1k(StubSystem([]), [BenchOutput(latency_s=1.0)], hourly_usd=3.6)
    dear = cost_per_1k(StubSystem([]), [BenchOutput(latency_s=2.0)], hourly_usd=3.6)
    assert dear == pytest.approx(2 * cheap)
    assert cheap == pytest.approx(1.0)          # 1 s x 1000 / 3600 h x $3.6


def test_api_cost_uses_measured_tokens():
    from src.benchmark.systems import OpenAISystem

    system = OpenAISystem(model="gpt-4o")
    out = BenchOutput(scores={}, latency_s=1.0, tokens_in=1000, tokens_out=100)
    # 1000 in @ $2.50/M + 100 out @ $10/M = $0.0035 per call = $3.50 per 1k
    assert cost_per_1k(system, [out]) == pytest.approx(3.5, rel=1e-6)


def test_unknown_api_model_yields_nan_rather_than_a_guess():
    from src.benchmark.systems import OpenAISystem

    system = OpenAISystem(model="gpt-9-imaginary")
    assert np.isnan(cost_per_1k(system, [BenchOutput(tokens_in=10, tokens_out=10)]))


# --- summarize ----------------------------------------------------------------
def test_summarize_counts_errors_separately():
    y = np.zeros((4, 5), dtype=int)
    y[:, 0] = [1, 0, 1, 0]
    outs = [BenchOutput(scores={s: 0.5 for s in SUPERCLASSES}, latency_s=0.1)
            for _ in range(3)] + [BenchOutput(error="timeout")]
    scores = summarize(StubSystem([]), outs, y)
    assert scores.n_records == 4 and scores.n_errors == 1


# --- registry -----------------------------------------------------------------
def test_registry_lists_both_local_and_hosted_systems():
    systems = available_systems()
    assert {"apex", "apex-student", "local-llm", "gpt-4o", "claude"} <= set(systems)


def test_build_system_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown system"):
        build_system("gpt-5-turbo-ultra")


def test_hosted_systems_are_marked_as_such():
    """Hosting is the privacy column — it must be a property of the system, not a footnote."""
    assert build_system("apex").hosting == "local"
    assert build_system("gpt-4o").hosting == "api"
    assert build_system("gpt-4o").kind == "generalist"
    assert build_system("apex").kind == "specialist"
