"""Phase-8 schema-validation + serializer test suite (no model/data needed).

Covers: the Pydantic schema itself (field constraints, JSON round-trip), input
validation (lead count / duration), and `build_report` composition (flag attribution,
review gating, consistency). A model-driven end-to-end test of `/analyze` runs only if
torch + the checkpoint are present.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.generation.prompts import target_text
from src.generation.templater import build_structured_input, render_report
from src.serving.schema import (
    MIN_LEADS,
    MIN_RELIABLE_SECONDS,
    APEXReport,
    ConsistencyOut,
    FindingOut,
    Flag,
    FlagType,
    InputValidationError,
)
from src.serving.serializer import build_report, validate_signal


def _explanation(si):
    rep = render_report(si)
    return target_text(rep["findings"], rep["impression"])


# --- schema: field constraints -----------------------------------------------
def test_finding_confidence_must_be_in_unit_interval():
    FindingOut(label="AFIB", confidence=0.5, needs_review=False)  # ok
    with pytest.raises(ValidationError):
        FindingOut(label="AFIB", confidence=1.5, needs_review=False)
    with pytest.raises(ValidationError):
        FindingOut(label="AFIB", confidence=-0.1, needs_review=False)


def test_finding_requires_label_and_needs_review():
    with pytest.raises(ValidationError):
        FindingOut(confidence=0.5, needs_review=False)  # no label
    with pytest.raises(ValidationError):
        FindingOut(label="AFIB", confidence=0.5)  # no needs_review


def test_flag_type_enum_rejects_unknown():
    Flag(type=FlagType.LOW_CONFIDENCE, message="x")  # ok
    with pytest.raises(ValidationError):
        Flag(type="not_a_flag", message="x")


def test_apex_report_json_round_trips():
    report = APEXReport(
        findings=[FindingOut(label="AFIB", description="atrial fibrillation",
                            confidence=0.9, leads=["II"], needs_review=False)],
        impression="Findings consistent with atrial fibrillation.",
        explanation="Findings:\n...\n\nImpression:\n...",
        consistency=ConsistencyOut(consistent=True, asserted=["AFIB"], surfaced=["AFIB"]),
        review_recommended=False,
    )
    restored = APEXReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.schema_version == "1.0"
    assert restored.disclaimer.startswith("Decision support only")


def test_apex_report_requires_consistency_and_review_flag():
    with pytest.raises(ValidationError):
        APEXReport()  # missing required consistency + review_recommended


# --- input validation --------------------------------------------------------
def test_validate_twelve_lead_ten_seconds_ok():
    v = validate_signal([[0.0] * 1000 for _ in range(12)], 100)
    assert v.ok and v.reliable
    assert v.errors == [] and v.warnings == []
    assert v.n_leads == 12 and v.duration_s == 10.0


def test_validate_rejects_fewer_than_twelve_leads():
    v = validate_signal([[0.0] * 1000 for _ in range(8)], 100)
    assert not v.ok
    assert any("8 leads" in e for e in v.errors)


def test_validate_rejects_more_than_twelve_leads():
    v = validate_signal([[0.0] * 1000 for _ in range(15)], 100)
    assert not v.ok
    assert any("15 leads" in e for e in v.errors)


def test_validate_flags_short_recording_as_unreliable_but_processable():
    v = validate_signal([[0.0] * 300 for _ in range(12)], 100)  # 3.0 s
    assert v.ok            # not rejected -- still processed
    assert not v.reliable  # but flagged
    assert any("too short" in w for w in v.warnings)


def test_validate_five_second_boundary_is_reliable():
    v = validate_signal([[0.0] * int(MIN_RELIABLE_SECONDS * 100) for _ in range(12)], 100)
    assert v.ok and v.reliable  # exactly 5 s is not "shorter than 5 s"


def test_validate_nonpositive_sampling_rate_errors():
    v = validate_signal([[0.0] * 1000 for _ in range(12)], 0)
    assert not v.ok
    assert any("sampling_rate" in e for e in v.errors)


def test_validate_unequal_lead_lengths_raises():
    sig = [[0.0] * 1000 for _ in range(11)] + [[0.0] * 500]
    with pytest.raises(InputValidationError):
        validate_signal(sig, 100)


def test_validate_empty_signal_raises():
    with pytest.raises(InputValidationError):
        validate_signal([], 100)


def test_validate_accepts_numpy_array():
    v = validate_signal(np.zeros((12, 1000), dtype=np.float32), 100)
    assert v.ok and v.reliable and v.n_leads == 12


def test_min_leads_constant_is_twelve():
    assert MIN_LEADS == 12


# --- build_report composition ------------------------------------------------
def test_build_report_shapes_findings_with_all_fields():
    si = build_structured_input(
        ["AFIB", "IMI"], confidences={"AFIB": 0.93, "IMI": 0.6},
        descriptions={"AFIB": "atrial fibrillation", "IMI": "inferior myocardial infarction"},
    )
    report = build_report(si, _explanation(si))
    by_label = {f.label: f for f in report.findings}
    assert by_label["AFIB"].description == "atrial fibrillation"
    assert by_label["IMI"].leads == ["II", "III", "aVF"]      # territory leads
    assert 0.0 <= by_label["AFIB"].confidence <= 1.0


def test_build_report_flags_low_confidence_finding():
    si = build_structured_input(["IMI"], confidences={"IMI": 0.6})  # 0.5 <= 0.6 < 0.7
    report = build_report(si, _explanation(si))
    imi = report.findings[0]
    assert imi.needs_review
    assert any(f.type == FlagType.LOW_CONFIDENCE for f in imi.flags)
    assert report.review_recommended


def test_build_report_clean_high_confidence_needs_no_review():
    si = build_structured_input(["NORM", "SR"], confidences={"NORM": 0.98, "SR": 0.95})
    report = build_report(si, _explanation(si))
    assert not report.review_recommended
    assert all(not f.needs_review for f in report.findings)


def test_build_report_mutual_exclusivity_flags_both_and_gates_review():
    si = build_structured_input(["NORM", "AFIB"], confidences={"NORM": 0.9, "AFIB": 0.9})
    report = build_report(si, _explanation(si))
    assert report.review_recommended
    flagged = {f.label for f in report.findings
              if any(fl.type == FlagType.MUTUAL_EXCLUSIVITY for fl in f.flags)}
    assert flagged == {"NORM", "AFIB"}


def test_build_report_detects_inconsistent_explanation():
    si = build_structured_input(["SR"], confidences={"SR": 0.9})
    # an explanation that asserts a finding the detector never surfaced
    bad = "Findings:\nIrregular rhythm.\n\nImpression:\nAtrial fibrillation."
    report = build_report(si, bad)
    assert not report.consistency.consistent
    assert "AFIB" in report.consistency.unsupported
    assert report.review_recommended


def test_build_report_unreliable_input_forces_review_and_flag():
    si = build_structured_input(["NORM", "SR"], confidences={"NORM": 0.98, "SR": 0.95})
    short = validate_signal([[0.0] * 300 for _ in range(12)], 100)  # unreliable
    report = build_report(si, _explanation(si), input_validation=short)
    assert report.review_recommended
    assert any(f.type == FlagType.UNRELIABLE_INPUT
              for finding in report.findings for f in finding.flags)


def test_build_report_impression_matches_generated_section():
    si = build_structured_input(["AFIB"], confidences={"AFIB": 0.9})
    report = build_report(si, _explanation(si))
    assert "atrial fibrillation" in report.impression.lower()
    assert report.explanation.startswith("Findings:")


# --- FastAPI endpoint (only if fastapi installed) ----------------------------
def test_endpoint_health_and_validation():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.backend.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    # < 12 leads -> hard reject via /analyze (422) and reported by /validate
    bad = {"signal": [[0.0] * 1000 for _ in range(8)], "sampling_rate": 100}
    assert client.post("/analyze", json=bad).status_code == 422
    v = client.post("/validate", json=bad).json()
    assert not v["ok"]


@pytest.mark.parametrize("with_grounding", [False])
def test_endpoint_analyze_end_to_end(with_grounding):
    pytest.importorskip("fastapi")
    torch = pytest.importorskip("torch")

    from src.config import ROOT

    if not (ROOT / "outputs" / "final_best.pt").exists():
        pytest.skip("no detector checkpoint available")
    from fastapi.testclient import TestClient

    from app.backend.main import app

    client = TestClient(app)
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((12, 1000)).astype(float).tolist()
    resp = client.post("/analyze", json={"signal": signal, "sampling_rate": 100,
                                        "backend": "template", "with_grounding": with_grounding})
    assert resp.status_code == 200
    report = APEXReport.model_validate(resp.json())  # response conforms to the schema
    assert report.input_validation.ok
    assert isinstance(report.review_recommended, bool)
    _ = torch  # silence unused
