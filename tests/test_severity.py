"""Tests for the Phase-11 severity banner logic (no model/data needed)."""

from src.generation.prompts import target_text
from src.generation.templater import build_structured_input, render_report
from src.serving.serializer import build_report
from src.serving.severity import URGENT_CODES, banner_meta, severity, urgent_findings


def _report(codes, confidences):
    si = build_structured_input(codes, confidences=confidences)
    rep = render_report(si)
    return build_report(si, target_text(rep["findings"], rep["impression"]))


def test_green_for_clean_normal():
    assert severity(_report(["NORM", "SR"], {"NORM": 0.98, "SR": 0.95})) == "green"


def test_yellow_when_review_recommended_but_not_urgent():
    # low-confidence finding -> review recommended, but nothing acutely urgent
    report = _report(["IMI"], {"IMI": 0.6})
    assert report.review_recommended
    assert severity(report) == "yellow"


def test_red_for_st_elevation():
    report = _report(["STE_"], {"STE_": 0.9})
    assert severity(report) == "red"
    assert "STE_" in urgent_findings(report)


def test_red_for_subendocardial_injury():
    assert severity(_report(["INJIN"], {"INJIN": 0.85})) == "red"


def test_red_takes_precedence_over_yellow():
    # an urgent code plus a low-confidence one still reads red, not yellow
    report = _report(["STE_", "NDT"], {"STE_": 0.9, "NDT": 0.6})
    assert severity(report) == "red"


def test_urgent_findings_empty_when_not_red():
    assert urgent_findings(_report(["NORM", "SR"], {"NORM": 0.9, "SR": 0.9})) == []


def test_banner_meta_has_all_levels():
    for level in ("red", "yellow", "green"):
        meta = banner_meta(level)
        assert {"label", "detail", "color", "bg"} <= set(meta)


def test_urgent_codes_are_known_scp_codes():
    from src.generation.vocab import VOCAB

    assert URGENT_CODES <= set(VOCAB)
