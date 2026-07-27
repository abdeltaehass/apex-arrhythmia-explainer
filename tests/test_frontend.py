"""Tests for the Phase-11 dashboard views (matplotlib figure + HTML; no gradio, no model)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from app.frontend.views import (  # noqa: E402
    disclaimer_html,
    ecg_figure,
    finding_colors,
    report_html,
    severity_banner_html,
)
from src.generation.prompts import target_text  # noqa: E402
from src.generation.templater import build_structured_input, render_report  # noqa: E402
from src.grounding.saliency import LeadSaliency  # noqa: E402
from src.serving.serializer import build_report  # noqa: E402


def _report(codes, confidences):
    si = build_structured_input(codes, confidences=confidences,
                                descriptions={c: c.lower() for c in codes})
    rep = render_report(si)
    return build_report(si, target_text(rep["findings"], rep["impression"]))


def _saliency(code, top_lead=1, T=1000):
    per_lead = np.zeros((12, T), dtype=np.float32)
    per_lead[top_lead, 300:360] = 1.0
    mass = per_lead.sum(axis=1)
    return LeadSaliency(label_index=0, label=code, method="guided_gradcam", logit=1.0, prob=0.9,
                        per_lead=per_lead, temporal=per_lead.max(axis=0),
                        lead_importance=mass / mass.sum(), fs=100)


# --- colours -----------------------------------------------------------------
def test_finding_colors_stable_and_distinct():
    colors = finding_colors(["AFIB", "IMI", "LVH"])
    assert colors["AFIB"] != colors["IMI"]
    assert finding_colors(["AFIB", "IMI", "LVH"]) == colors  # deterministic


# --- ECG figure --------------------------------------------------------------
def test_ecg_figure_builds_with_overlays():
    import matplotlib.pyplot as plt

    clean = np.random.default_rng(0).standard_normal((12, 1000)).astype(np.float32)
    sal = {"AFIB": _saliency("AFIB", top_lead=1), "IMI": _saliency("IMI", top_lead=2)}
    fig = ecg_figure(clean, sal, fs=100)
    assert fig.axes  # a real figure with an axis
    plt.close(fig)


def test_ecg_figure_highlight_runs():
    import matplotlib.pyplot as plt

    clean = np.zeros((12, 500), dtype=np.float32)
    sal = {"AFIB": _saliency("AFIB", top_lead=1, T=500)}
    fig = ecg_figure(clean, sal, fs=100, highlight="AFIB")
    plt.close(fig)


def test_ecg_figure_no_findings_ok():
    import matplotlib.pyplot as plt

    fig = ecg_figure(np.zeros((12, 500), dtype=np.float32), {}, fs=100)
    plt.close(fig)


# --- report HTML -------------------------------------------------------------
def test_report_html_has_findings_confidence_and_flags():
    report = _report(["AFIB", "IMI"], {"AFIB": 0.93, "IMI": 0.6})
    html = report_html(report)
    assert "AFIB" in html and "IMI" in html
    assert "width:" in html                      # confidence bars
    assert "REVIEW" in html                       # IMI (0.6) needs review
    assert "Impression" in html and "Full report" in html


def test_report_html_escapes_and_shows_consistency():
    report = _report(["NORM", "SR"], {"NORM": 0.98, "SR": 0.95})
    html = report_html(report)
    assert "consistent" in html.lower()


# --- banners -----------------------------------------------------------------
def test_disclaimer_html_has_required_text():
    html = disclaimer_html()
    assert "clinical decision-support tool" in html
    assert "should not be used as a standalone diagnosis" in html


def test_severity_banner_per_level():
    assert "Urgent" in severity_banner_html("red", ["STE_"])
    assert "STE_" in severity_banner_html("red", ["STE_"])
    assert "Review recommended" in severity_banner_html("yellow")
    assert "No urgent" in severity_banner_html("green")
