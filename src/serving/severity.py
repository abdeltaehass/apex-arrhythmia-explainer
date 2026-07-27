"""Triage severity for the dashboard banner (Phase 11).

Collapses a full :class:`~src.serving.schema.APEXReport` into one of three levels for
the top-of-screen banner:

    "red"    — an urgent, time-critical pattern is present (acute ST-elevation / injury,
               i.e. a STEMI-equivalent). Escalate now.
    "yellow" — review recommended (any reliability flag / low confidence / inconsistency
               / unreliable input), but nothing flagged as acutely urgent.
    "green"  — nothing surfaced that needs review.

Deliberately conservative: the urgent set is ST-elevation and subendocardial-injury
codes, the findings a clinician must not sit on. It is a *triage prompt*, never a
diagnosis — the disclaimer and the review gate still apply at every level.
"""

from __future__ import annotations

from src.serving.schema import APEXReport

# ST-elevation / acute injury (STEMI-equivalent) SCP codes — the "call someone now" set.
URGENT_CODES = frozenset({
    "STE_",                                  # ST-segment elevation
    "INJAS", "INJAL", "INJIN", "INJIL", "INJLA",  # subendocardial injury by territory
    "ANEUR",                                 # persistent ST-elevation (aneurysm morphology)
})

_LEVELS = {
    "red": {
        "label": "Urgent pattern detected",
        "detail": "Acute ST-elevation / injury pattern — escalate for immediate clinical review.",
        "color": "#b3261e", "bg": "#fce8e6",
    },
    "yellow": {
        "label": "Review recommended",
        "detail": "One or more findings need clinician confirmation before acting.",
        "color": "#8a6d00", "bg": "#fef7e0",
    },
    "green": {
        "label": "No urgent findings",
        "detail": "Nothing surfaced that requires review — still verify against the clinical picture.",
        "color": "#0b6b3a", "bg": "#e6f4ea",
    },
}


def severity(report: APEXReport) -> str:
    """Return the banner level ``"red" | "yellow" | "green"`` for a report."""
    surfaced = {f.label for f in report.findings}
    if surfaced & URGENT_CODES:
        return "red"
    if report.review_recommended:
        return "yellow"
    return "green"


def urgent_findings(report: APEXReport) -> list[str]:
    """The surfaced labels that drove a red banner (empty otherwise)."""
    return sorted({f.label for f in report.findings} & URGENT_CODES)


def banner_meta(level: str) -> dict:
    """Label / detail text / colours for a severity level, for the UI banner."""
    return _LEVELS[level]
