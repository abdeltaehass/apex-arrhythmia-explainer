"""Consistency checker: does the generated explanation only assert findings the
detector actually surfaced?

An explanation is *consistent* iff the set of diagnostic findings it mentions is a
subset of the labels the detector predicted above threshold. This is checked
programmatically on every eval run — it never relies on a human rater.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsistencyResult:
    consistent: bool
    asserted: set[str]          # findings the text claims
    surfaced: set[str]          # labels the detector predicted
    unsupported: set[str]       # asserted - surfaced (these are hallucinations)


def check(asserted_findings: set[str], surfaced_labels: set[str]) -> ConsistencyResult:
    unsupported = asserted_findings - surfaced_labels
    return ConsistencyResult(
        consistent=not unsupported,
        asserted=asserted_findings,
        surfaced=surfaced_labels,
        unsupported=unsupported,
    )


def consistency_rate(results: list[ConsistencyResult]) -> float:
    """Fraction of explanations that were fully consistent."""
    if not results:
        return float("nan")
    return sum(r.consistent for r in results) / len(results)
