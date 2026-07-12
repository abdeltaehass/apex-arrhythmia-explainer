"""Hallucination flagging for generated explanations.

Two independent signals decide whether an explanation is withheld / flagged:

1. **Consistency** (see consistency.py): the text asserts a finding the detector
   never surfaced. This is a hard fail — the explanation is withheld.
2. **Grounding**: an asserted finding has no supporting saliency region (lead +
   time window) from the grounding layer. Ungrounded assertions are flagged.

The batch-level hallucination rate is what we log against the ≤ 0.02 target.
"""

from __future__ import annotations

from dataclasses import dataclass

from .consistency import ConsistencyResult


@dataclass
class HallucinationFlag:
    record_id: str
    unsupported_findings: set[str]   # asserted but not surfaced by detector
    ungrounded_findings: set[str]    # surfaced but with no saliency support
    flagged: bool                    # True -> explanation withheld / sent to review


def evaluate(
    record_id: str,
    consistency: ConsistencyResult,
    grounded_findings: set[str],
) -> HallucinationFlag:
    ungrounded = consistency.asserted - consistency.unsupported - grounded_findings
    flagged = bool(consistency.unsupported) or bool(ungrounded)
    return HallucinationFlag(
        record_id=record_id,
        unsupported_findings=consistency.unsupported,
        ungrounded_findings=ungrounded,
        flagged=flagged,
    )


def hallucination_rate(flags: list[HallucinationFlag]) -> float:
    """Fraction of records with at least one unsupported (fabricated) finding."""
    if not flags:
        return float("nan")
    return sum(bool(f.unsupported_findings) for f in flags) / len(flags)
