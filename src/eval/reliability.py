"""Phase 7 — Consistency & Reliability Checker.

Composes four independent checks into one report per (recording, explanation):

1. **Consistency warning** — the explanation names a finding (e.g. atrial
   fibrillation) the detector did not surface above `review_threshold`. Reuses
   `src/eval/consistency.py`; this module just names it a "warning" and gives it a
   human-readable message, per the Phase-7 spec's own wording.
2. **Grounding conflict** — the explanation cites *specific leads* for a finding (e.g.
   "ST elevation in V2, V3") that rank among the *least*-important leads in the
   grounding layer's own per-lead saliency for that finding (`LeadSaliency.
   lead_importance`, `src/grounding/`). Rank-based rather than an absolute magnitude
   threshold: guided Grad-CAM's per-lead signal is comparatively mild (the CNN mixes
   all leads at its first layer, per the Phase-5 grounding writeup), so an absolute
   bar borrowed from `grounding.is_grounded` (tuned for a single whole-signal trace,
   not a 12-way comparison) flags nearly every lead as "unsupported" — checked
   empirically against the validation set before picking this design; see
   `docs/reliability/report.md`. This is **lead-level**, finer-grained than
   `src/eval/hallucination.py`'s whole-finding "ungrounded" check.
3. **Low-confidence flag** — any surfaced finding below a *separate*, higher, tunable
   threshold (default 0.7, `CFG.low_confidence_threshold`) is tagged "low confidence —
   manual review recommended", even if it already cleared the base 0.5
   surfacing/`review_threshold` bar. A defense-in-depth re-check at the eval layer,
   independent of whatever the generator already did.
4. **Mutual exclusivity** — the detector fires above threshold on two labels that
   cannot both be true at once (e.g. sinus rhythm + atrial fibrillation). Operates
   purely on the detector's surfaced-label set, independent of any generated text.

None of these checks trusts the model's self-report: 1 and 2 check the *generated
text* against independent signals (the detector, the grounding layer); 3 and 4 check
the *detector output* against itself. See `docs/reliability/report.md` for flag rates
measured on the validation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import CFG
from src.eval.consistency import ConsistencyResult, check
from src.generation.templater import NORM_COMPATIBLE
from src.generation.vocab import VOCAB
from src.grounding.saliency import LEAD_NAMES

LEAD_INDEX: dict[str, int] = {name: i for i, name in enumerate(LEAD_NAMES)}


# --- Mutual exclusivity rules -----------------------------------------------
# Built once, at import time, into a flat set of (frozenset({a, b}), reason) pairs —
# every rule below is a *definitional* clinical contradiction, not merely an unusual
# combination (AFIB+AFLT, for instance, genuinely co-occur in PTB-XL and are NOT listed).

# Codes whose own finding text (vocab.py) requires an organized, sinus-node-origin
# atrial rhythm with discernible P waves — definitionally incompatible with AFIB/AFLT,
# whose finding text requires the *absence* of organized P waves.
_SINUS_ORGANIZED = {"SR", "STACH", "SBRAD", "SARRH"}
_DISORGANIZED_ATRIAL = {"AFIB", "AFLT"}

# (name, group_a, group_b, reason) — every cross pair a-in-A, b-in-B conflicts.
_CROSS_GROUP_RULES: list[tuple[set[str], set[str], str]] = [
    (_SINUS_ORGANIZED, _DISORGANIZED_ATRIAL,
     "a sinus-origin rhythm requires organized P waves, which {b} defines as absent"),
]

# (a, b, reason) — single explicit pairs.
_PAIR_RULES: list[tuple[str, str, str]] = [
    ("SBRAD", "STACH", "a rate cannot be simultaneously below 60 bpm and above 100 bpm"),
    ("CRBBB", "CLBBB", "both bundles cannot be completely blocked at once"),
    ("IRBBB", "CRBBB", "the right bundle cannot be both incompletely and completely blocked"),
    ("ILBBB", "CLBBB", "the left bundle cannot be both incompletely and completely blocked"),
]

# (group, reason) — at most one member of the group may be positive.
_AT_MOST_ONE_RULES: list[tuple[set[str], str]] = [
    ({"1AVB", "2AVB", "3AVB"}, "AV block degree is a single measurement, not three"),
]


def _build_exclusivity_pairs() -> dict[frozenset, str]:
    pairs: dict[frozenset, str] = {}
    for group_a, group_b, reason in _CROSS_GROUP_RULES:
        for a in group_a:
            for b in group_b:
                pairs[frozenset((a, b))] = reason.format(b=b)
    for a, b, reason in _PAIR_RULES:
        pairs[frozenset((a, b))] = reason
    for group, reason in _AT_MOST_ONE_RULES:
        members = sorted(group)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs[frozenset((a, b))] = reason
    # NORM is compatible only with physiological rate/rhythm variants (see
    # templater.NORM_COMPATIBLE); any other code co-occurring with it above threshold
    # is the same tension found and documented in the Phase-6 review.
    for code, entry in VOCAB.items():
        if code in NORM_COMPATIBLE or entry.impression is None:
            continue
        pairs[frozenset(("NORM", code))] = (
            f"NORM is only clinically compatible with {sorted(NORM_COMPATIBLE)}, not {code}"
        )
    return pairs


EXCLUSIVITY_PAIRS: dict[frozenset, str] = _build_exclusivity_pairs()


# --- Dataclasses -------------------------------------------------------------
@dataclass
class ConsistencyWarning:
    code: str
    message: str


@dataclass
class GroundingConflict:
    code: str
    lead: str
    message: str


@dataclass
class LowConfidenceFlag:
    code: str
    confidence: float
    threshold: float
    message: str = "low confidence — manual review recommended"


@dataclass
class MutualExclusivityConflict:
    code_a: str
    code_b: str
    reason: str
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"{self.code_a} and {self.code_b} fired together: {self.reason}"


@dataclass
class ReliabilityReport:
    record_id: str
    consistency: ConsistencyResult
    consistency_warnings: list[ConsistencyWarning]
    grounding_conflicts: list[GroundingConflict]
    low_confidence: list[LowConfidenceFlag]
    mutual_exclusivity: list[MutualExclusivityConflict]

    @property
    def any_flag(self) -> bool:
        return bool(
            self.consistency_warnings or self.grounding_conflicts
            or self.low_confidence or self.mutual_exclusivity
        )


# --- Individual checks --------------------------------------------------------
def check_consistency_warnings(
    asserted: set[str], surfaced: set[str],
) -> tuple[ConsistencyResult, list[ConsistencyWarning]]:
    """Findings the text names that the detector never surfaced above threshold."""
    result = check(asserted, surfaced)
    warnings = []
    for code in sorted(result.unsupported):
        desc = VOCAB[code].impression if code in VOCAB and VOCAB[code].impression else code
        warnings.append(ConsistencyWarning(
            code=code,
            message=f"text mentions {desc} ({code}) but the detector did not flag it above threshold",
        ))
    return result, warnings


def check_grounding_conflicts(
    leads_by_code: dict[str, list[str]],
    saliency_by_code: dict,
    bottom_k: int = 2,
) -> list[GroundingConflict]:
    """Cited leads that rank among the least-important for their finding.

    ``leads_by_code``: code -> the specific leads the explanation cites for it (e.g.
    from `vocab.leads_for` / a `StructuredInput`'s `Finding.leads`).
    ``saliency_by_code``: code -> a `grounding.saliency.LeadSaliency` (from
    `src.grounding.ground`). Codes absent from either dict are skipped (nothing to
    check), not flagged.

    A cited lead conflicts if it ranks in the bottom ``bottom_k`` of all 12 leads by
    ``lead_importance`` (the model's own per-lead saliency mass for that finding) —
    e.g. ``bottom_k=2`` flags a cited lead that is the least- or second-least-
    important lead the model considered, even though the explanation names it as the
    evidence. Rank-based, not an absolute magnitude bar — see the module docstring
    for why.
    """
    conflicts = []
    for code, leads in leads_by_code.items():
        sal = saliency_by_code.get(code)
        if sal is None or not leads:
            continue
        # rank 1 = most important lead for this finding, 12 = least important
        order = list(np.argsort(sal.lead_importance)[::-1])
        ranks = {LEAD_NAMES[idx]: r + 1 for r, idx in enumerate(order)}
        n_leads = len(sal.lead_importance)
        for lead in leads:
            rank = ranks.get(lead)
            if rank is not None and rank > n_leads - bottom_k:
                conflicts.append(GroundingConflict(
                    code=code, lead=lead,
                    message=(f"{code} cites lead {lead}, but it ranks {rank}/{n_leads} in the "
                            f"model's own per-lead saliency for this finding"),
                ))
    return conflicts


def check_low_confidence(
    confidences: dict[str, float],
    threshold: float = CFG.low_confidence_threshold,
) -> list[LowConfidenceFlag]:
    """Surfaced findings below the (higher, separate) manual-review confidence bar."""
    return [
        LowConfidenceFlag(code=code, confidence=conf, threshold=threshold)
        for code, conf in sorted(confidences.items())
        if conf < threshold
    ]


def check_mutual_exclusivity(surfaced: set[str]) -> list[MutualExclusivityConflict]:
    """Detector-only check: any two surfaced labels that are clinical contradictions."""
    conflicts = []
    codes = sorted(surfaced)
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            reason = EXCLUSIVITY_PAIRS.get(frozenset((a, b)))
            if reason:
                conflicts.append(MutualExclusivityConflict(code_a=a, code_b=b, reason=reason))
    return conflicts


# --- Umbrella entrypoint -----------------------------------------------------
def check_reliability(
    record_id: str,
    asserted: set[str],
    surfaced: set[str],
    confidences: dict[str, float],
    leads_by_code: dict[str, list[str]] | None = None,
    saliency_by_code: dict | None = None,
    low_confidence_threshold: float = CFG.low_confidence_threshold,
    grounding_bottom_k: int = 2,
) -> ReliabilityReport:
    """Run all four Phase-7 checks and bundle them into one report.

    ``leads_by_code``/``saliency_by_code`` are optional — omit them (e.g. when
    grounding wasn't computed for this record, which is expensive) and the grounding
    check simply returns no conflicts rather than failing.
    """
    consistency, consistency_warnings = check_consistency_warnings(asserted, surfaced)
    grounding_conflicts = (
        check_grounding_conflicts(leads_by_code, saliency_by_code, grounding_bottom_k)
        if leads_by_code and saliency_by_code else []
    )
    surfaced_confidences = {c: v for c, v in confidences.items() if c in surfaced}
    low_confidence = check_low_confidence(surfaced_confidences, low_confidence_threshold)
    mutual_exclusivity = check_mutual_exclusivity(surfaced)
    return ReliabilityReport(
        record_id=record_id,
        consistency=consistency,
        consistency_warnings=consistency_warnings,
        grounding_conflicts=grounding_conflicts,
        low_confidence=low_confidence,
        mutual_exclusivity=mutual_exclusivity,
    )


def summarize(reports: list[ReliabilityReport]) -> dict:
    """Flag-rate breakdown across a batch — the numbers `docs/reliability/report.md` reports."""
    n = len(reports)
    if n == 0:
        return {"n_total": 0}
    return {
        "n_total": n,
        "consistency_warning_rate": sum(bool(r.consistency_warnings) for r in reports) / n,
        "grounding_conflict_rate": sum(bool(r.grounding_conflicts) for r in reports) / n,
        "low_confidence_rate": sum(bool(r.low_confidence) for r in reports) / n,
        "mutual_exclusivity_rate": sum(bool(r.mutual_exclusivity) for r in reports) / n,
        "any_flag_rate": sum(r.any_flag for r in reports) / n,
        "n_consistency_warnings": sum(len(r.consistency_warnings) for r in reports),
        "n_grounding_conflicts": sum(len(r.grounding_conflicts) for r in reports),
        "n_low_confidence_flags": sum(len(r.low_confidence) for r in reports),
        "n_mutual_exclusivity_conflicts": sum(len(r.mutual_exclusivity) for r in reports),
    }
