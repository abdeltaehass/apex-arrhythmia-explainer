"""Unit tests for the Phase-7 reliability checker (no model/data needed)."""

import numpy as np
import pytest

from src.eval.reliability import (
    EXCLUSIVITY_PAIRS,
    check_consistency_warnings,
    check_grounding_conflicts,
    check_low_confidence,
    check_mutual_exclusivity,
    check_reliability,
    summarize,
)
from src.generation.vocab import VOCAB
from src.grounding.saliency import LEAD_NAMES, LeadSaliency


def _saliency_with_importances(values: dict[str, float], code: str = "X", t: int = 100) -> LeadSaliency:
    """Build a `LeadSaliency` whose `lead_importance` ranking is exactly controlled --
    each lead's `per_lead` row is a constant equal to its target value, so the
    resulting mass (and hence rank) is unambiguous, not an artifact of argsort ties."""
    per_lead = np.array([[values.get(name, 0.1)] * t for name in LEAD_NAMES], dtype=np.float32)
    mass = per_lead.sum(axis=1)
    return LeadSaliency(
        label_index=0, label=code, method="guided_gradcam", logit=1.0, prob=0.9,
        per_lead=per_lead, temporal=per_lead.max(axis=0), lead_importance=mass / mass.sum(), fs=100,
    )


# --- consistency warnings ----------------------------------------------------
def test_consistency_warning_fires_for_unsurfaced_finding():
    result, warnings = check_consistency_warnings({"AFIB"}, {"SR"})
    assert not result.consistent
    assert len(warnings) == 1
    assert warnings[0].code == "AFIB"
    assert "detector did not flag it" in warnings[0].message


def test_consistency_warning_silent_when_supported():
    result, warnings = check_consistency_warnings({"AFIB"}, {"AFIB", "SR"})
    assert result.consistent
    assert warnings == []


# --- grounding conflicts ------------------------------------------------------
# II ranks 2nd (well-grounded), V1 ranks 12th/last (the bottom_k=2 default catches it).
_RANKED = {"I": 12, "II": 11, "III": 10, "aVR": 9, "aVL": 8, "aVF": 7,
          "V1": 1, "V2": 6, "V3": 5, "V4": 4, "V5": 3, "V6": 2}
assert len(_RANKED) == len(LEAD_NAMES)


def test_grounding_conflict_flags_bottom_ranked_cited_lead():
    sal = _saliency_with_importances(_RANKED, "IMI")
    conflicts = check_grounding_conflicts({"IMI": ["II", "V1"]}, {"IMI": sal})
    assert len(conflicts) == 1
    assert conflicts[0].code == "IMI"
    assert conflicts[0].lead == "V1"
    assert "12/12" in conflicts[0].message


def test_grounding_conflict_silent_when_all_cited_leads_well_ranked():
    sal = _saliency_with_importances(_RANKED, "IMI")
    conflicts = check_grounding_conflicts({"IMI": ["I", "II"]}, {"IMI": sal})  # ranks 1, 2
    assert conflicts == []


def test_grounding_conflict_bottom_k_is_tunable():
    sal = _saliency_with_importances(_RANKED, "IMI")
    # V6 ranks 2nd-worst (11th/12); default bottom_k=2 catches it, bottom_k=1 does not
    assert check_grounding_conflicts({"IMI": ["V6"]}, {"IMI": sal}, bottom_k=2) != []
    assert check_grounding_conflicts({"IMI": ["V6"]}, {"IMI": sal}, bottom_k=1) == []


def test_grounding_conflict_skips_codes_without_saliency_data():
    # a surfaced code with leads but no computed saliency is skipped, not flagged --
    # grounding is expensive and may not have been run for every surfaced label.
    conflicts = check_grounding_conflicts({"IMI": ["II", "III"]}, {})
    assert conflicts == []


def test_grounding_conflict_skips_findings_with_no_cited_leads():
    sal = _saliency_with_importances(_RANKED, "ISC_")
    conflicts = check_grounding_conflicts({"ISC_": []}, {"ISC_": sal})
    assert conflicts == []


# --- low confidence -----------------------------------------------------------
def test_low_confidence_default_threshold_is_point_seven():
    flags = check_low_confidence({"IMI": 0.6, "AFIB": 0.9})
    assert len(flags) == 1
    assert flags[0].code == "IMI"
    assert flags[0].threshold == 0.7
    assert flags[0].message == "low confidence — manual review recommended"


def test_low_confidence_custom_threshold():
    flags = check_low_confidence({"IMI": 0.6}, threshold=0.5)
    assert flags == []  # 0.6 clears a 0.5 bar even though it wouldn't clear 0.7


# --- mutual exclusivity --------------------------------------------------------
def test_norm_plus_afib_conflicts():
    conflicts = check_mutual_exclusivity({"NORM", "AFIB"})
    assert len(conflicts) == 1
    assert {conflicts[0].code_a, conflicts[0].code_b} == {"NORM", "AFIB"}


def test_sr_plus_afib_conflicts_sinus_vs_disorganized():
    conflicts = check_mutual_exclusivity({"SR", "AFIB"})
    assert len(conflicts) == 1


def test_afib_plus_aflt_does_not_conflict():
    # these genuinely co-occur in PTB-XL (alternating/ambiguous rhythm strips) --
    # must NOT be treated as a hard contradiction.
    assert check_mutual_exclusivity({"AFIB", "AFLT"}) == []


def test_sbrad_plus_norm_does_not_conflict():
    # benign rate variant + NORM is clinically coherent (Phase-6 NORM_COMPATIBLE).
    assert check_mutual_exclusivity({"SBRAD", "NORM"}) == []


def test_sbrad_plus_stach_conflicts():
    conflicts = check_mutual_exclusivity({"SBRAD", "STACH"})
    assert len(conflicts) == 1


def test_crbbb_plus_clbbb_conflicts():
    assert len(check_mutual_exclusivity({"CRBBB", "CLBBB"})) == 1


def test_av_block_degrees_mutually_exclusive():
    assert len(check_mutual_exclusivity({"1AVB", "3AVB"})) == 1
    assert len(check_mutual_exclusivity({"1AVB", "2AVB", "3AVB"})) == 3  # all 3 pairs


def test_sr_plus_1avb_does_not_conflict():
    # AV block describes conduction, not atrial rhythm -- fully compatible with sinus.
    assert check_mutual_exclusivity({"SR", "1AVB"}) == []


def test_no_conflict_among_unrelated_codes():
    assert check_mutual_exclusivity({"IMI", "LVH", "STD_"}) == []


def test_single_label_never_conflicts():
    assert check_mutual_exclusivity({"AFIB"}) == []
    assert check_mutual_exclusivity(set()) == []


def test_exclusivity_pairs_only_contain_known_codes():
    known = set(VOCAB)
    for pair in EXCLUSIVITY_PAIRS:
        assert pair <= known, f"unknown code in exclusivity pair {pair}"


# --- umbrella + summarize -----------------------------------------------------
def test_check_reliability_bundles_all_four_checks():
    r = check_reliability(
        "rec1",
        asserted={"AFIB"},            # not surfaced -> consistency warning
        surfaced={"SR", "NORM"},       # SR+NORM is fine, but AFIB missing triggers warning above
        confidences={"SR": 0.9, "NORM": 0.6},  # NORM below 0.7 -> low-confidence flag
    )
    assert len(r.consistency_warnings) == 1
    assert len(r.low_confidence) == 1
    assert r.mutual_exclusivity == []  # SR+NORM is in NORM_COMPATIBLE, no conflict
    assert r.grounding_conflicts == []  # no leads/saliency provided
    assert r.any_flag


def test_check_reliability_no_flags_on_a_clean_record():
    r = check_reliability(
        "rec2", asserted={"NORM"}, surfaced={"NORM"}, confidences={"NORM": 0.95},
    )
    assert not r.any_flag


def test_summarize_rates():
    clean = check_reliability("a", asserted={"NORM"}, surfaced={"NORM"}, confidences={"NORM": 0.9})
    flagged = check_reliability(
        "b", asserted={"AFIB"}, surfaced={"NORM"}, confidences={"NORM": 0.6},
    )
    summary = summarize([clean, flagged])
    assert summary["n_total"] == 2
    assert summary["consistency_warning_rate"] == pytest.approx(0.5)
    assert summary["low_confidence_rate"] == pytest.approx(0.5)
    assert summary["any_flag_rate"] == pytest.approx(0.5)


def test_summarize_empty():
    assert summarize([]) == {"n_total": 0}
