"""Unit tests for the Phase-6 generation pipeline (no model download, no PTB-XL data)."""

import numpy as np
import pytest

from src.eval.consistency import check
from src.eval.hallucination import evaluate
from src.generation.dataset import (
    axis_phrase,
    build_example,
    example_to_structured_input,
    sample_confidences,
)
from src.generation.parse import asserted_findings, parse_report
from src.generation.prompts import (
    build_chat_example,
    build_prompt_completion_example,
    build_user_prompt,
    serialize_structured_input,
)
from src.generation.templater import build_structured_input, render_report
from src.generation.vocab import VOCAB, impression_terms


# --- vocab --------------------------------------------------------------
def test_vocab_covers_all_71_codes():
    assert len(VOCAB) == 71


def test_every_impression_phrase_is_unique_text():
    terms = impression_terms()
    phrases = [e.impression for e in VOCAB.values() if e.impression]
    assert len(terms) == len(set(phrases))  # no two codes share literal text


# --- templater ------------------------------------------------------------
def test_render_report_afib():
    si = build_structured_input(["AFIB"], confidences={"AFIB": 0.93}, heart_rate_bpm=132)
    rep = render_report(si)
    assert "irregularly irregular" in rep["findings"].lower()
    assert "atrial fibrillation" in rep["impression"].lower()
    assert "no acute ischemic changes" in rep["impression"].lower()


def test_render_report_normal():
    si = build_structured_input(["NORM", "SR"], confidences={"NORM": 0.99, "SR": 0.95})
    rep = render_report(si)
    assert rep["impression"] == "Normal ECG. No acute abnormality."


def test_render_report_sr_alone_is_not_normal():
    # SR without NORM must NOT collapse to "Normal ECG" (that would assert a code
    # -- NORM -- the input never contained; see docstring in templater.render_report).
    si = build_structured_input(["SR"], confidences={"SR": 0.9})
    rep = render_report(si)
    assert "sinus rhythm" in rep["impression"].lower()
    assert "normal ecg" not in rep["impression"].lower()


@pytest.mark.parametrize("pathological, phrase", [
    ("AFIB", "atrial fibrillation"), ("PVC", "ventricular premature complexes"),
    ("IMI", "inferior myocardial infarction"), ("CRBBB", "complete right bundle branch block"),
])
def test_norm_does_not_swallow_a_co_labeled_pathological_finding(pathological, phrase):
    # Found during the Phase-6 manual review (docs/generation/examples_review.md,
    # cases 4 & 17): PTB-XL occasionally co-labels NORM with a real abnormality (e.g.
    # AFIB, PVC). The old `is_normal = "NORM" in codes` collapsed the Impression to a
    # bare "Normal ECG.", silently dropping the other finding even though Findings
    # still described it. Only genuinely benign rate/rhythm variants may still read
    # as "normal" alongside NORM -- anything else must still be *named*, whether or
    # not "Normal ECG" also appears alongside it (both codes were surfaced, after all).
    si = build_structured_input([pathological, "NORM"],
                                confidences={pathological: 0.9, "NORM": 0.9})
    rep = render_report(si)
    assert phrase in rep["impression"].lower(), (
        f"{pathological}+NORM must still name {phrase!r}, not just report 'Normal ECG'"
    )


@pytest.mark.parametrize("benign", ["SBRAD", "STACH", "SARRH"])
def test_norm_still_collapses_with_benign_rate_variants(benign):
    # The other half of the same fix: SBRAD/STACH/SARRH are physiological rate/rhythm
    # variants, not pathology, so "Normal ECG, sinus bradycardia" is a legitimate,
    # clinically coherent reading -- unlike NORM+AFIB or NORM+PVC above.
    si = build_structured_input([benign, "NORM"], confidences={benign: 0.9, "NORM": 0.9})
    rep = render_report(si)
    assert "normal ecg" in rep["impression"].lower()


def test_low_confidence_flags_requires_confirmation():
    si = build_structured_input(["PVC"], confidences={"PVC": 0.3}, review_threshold=0.5)
    rep = render_report(si)
    assert "requires clinician confirmation" in rep["findings"]


def test_ischemic_finding_omits_no_acute_ischemic_changes():
    si = build_structured_input(["IMI"], confidences={"IMI": 0.9})
    rep = render_report(si)
    assert "no acute ischemic changes" not in rep["impression"].lower()


def test_axis_phrase_no_duplicate_word():
    si = build_structured_input(["CRBBB"], confidences={"CRBBB": 0.8}, heart_axis="right axis deviation")
    rep = render_report(si)
    assert "axis axis" not in rep["findings"].lower()
    assert "right axis deviation" in rep["findings"].lower()


def test_multi_territory_ischemia_merges_into_one_sentence():
    si = build_structured_input(["ISCAN", "ISCIN"], confidences={"ISCAN": 0.7, "ISCIN": 0.65})
    rep = render_report(si)
    assert "anterior" in rep["findings"].lower() and "inferior" in rep["findings"].lower()
    assert rep["findings"].lower().count("t-wave inversion") == 1  # merged, not repeated


def test_unknown_code_is_skipped():
    si = build_structured_input(["NOT_A_REAL_CODE", "NORM"])
    assert si.codes() == ["NORM"]


# --- round trip: every code renders -> re-parses to itself -----------------
@pytest.mark.parametrize("code", sorted(c for c, e in VOCAB.items() if e.impression))
def test_round_trip_every_code(code):
    si = build_structured_input([code], confidences={code: 0.9})
    rep = render_report(si)
    full = f"Findings:\n{rep['findings']}\n\nImpression:\n{rep['impression']}"
    assert asserted_findings(full) == {code}


def test_substring_collision_does_not_double_count():
    # "lateral ischemia" (ISCLA) is a literal substring of "anterolateral ischemia" (ISCAL)
    text = "Findings:\nT-wave inversion.\n\nImpression:\nAnterolateral ischemia."
    assert asserted_findings(text) == {"ISCAL"}


# --- parse ------------------------------------------------------------------
def test_parse_report_well_formed():
    text = "Findings:\nSomething.\n\nImpression:\nSomething else."
    p = parse_report(text)
    assert p.well_formed
    assert p.findings == "Something."
    assert p.impression == "Something else."


def test_parse_report_malformed():
    p = parse_report("just some free text with no headers")
    assert not p.well_formed
    assert p.impression == ""


def test_parse_report_tolerates_markdown_bold():
    text = "**Findings:**\nA.\n\n**Impression:**\nB."
    p = parse_report(text)
    assert p.well_formed
    assert p.findings == "A."


# --- dataset ------------------------------------------------------------
def test_sample_confidences_bounds():
    rng = np.random.default_rng(0)
    conf = sample_confidences(["AFIB", "NORM", "IMI"] * 20, rng, low_conf_prob=0.15)
    assert all(0.0 <= v <= 1.0 for v in conf.values())


def test_axis_phrase_surfaces_only_unambiguous_codes():
    assert axis_phrase("LAD") == "left axis deviation"
    assert axis_phrase("MID") is None  # tracked but not surfaced
    assert axis_phrase("garbage") is None
    assert axis_phrase(None) is None


class _Row(dict):
    """Minimal stand-in for a pandas Series (attribute-style get + .name)."""

    def get(self, k, default=None):
        return self[k] if k in self and self[k] is not None else default

    name = 12345


def test_build_example_round_trips_through_structured_input():
    row = _Row(scp_codes={"AFIB": 0.0, "CRBBB": 100.0}, sex=0, age=61, heart_axis="RAD", report="afib crbbb")
    rng = np.random.default_rng(1)
    ex = build_example(row, rng, heart_rate_bpm=110.0)
    assert ex["codes"] == ["AFIB", "CRBBB"]
    assert ex["sex"] == "M"
    si = example_to_structured_input(ex)
    rep = render_report(si)
    assert rep["findings"] == ex["findings"]
    assert rep["impression"] == ex["impression"]


def test_build_example_returns_none_for_no_recognized_codes():
    row = _Row(scp_codes={"ZZZNOTREAL": 0.0}, sex=1, age=40, heart_axis=None, report="")
    rng = np.random.default_rng(2)
    assert build_example(row, rng) is None


# --- prompts ------------------------------------------------------------
def test_serialize_structured_input_lists_every_finding():
    si = build_structured_input(["AFIB", "IMI"], confidences={"AFIB": 0.9, "IMI": 0.8})
    block = serialize_structured_input(si)
    assert "AFIB" in block and "IMI" in block
    assert "Review threshold" in block


def test_build_chat_example_shape():
    si = build_structured_input(["NORM"], confidences={"NORM": 0.95})
    msgs = build_chat_example(si, "Normal.", "Normal ECG.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == "Findings:\nNormal.\n\nImpression:\nNormal ECG."
    assert "Write the report now." in build_user_prompt(si)


def test_build_prompt_completion_example_shape():
    si = build_structured_input(["NORM"], confidences={"NORM": 0.95})
    ex = build_prompt_completion_example(si, "Normal.", "Normal ECG.")
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert [m["role"] for m in ex["completion"]] == ["assistant"]
    assert ex["completion"][0]["content"] == "Findings:\nNormal.\n\nImpression:\nNormal ECG."
    # same conversation as build_chat_example, just split at the assistant turn
    assert ex["prompt"] + ex["completion"] == build_chat_example(si, "Normal.", "Normal ECG.")


# --- integration with src/eval (consistency + hallucination) --------------
def test_template_output_is_always_consistent_with_its_own_input():
    si = build_structured_input(["AFIB", "CRBBB"], confidences={"AFIB": 0.9, "CRBBB": 0.85})
    rep = render_report(si)
    full = f"Findings:\n{rep['findings']}\n\nImpression:\n{rep['impression']}"
    result = check(asserted_findings(full), set(si.codes()))
    assert result.consistent
    assert result.unsupported == set()


def test_hallucinated_finding_is_flagged():
    # detector only surfaced AFIB, but the "generated" text also names CRBBB
    surfaced = {"AFIB"}
    text = ("Findings:\nIrregular rhythm.\n\n"
           "Impression:\nAtrial fibrillation. Complete right bundle branch block.")
    result = check(asserted_findings(text), surfaced)
    assert not result.consistent
    assert result.unsupported == {"CRBBB"}
    flag = evaluate("rec1", result, grounded_findings=surfaced)
    assert flag.flagged
