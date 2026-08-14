"""Phase 27 — tests for Spanish clinical explanation.

Data-independent. The load-bearing tests are the *parity* ones: English output must be
byte-identical to the Phase-6 templater, the English reading path must agree with the
Phase-6 parser, and the hallucination gate must behave the same in both languages. Those
are what stop the Spanish path quietly becoming a second-class one.
"""

from __future__ import annotations

import random

import pytest

from src.generation.parse import asserted_findings as asserted_en
from src.generation.prompts import SYSTEM_PROMPT, system_prompt
from src.generation.templater import build_structured_input
from src.generation.templater import render_report as render_reference
from src.generation.vocab import VOCAB
from src.i18n import (
    VOCAB_ES,
    asserted_findings,
    check_consistency,
    detect_language,
    get_language,
    parse_report,
    render_report,
    render_text,
    strip_accents,
    supported_languages,
)


def si_for(codes, **kw):
    return build_structured_input(codes, confidences=dict.fromkeys(codes, 0.9), **kw)


def random_cases(n: int = 200, seed: int = 0):
    rng = random.Random(seed)
    codes = list(VOCAB)
    for _ in range(n):
        picked = rng.sample(codes, rng.randint(1, 5))
        yield build_structured_input(
            picked, confidences={c: rng.choice([0.3, 0.9]) for c in picked},
            heart_rate_bpm=rng.choice([None, 58, 72, 115]))


# --- vocabulary parity --------------------------------------------------------
def test_spanish_vocabulary_covers_every_code():
    assert set(VOCAB_ES) == set(VOCAB)


def test_spanish_entries_match_english_structure():
    """Group and territory drive rendering; a mismatch silently reorders the report."""
    for code, en in VOCAB.items():
        es = VOCAB_ES[code]
        assert es.group == en.group, code
        assert es.territory == en.territory, code
        assert (es.impression is None) == (en.impression is None), code


def test_spanish_text_is_not_english():
    for code, es in VOCAB_ES.items():
        assert es.finding != VOCAB[code].finding, f"{code} finding left untranslated"


def test_impression_terms_are_unique_enough_to_identify_a_code():
    terms = [e.impression for e in VOCAB_ES.values() if e.impression]
    assert len(terms) == len(set(terms)), "two codes share a Spanish impression term"


# --- rendering parity ---------------------------------------------------------
def test_english_rendering_is_byte_identical_to_the_phase6_templater():
    """The anti-drift invariant: one renderer, and it must not change English output."""
    for si in random_cases(200):
        assert render_report(si, "en") == render_reference(si)


def test_spanish_rendering_differs_and_is_well_formed():
    si = si_for(["AFIB", "ASMI"])
    es = render_report(si, "es")
    assert es != render_report(si, "en")
    assert "fibrilación auricular" in es["impression"]


def test_spanish_uses_localized_units_and_headers():
    si = si_for(["SR"], heart_rate_bpm=72)
    text = render_text(si, "es")
    assert "Hallazgos:" in text and "Impresión:" in text
    assert "lpm" in text and "bpm" not in text


def test_spanish_territory_adjectives_agree():
    """'derivaciones' is feminine plural — 'derivaciones inferior' would be wrong."""
    si = si_for(["IMI"])
    assert "derivaciones inferiores" in render_report(si, "es")["findings"]


def test_spanish_list_has_no_serial_comma():
    si = si_for(["ASMI"])
    findings = render_report(si, "es")["findings"]
    assert "V1, V2 y V3" in findings
    assert ", y V3" not in findings


def test_normal_study_collapses_in_both_languages():
    si = si_for(["NORM", "SR"])
    assert "Normal ECG" in render_report(si, "en")["impression"]
    assert "ECG normal" in render_report(si, "es")["impression"]


def test_low_confidence_hedge_is_translated():
    si = build_structured_input(["AFIB"], confidences={"AFIB": 0.2})
    assert "requiere confirmación clínica" in render_report(si, "es")["findings"]


# --- language resolution ------------------------------------------------------
def test_supported_languages():
    assert set(supported_languages()) == {"en", "es"}


@pytest.mark.parametrize("alias", ["es", "ES", "spa", "spanish", "español", "espanol"])
def test_language_aliases(alias):
    assert get_language(alias).code == "es"


def test_unsupported_language_raises_rather_than_falling_back():
    """Silently returning English to a caller who asked for Portuguese is a patient-facing bug."""
    with pytest.raises(ValueError, match="unsupported language"):
        get_language("pt")


def test_detect_language_from_headers():
    si = si_for(["AFIB"])
    assert detect_language(render_text(si, "es")) == "es"
    assert detect_language(render_text(si, "en")) == "en"
    assert detect_language("no headers here") is None


# --- the consistency gate -----------------------------------------------------
def test_english_parsing_agrees_with_the_phase6_parser():
    for si in random_cases(200, seed=1):
        text = render_text(si, "en")
        assert asserted_findings(text, "en") == asserted_en(text)


def test_spanish_report_round_trips_its_findings():
    si = si_for(["AFIB", "ASMI"])
    assert asserted_findings(render_text(si, "es"), "es") == {"AFIB", "ASMI"}


def test_gate_catches_a_fabricated_spanish_finding():
    """The hole this phase exists to close: before it, this passed."""
    si = si_for(["AFIB"])
    fabricated = render_text(si, "es") + " Bloqueo completo de rama izquierda."
    result = check_consistency(fabricated, {"AFIB"}, "es")
    assert not result.consistent and "CLBBB" in result.unsupported


def test_english_only_parser_misses_spanish_fabrications():
    """Regression guard on the bug itself — if this ever passes, the gate is language-bound again."""
    si = si_for(["AFIB"])
    fabricated = render_text(si, "es") + " Bloqueo completo de rama izquierda."
    assert asserted_en(fabricated) == set(), "English parser unexpectedly reads Spanish"


def test_gate_is_accent_insensitive():
    """A missing accent must not disable the guardrail."""
    si = si_for(["AFIB"])
    fabricated = render_text(si, "es") + " Bloqueo completo de rama izquierda."
    stripped = strip_accents(fabricated)
    assert "CLBBB" in check_consistency(stripped, {"AFIB"}, "es").unsupported


def test_nested_terms_do_not_produce_false_positives():
    """'extrasístoles' is a substring of 'extrasístoles auriculares'."""
    si = si_for(["PAC"])
    text = render_text(si, "es")
    assert asserted_findings(text, "es") == {"PAC"}
    assert check_consistency(text, {"PAC"}, "es").consistent


def test_morphology_in_findings_is_not_read_as_a_diagnosis():
    """ISCAN's finding sentence is verbatim INVT's impression term."""
    si = si_for(["ISCAN"])
    for lang in ("en", "es"):
        assert "INVT" not in asserted_findings(render_text(si, lang), lang)


def test_parse_report_recovers_accented_text():
    si = si_for(["AFIB"])
    parsed = parse_report(render_text(si, "es"), "es")
    assert parsed.well_formed and parsed.language == "es"
    assert "fibrilación" in parsed.impression


# --- prompts ------------------------------------------------------------------
def test_english_system_prompt_is_unchanged():
    assert system_prompt("en") == SYSTEM_PROMPT


def test_spanish_prompt_extends_rather_than_replaces_the_rules():
    """Translating the constraint block risks softening the rule that holds the gate up."""
    es = system_prompt("es")
    assert es.startswith(SYSTEM_PROMPT)
    assert "Hallazgos:" in es and "Impresión:" in es


# --- serving ------------------------------------------------------------------
def test_build_report_uses_the_right_language_for_consistency():
    from src.serving.serializer import build_report

    si = si_for(["AFIB"])
    fabricated = render_text(si, "es") + " Bloqueo completo de rama izquierda."
    assert not build_report(si, fabricated, lang="es").consistency.consistent
    # and the old behaviour, for contrast
    assert build_report(si, fabricated).consistency.consistent
