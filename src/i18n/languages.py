"""Phase 27 — the phrase bank: everything in a report that is not a clinical term.

A report is more than its findings. It has section headers, a rate sentence, lead clauses,
list joiners, hedges, and a set of fixed impression sentences — and every one of them was
hardcoded English in the Phase-6 templater. Translating the vocabulary alone would produce
Spanish terms embedded in English scaffolding.

Each language is one :class:`PhraseBank`. English is defined here too, rather than left
implicit in the templater, so that :mod:`src.i18n.render` has a single code path for all
languages. ``tests/test_i18n.py`` asserts the English output of that path is byte-identical
to `templater.render_report`, which is what stops the two implementations drifting apart —
the usual fate of a "translated" second renderer.

Grammar the bank has to carry, not just words:

- **Adjective agreement.** Spanish territory names agree with *derivaciones* (feminine
  plural), so the lead clause needs "derivaciones inferiores", not "derivaciones inferior".
  The agreeing forms live in :data:`~src.i18n.vocab_es.TERRITORIES_ES`.
- **List punctuation.** English takes a serial comma before "and"; Spanish does not use one
  before "y". Reusing the English joiner would produce a comma splice in every multi-finding
  report.
- **Units.** Beats per minute is *lpm* (latidos por minuto), not *bpm*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.vocab import TERRITORIES, VOCAB, Entry
from src.i18n.vocab_es import TERRITORIES_ES, VOCAB_ES

DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class PhraseBank:
    """Every non-clinical string a rendered report needs, for one language."""

    code: str
    name: str
    vocab: dict[str, Entry]
    territories: dict[str, str]

    # section structure
    findings_header: str
    impression_header: str

    # findings section
    rate_label: str                 # "Rate" / "Frecuencia"
    ventricular_rate_label: str
    rate_template: str              # "{label} approximately {bpm} bpm"
    axis_word: str                  # "axis" / "eje"
    low_confidence: str             # "(requires clinician confirmation)"
    lead_clause_territory: str      # " in the {territory} leads ({leads})"
    lead_clause_plain: str          # " in leads {leads}"
    merged_territory_clause: str    # "{core} in the {parts} leads"
    no_findings: str

    # impression section
    consistent_with: str            # "Findings consistent with {term}."
    normal_only: str
    normal_with_extra: str          # "Normal ECG, with {extra}. No acute abnormality."
    non_specific: str
    no_acute_ischemia: str

    # list joining
    two_joiner: str                 # " and " / " y "
    serial_joiner: str              # ", and " / " y "  (Spanish takes no serial comma)
    list_separator: str = ", "
    extra_separator: str = ", "

    aliases: tuple[str, ...] = field(default_factory=tuple)


ENGLISH = PhraseBank(
    code="en",
    name="English",
    vocab=VOCAB,
    territories={k: k for k in TERRITORIES},
    findings_header="Findings",
    impression_header="Impression",
    rate_label="Rate",
    ventricular_rate_label="Ventricular rate",
    rate_template="{label} approximately {bpm} bpm",
    axis_word="axis",
    low_confidence="(requires clinician confirmation)",
    lead_clause_territory=" in the {territory} leads ({leads})",
    lead_clause_plain=" in leads {leads}",
    merged_territory_clause="{core} in the {parts} leads",
    no_findings="No structured findings provided.",
    consistent_with="Findings consistent with {term}.",
    normal_only="Normal ECG. No acute abnormality.",
    normal_with_extra="Normal ECG, with {extra}. No acute abnormality.",
    non_specific="Non-specific findings, as above.",
    no_acute_ischemia="No acute ischemic changes identified.",
    two_joiner=" and ",
    serial_joiner=", and ",
    aliases=("eng", "english"),
)

SPANISH = PhraseBank(
    code="es",
    name="Español",
    vocab=VOCAB_ES,
    territories=TERRITORIES_ES,
    findings_header="Hallazgos",
    impression_header="Impresión",
    rate_label="Frecuencia",
    ventricular_rate_label="Frecuencia ventricular",
    rate_template="{label} aproximadamente {bpm} lpm",
    axis_word="eje",
    low_confidence="(requiere confirmación clínica)",
    lead_clause_territory=" en las derivaciones {territory} ({leads})",
    lead_clause_plain=" en las derivaciones {leads}",
    merged_territory_clause="{core} en las derivaciones {parts}",
    no_findings="No se proporcionaron hallazgos estructurados.",
    consistent_with="Hallazgos compatibles con {term}.",
    normal_only="ECG normal. Sin alteraciones agudas.",
    normal_with_extra="ECG normal, con {extra}. Sin alteraciones agudas.",
    non_specific="Hallazgos inespecíficos, según lo anterior.",
    no_acute_ischemia="No se identifican cambios isquémicos agudos.",
    two_joiner=" y ",
    # Spanish does not take a serial comma before "y".
    serial_joiner=" y ",
    aliases=("spa", "spanish", "español", "espanol"),
)

LANGUAGES: dict[str, PhraseBank] = {"en": ENGLISH, "es": SPANISH}


def get_language(code: str | None) -> PhraseBank:
    """Resolve a language code (or alias) to its phrase bank.

    Unknown codes raise rather than silently falling back to English: a caller who asks for
    Portuguese and receives English without being told has been handed a bug that will
    surface in front of a patient.
    """
    if code is None:
        return LANGUAGES[DEFAULT_LANGUAGE]
    key = str(code).strip().lower()
    if key in LANGUAGES:
        return LANGUAGES[key]
    for bank in LANGUAGES.values():
        if key in bank.aliases:
            return bank
    raise ValueError(f"unsupported language {code!r}; available: {sorted(LANGUAGES)}")


def supported_languages() -> list[str]:
    return sorted(LANGUAGES)
