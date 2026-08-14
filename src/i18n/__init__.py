"""Phase 27 — multilingual clinical explanation.

Spanish output for the Phase-6 report generator, with the same safety guarantees as English.

    from src.i18n import render_text, asserted_findings

    render_text(structured_input, "es")
    asserted_findings(spanish_report, "es")     # feeds the Phase-7 consistency gate

Or end to end::

    analyze_signal(signal, 100, backend="template", lang="es")

The parts:

- :mod:`~src.i18n.vocab_es`   hand-authored Spanish clinical vocabulary, all 71 SCP codes
- :mod:`~src.i18n.languages`  the phrase bank — headers, units, joiners, agreement
- :mod:`~src.i18n.render`     one renderer for every language (English output is
                              byte-identical to the Phase-6 templater, asserted by test)
- :mod:`~src.i18n.parse`      language-aware reading, so the hallucination gate works in
                              Spanish too — it did not, and a Spanish report inventing a
                              diagnosis used to pass unconditionally
- :mod:`~src.i18n.glossary`   terminology validation against Spanish cardiology prose
"""

from src.i18n.glossary import check_terms, load_corpus
from src.i18n.languages import LANGUAGES, PhraseBank, get_language, supported_languages
from src.i18n.parse import (
    ParsedReport,
    asserted_findings,
    check_consistency,
    detect_language,
    parse_report,
    strip_accents,
)
from src.i18n.render import render_report, render_text
from src.i18n.vocab_es import TERRITORIES_ES, VOCAB_ES

__all__ = [
    "LANGUAGES", "TERRITORIES_ES", "VOCAB_ES", "ParsedReport", "PhraseBank",
    "asserted_findings", "check_consistency", "check_terms", "detect_language",
    "get_language", "load_corpus", "parse_report", "render_report", "render_text",
    "strip_accents", "supported_languages",
]
