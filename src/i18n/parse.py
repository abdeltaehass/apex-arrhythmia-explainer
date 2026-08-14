"""Phase 27 — reading a report back, in whatever language it was written.

This module exists because of a specific, measured hole. The Phase-7 consistency gate is
what stops APEX asserting findings the detector never surfaced, and it works by matching
**English** impression phrases (`generation.parse.asserted_findings`). Point it at a Spanish
report and it finds nothing — so the report parses as asserting no findings and passes the
gate unconditionally.

That is not a cosmetic gap. Measured before this module existed: a Spanish report that
fabricated *bloqueo completo de rama izquierda* on a patient whose only detected finding was
atrial fibrillation was reported **consistent**, while the identical fabrication in English
was caught. Shipping Spanish without this file would have given Spanish-speaking patients a
system with its hallucination guardrail silently switched off — a health-equity failure that
is concrete rather than rhetorical, and invisible to anyone reading only English output.

Two properties the matcher needs that the English one did not:

**Accent tolerance on input.** The vocabulary spells *fibrilación* correctly, but a
clinician typing quickly, an LLM with an imperfect tokenizer, or a system that mangles
encoding will produce *fibrilacion*. Matching is therefore done on accent-stripped text,
while the vocabulary keeps its accents — tolerance belongs in the reader, not in what is
written. A gate that a missing accent can disable is not a gate.

**Longest match wins.** *extrasístoles* (PRC(S)) is a substring of *extrasístoles
auriculares* (PAC), exactly as "premature complexes" sits inside "atrial premature
complexes" in English. Naive substring matching charges a correct PAC report with also
asserting PRC(S), and the gate reports a hallucination that did not happen. A gate that
cries wolf gets switched off, so nested matches are dropped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.i18n.languages import LANGUAGES, PhraseBank, get_language


def strip_accents(text: str) -> str:
    """Fold accents for matching: ``fibrilación`` -> ``fibrilacion``."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def _norm(text: str) -> str:
    return strip_accents(text.replace("**", "").replace("__", "")).lower()


@dataclass
class ParsedReport:
    findings: str
    impression: str
    well_formed: bool
    language: str


def _section_re(bank: PhraseBank) -> re.Pattern:
    f = re.escape(_norm(bank.findings_header))
    i = re.escape(_norm(bank.impression_header))
    return re.compile(rf"{f}\s*:\s*(?P<findings>.*?)(?:\n\s*{i}\s*:\s*(?P<impression>.*))?$",
                      re.IGNORECASE | re.DOTALL)


def detect_language(text: str) -> str | None:
    """Which language's section headers this text uses, or ``None`` if neither.

    Header-based rather than statistical: the report format is fixed, so the headers are a
    reliable signal, and a wrong guess here would route text to the wrong term table and
    silently empty the gate.
    """
    normalized = _norm(text)
    for code, bank in LANGUAGES.items():
        if re.search(rf"{re.escape(_norm(bank.findings_header))}\s*:", normalized):
            return code
    return None


def parse_report(text: str, lang: str | PhraseBank | None = None) -> ParsedReport:
    """Split a report into its two sections, in the given (or detected) language."""
    resolved = lang if lang is not None else (detect_language(text) or "en")
    bank = resolved if isinstance(resolved, PhraseBank) else get_language(resolved)
    cleaned = text.strip()
    m = _section_re(bank).search(_norm(cleaned))
    if not m or not m.group("impression"):
        return ParsedReport(findings=cleaned, impression="", well_formed=False,
                            language=bank.code)

    # Matching happened on the normalized copy; slice the *original* so the returned text
    # keeps its accents and capitalization.
    body = cleaned.replace("**", "").replace("__", "")
    return ParsedReport(findings=body[m.start("findings"):m.end("findings")].strip(),
                        impression=body[m.start("impression"):m.end("impression")].strip(),
                        well_formed=True, language=bank.code)


def asserted_findings(text: str, lang: str | PhraseBank | None = None) -> set[str]:
    """SCP codes whose impression term appears in ``text``, longest match wins.

    The language-aware counterpart of `generation.parse.asserted_findings`, and the input to
    the Phase-7 consistency check for non-English reports.

    **Only the Impression section is searched**, matching the English implementation. A
    diagnosis is asserted when the report *names the condition*, not when it restates the
    morphology — and several morphological sentences are word-for-word identical to another
    code's impression term. ``ISCAN``'s finding is "T-wave inversion", which is exactly
    ``INVT``'s impression; ``INJAS``'s is "ST-segment depression", exactly ``STD_``'s. An
    earlier version of this function searched the whole report and duly charged every
    ischemia report with also asserting INVT and STD_, which would have made the Spanish
    gate noisier than the English one — the precise asymmetry this phase exists to remove.
    """
    resolved = lang if lang is not None else (detect_language(text) or "en")
    bank = resolved if isinstance(resolved, PhraseBank) else get_language(resolved)
    parsed = parse_report(text, bank)
    haystack = _norm(parsed.impression if parsed.well_formed else text)

    hits: list[tuple[int, int, str]] = []
    for code, entry in bank.vocab.items():
        if not entry.impression:
            continue
        needle = _norm(entry.impression)
        for m in re.finditer(re.escape(needle), haystack):
            hits.append((m.start(), m.end(), code))

    keep: set[str] = set()
    for start, end, code in hits:
        if any(s <= start and end <= e and (e - s) > (end - start) for s, e, _ in hits):
            continue
        keep.add(code)
    return keep


def check_consistency(text: str, surfaced: set[str], lang: str | PhraseBank | None = None):
    """Phase-7 consistency check, language-aware.

    Returns the same :class:`~src.eval.consistency.ConsistencyResult` the English path
    produces, so callers and tests treat both languages identically.
    """
    from src.eval.consistency import check

    return check(asserted_findings(text, lang), set(surfaced))
