"""Parse generated report text back into structured pieces.

The generator's contract (`prompts.SYSTEM_PROMPT`) is an exact two-section format:

    Findings:
    <factual sentences>

    Impression:
    <interpretive sentences>

This module is the other direction of `templater.render_report`: given raw generated
text, split it into sections and recover which SCP codes it asserts, so
`src/eval/consistency.py` can check that set against what the detector actually
surfaced without ever trusting the model's self-report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.generation.vocab import impression_terms

_SECTION_RE = re.compile(
    r"findings\s*:\s*(?P<findings>.*?)(?:\n\s*impression\s*:\s*(?P<impression>.*))?$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedReport:
    findings: str
    impression: str
    well_formed: bool  # True iff both section headers were found in order


def parse_report(text: str) -> ParsedReport:
    """Split generated text into ``(findings, impression)``.

    Tolerant of surrounding whitespace/markdown bolding (``**Findings:**``). If the
    Impression header is missing, ``impression`` is empty and ``well_formed`` is False
    — a malformed output the caller should treat as unusable rather than guess at.
    """
    cleaned = text.strip()
    m = _SECTION_RE.search(_strip_markdown(cleaned))
    if not m or not m.group("impression"):
        return ParsedReport(findings=cleaned, impression="", well_formed=False)
    return ParsedReport(
        findings=m.group("findings").strip(),
        impression=m.group("impression").strip(),
        well_formed=True,
    )


def _strip_markdown(text: str) -> str:
    return text.replace("**", "").replace("__", "")


def asserted_findings(text: str) -> set[str]:
    """SCP codes whose clinical *impression* phrase appears in generated text.

    Matches against the impression vocabulary (`vocab.impression_terms`), not the raw
    Findings sentences — a diagnosis is "asserted" when the model names the condition
    (e.g. "atrial fibrillation"), not merely when it restates a morphological
    observation.

    Several impression phrases are literal substrings of others (e.g. "lateral
    ischemia" inside "anterolateral ischemia", "premature complexes" inside "atrial
    premature complexes") — matched longest-first, each hit masks its span before
    shorter phrases are checked, so a mention of the specific finding does not also
    silently assert the more general one.
    """
    parsed = parse_report(text) if _looks_sectioned(text) else None
    haystack = (parsed.impression if parsed and parsed.well_formed else text).lower()
    terms = impression_terms()
    found = set()
    for phrase in sorted(terms, key=len, reverse=True):
        idx = haystack.find(phrase)
        if idx != -1:
            found.add(terms[phrase])
            haystack = haystack[:idx] + (" " * len(phrase)) + haystack[idx + len(phrase):]
    return found


def _looks_sectioned(text: str) -> bool:
    return "impression" in text.lower()
