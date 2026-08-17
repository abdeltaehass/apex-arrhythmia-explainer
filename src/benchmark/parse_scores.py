"""Phase 28 — recovering numeric scores from a language model's reply.

The prompt asks for five lines of ``LABEL: <0-10>``. Models mostly comply and sometimes do
not: they add prose after the number, wrap it in markdown, write ``7/10``, or interleave a
sentence of reasoning between the lines. A parser that only accepts the exact format would
score those replies as missing and quietly report the model as worse than it is, which is
the wrong direction for an honest comparison — so this one is deliberately permissive about
*form* while strict about *substance*.

Strict about substance means: a score is only recovered when it is unambiguously attached to
its label. Nothing is inferred from context, no default is substituted for an absent label,
and a reply that never mentions ``HYP`` yields ``None`` for HYP rather than a zero. A missing
answer and a confident "absent" are different claims, and averaging them together is how a
benchmark flatters whichever model hedges most.
"""

from __future__ import annotations

import re

from src.benchmark.features import SUPERCLASSES

# "MI: 7", "MI: 7/10", "**MI**: 7", "- MI — 7", "MI = 7.5"
_PATTERN = r"[*\-\s]*{label}[*\s]*[:=—-]\s*(\d{{1,2}}(?:\.\d+)?)\s*(?:/\s*10)?"


def parse_scores(text: str) -> dict[str, float | None]:
    """Recover a 0-1 score per superclass; ``None`` where the reply gave none."""
    out: dict[str, float | None] = {}
    for label in SUPERCLASSES:
        m = re.search(_PATTERN.format(label=re.escape(label)), text, re.IGNORECASE)
        if not m:
            out[label] = None
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            out[label] = None
            continue
        # Clip rather than reject: "12/10" is a compliance failure, not a missing answer.
        out[label] = max(0.0, min(value, 10.0)) / 10.0
    return out


def parse_interpretation(text: str) -> str:
    """The free-text sentence, if the model produced one."""
    m = re.search(r"INTERPRETATION\s*[:\-—]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    # Fall back to the longest prose line that is not a score line.
    candidates = [ln.strip() for ln in text.splitlines()
                  if ln.strip() and not re.match(r"[*\-\s]*[A-Z]{2,5}[*\s]*[:=]", ln.strip())]
    return max(candidates, key=len) if candidates else ""


def compliance(text: str) -> float:
    """Fraction of the five requested scores the reply actually supplied."""
    scores = parse_scores(text)
    return sum(v is not None for v in scores.values()) / len(SUPERCLASSES)
