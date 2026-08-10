"""Phase 23 — condensing a full APEX report into one pasteable sentence.

The Phase-6 report is two sections and several sentences. What goes into an EHR's
interpretation field, or into the body of a clinical note, is one line — and a clinician
pastes it under their own name, which sets the requirements:

**It must lead with what changes management.** Findings are ordered by clinical urgency,
not by model confidence and not alphabetically. ST elevation comes before a rhythm
statement; a rhythm statement comes before a non-specific T-wave change. If an urgent
pattern is present the sentence opens with it in a form that cannot be skimmed past.

**It must never be mistakable for a physician's own words.** Every impression carries the
attribution clause. This is not boilerplate defensiveness: the sentence is designed to be
pasted into a note, so the one thing it must not do is arrive in the record looking like
a human read the tracing. The clause is not optional and not configurable.

**It must degrade honestly when there is too much to say.** Ten findings do not fit in a
sentence. Rather than truncating mid-clause or dropping the tail silently, the overflow is
counted and stated ("plus 4 further findings"), so the reader knows to open the full report.

**Territories collapse.** A model that fires on four neighbouring ischemic territories
produces "anterolateral ischemia, anteroseptal ischemia, inferolateral ischemia, and
inferior ischemia", which spends the whole sentence saying one thing. Findings that share a
clinical entity and differ only in territory are merged into "multi-territory ischemia" —
using each vocabulary entry's own ``territory`` field, so the merge follows the clinical
model rather than string matching. The per-territory detail is still in the full report and
in the FHIR observations; it is the *summary* that should not be swamped by it.

Rate and rhythm lead the sentence when a rate is available, because that is how ECG
impressions are conventionally written and a reader's eye expects it there.
"""

from __future__ import annotations

from src.generation.vocab import VOCAB
from src.serving.schema import APEXReport
from src.serving.severity import URGENT_CODES

ATTRIBUTION = "computer-assisted interpretation, requires physician confirmation"
REVIEW_CLAUSE = "low-confidence findings flagged for review"

# Clinical priority for the sentence, most consequential first. Deliberately not
# GROUP_ORDER, which is the order a *full* report reads best in; a one-liner has to spend
# its first clause on whatever most changes what happens next.
PRIORITY = ("infarction", "repolarization", "conduction", "rhythm", "pacing",
            "ectopy", "chamber", "normal", "technical")

MAX_CHARS = 300
MAX_FINDINGS = 4


def _rank(code: str) -> tuple[int, int, str]:
    entry = VOCAB.get(code)
    group = entry.group if entry else "technical"
    urgent = 0 if code in URGENT_CODES else 1
    idx = PRIORITY.index(group) if group in PRIORITY else len(PRIORITY)
    return (urgent, idx, code)


def _phrase(code: str, fallback: str = "") -> str:
    entry = VOCAB.get(code)
    if entry is None:
        return fallback or code
    return entry.impression or entry.finding


def _entity(code: str) -> str:
    """The clinical entity behind a finding, with its territory stripped off.

    ``"anteroseptal myocardial infarction"`` and ``"subendocardial injury, anteroseptal"``
    both reduce to their entity ("myocardial infarction", "subendocardial injury"), which is
    what lets several territories of the same process merge into one clause.
    """
    entry = VOCAB.get(code)
    phrase = _phrase(code)
    if entry is None or not entry.territory:
        return phrase
    territory = entry.territory
    if phrase.startswith(f"{territory} "):
        return phrase[len(territory) + 1:]
    if phrase.endswith(f", {territory}"):
        return phrase[: -len(territory) - 2]
    return phrase


def _collapse(codes: list[str]) -> list[str]:
    """Findings as clauses, merging same-entity findings that differ only by territory."""
    order: list[str] = []
    members: dict[str, list[str]] = {}
    for code in codes:
        entity = _entity(code)
        if entity not in members:
            order.append(entity)
            members[entity] = []
        members[entity].append(code)

    out: list[str] = []
    for entity in order:
        group = members[entity]
        localized = [c for c in group if VOCAB.get(c) and VOCAB[c].territory]
        if len(localized) >= 2:
            out.append(f"multi-territory {entity}")
        else:
            seen: set[str] = set()
            for code in group:
                phrase = _phrase(code)
                if phrase not in seen:
                    seen.add(phrase)
                    out.append(phrase)
    return out


def _rhythm_clause(codes: list[str], heart_rate: float | None) -> tuple[str, set[str]]:
    """The leading "sinus rhythm at 72 bpm" clause, and the codes it consumed."""
    rhythm = [c for c in codes if (VOCAB.get(c) and VOCAB[c].group in ("rhythm", "pacing"))]
    used: set[str] = set()
    if rhythm:
        lead = min(rhythm, key=_rank)
        used.add(lead)
        text = _phrase(lead)
    elif heart_rate is not None:
        text = "Rhythm"
    else:
        return "", used
    if heart_rate is not None:
        text += f" at {round(heart_rate)} bpm"
    return text[0].upper() + text[1:], used


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def one_line_impression(report: APEXReport, heart_rate: float | None = None,
                        max_chars: int = MAX_CHARS,
                        max_findings: int = MAX_FINDINGS) -> str:
    """Condense an :class:`APEXReport` into a single pasteable sentence.

    ``heart_rate`` comes from Phase 22's interval measurement when available; without it
    the sentence simply omits the rate rather than guessing one.
    """
    codes = [f.label for f in report.findings]
    if not codes:
        return f"No diagnostic findings surfaced; {ATTRIBUTION}."

    ordered = sorted(codes, key=_rank)
    urgent = [c for c in ordered if c in URGENT_CODES]

    rhythm_text, used = _rhythm_clause(ordered, heart_rate)

    # The urgent finding leads the sentence, so it must not also appear in the body list.
    if urgent:
        used.add(urgent[0])

    remaining = [c for c in ordered if c not in used and c != "NORM" and _phrase(c)]
    rest = _collapse(remaining)

    overflow = 0
    if len(rest) > max_findings:
        overflow = len(rest) - max_findings
        rest = rest[:max_findings]

    # --- assemble ------------------------------------------------------------
    if set(codes) <= {"NORM", "SR", "SBRAD", "STACH", "SARRH"} and "NORM" in codes:
        body = rhythm_text or "Normal ECG"
        if rhythm_text and "sinus rhythm" in rhythm_text.lower():
            body = f"{rhythm_text} — normal ECG"
        elif rhythm_text:
            body = f"{rhythm_text}, otherwise normal ECG"
    elif rhythm_text and rest:
        body = f"{rhythm_text} with {_join(rest)}"
    elif rhythm_text:
        body = rhythm_text
    elif rest:
        body = _join(rest)[0].upper() + _join(rest)[1:]
    else:
        body = "Abnormal ECG"

    if overflow:
        body += f", plus {overflow} further finding{'s' if overflow > 1 else ''}"

    if urgent:
        body = f"URGENT — {_phrase(urgent[0])}: {body}"

    tail = f"{REVIEW_CLAUSE}; {ATTRIBUTION}" if report.review_recommended else ATTRIBUTION
    sentence = f"{body}; {tail}."

    # Last resort: if it still will not fit, shed findings rather than cut mid-word.
    while len(sentence) > max_chars and rest:
        rest.pop()
        overflow += 1
        shortened = (f"{rhythm_text} with {_join(rest)}" if rest and rhythm_text
                     else rhythm_text or "Abnormal ECG")
        shortened += f", plus {overflow} further finding{'s' if overflow > 1 else ''}"
        if urgent:
            shortened = f"URGENT — {_phrase(urgent[0])}: {shortened}"
        sentence = f"{shortened}; {tail}."
    return sentence


def impression_components(report: APEXReport, heart_rate: float | None = None) -> dict:
    """The same content, structured — for callers that want to lay it out themselves."""
    codes = [f.label for f in report.findings]
    ordered = sorted(codes, key=_rank)
    return {
        "urgent": [c for c in ordered if c in URGENT_CODES],
        "ordered_codes": ordered,
        "phrases": [_phrase(c) for c in ordered],
        "heart_rate": heart_rate,
        "review_recommended": report.review_recommended,
        "attribution": ATTRIBUTION,
        "sentence": one_line_impression(report, heart_rate),
    }
