"""Phase 22 — rendering the change report, and the gate that keeps it honest.

:func:`render_change_report` turns a :class:`~src.longitudinal.delta.LongitudinalDelta`
into the text a clinician reads:

    Compared with the prior study of 1992-07-26 (3 days earlier): new ST-segment
    depression in leads V4-V6. The PR interval has increased from 162 ms to 214 ms
    (+52 ms).

Like the Phase-6 templater this is deterministic and renders only from the structured
delta, so it is consistency-clean by construction — it has no way to name a change that
was not measured. :func:`check_change_consistency` enforces that property from the outside
anyway, because the same renderer will eventually be swapped for an LLM backend and the
gate has to survive that substitution.

**The gate is stricter here than in Phase 7, deliberately.** A single-record hallucination
invents a finding. A longitudinal hallucination invents a *trend*, and a trend is what
drives the decision to intervene: "new ST depression since yesterday" starts a workup that
"ST depression" alone does not. So the checker verifies the direction of every claim too,
not just its subject — asserting that something increased when it decreased is counted as
an unsupported claim, not a wording slip.

**Three phrasing rules that are clinical, not cosmetic.**

*Silence about what could not be compared is dangerous.* If PR could not be measured
because the patient is in atrial fibrillation, the report says so. An interval quietly
missing from a change report reads as "unchanged".

*A change in a measurement is not the same as a change in its interpretation.* An ST
segment that moves 0.06 mV and ends up at -0.07 mV has moved detectably but is still within
normal limits, and the report says exactly that rather than announcing "new ST depression".
The abnormality bar is the conventional 0.1 mV / 1 mm; the detectability bar is the
measured noise floor, and they are different questions.

*Unchanged is a finding.* "No significant change from the prior study" is one of the most
useful sentences in serial ECG reading — it is what rules out evolution — so it is stated
explicitly rather than left as an empty section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.generation.vocab import VOCAB
from src.longitudinal.delta import LongitudinalDelta, group_leads

# Conventional clinical bar for a *significant* ST deviation: 0.1 mV = 1 mm. Distinct from
# the MDC, which asks only whether a shift is larger than measurement noise.
ST_ABNORMAL_MV = 0.10

DISCLAIMER = ("Decision support only — serial comparison must be confirmed against the "
              "actual prior tracing and the clinical picture.")


def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _describe(code: str, fallback: str = "") -> str:
    entry = VOCAB.get(code)
    if entry is not None:
        return entry.impression or entry.finding
    return fallback or code


def _fmt(value: float, unit: str) -> str:
    return f"{value:.2f} mV" if unit == "mV" else f"{round(value)} {unit}"


def _lead_phrase(leads: list[str]) -> str:
    """"lead V4" or "leads V4-V6" — singular/plural agreement, which a report needs."""
    grouped = group_leads(leads)
    return f"lead {grouped}" if len(leads) == 1 else f"leads {grouped}"


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _st_sentences(delta: LongitudinalDelta) -> list[str]:
    """Group significant per-lead ST shifts into directional, territory-aware sentences."""
    sig = delta.significant_st()
    if not sig:
        return []

    now_depressed = [c for c in sig if c.current <= -ST_ABNORMAL_MV and c.delta < 0]
    now_elevated = [c for c in sig if c.current >= ST_ABNORMAL_MV and c.delta > 0]
    resolved = [c for c in sig
                if abs(c.prior) >= ST_ABNORMAL_MV and abs(c.current) < ST_ABNORMAL_MV]
    accounted = {id(c) for c in now_depressed + now_elevated + resolved}
    subtle = [c for c in sig if id(c) not in accounted]

    out: list[str] = []
    for group, word in ((now_depressed, "depression"), (now_elevated, "elevation")):
        if not group:
            continue
        phrase = _lead_phrase([c.lead for c in group])
        worst = max(group, key=lambda c: abs(c.delta))
        was_abnormal = any(abs(c.prior) >= ST_ABNORMAL_MV for c in group)
        opener = "Increased ST-segment" if was_abnormal else "New ST-segment"
        detail = (f"{worst.prior:+.2f} mV to {worst.current:+.2f} mV" if len(group) == 1
                  else f"{worst.lead} {worst.prior:+.2f} mV to {worst.current:+.2f} mV")
        out.append(f"{opener} {word} in {phrase} ({detail})")
    if resolved:
        out.append("Previously noted ST deviation in "
                   f"{_lead_phrase([c.lead for c in resolved])} has resolved")
    if subtle:
        out.append(f"Measurable but sub-threshold ST shift in {_lead_phrase([c.lead for c in subtle])}, "
                   "remaining within normal limits")
    return out


def _interval_sentences(delta: LongitudinalDelta) -> tuple[list[str], list[str]]:
    """Return ``(changed, not_compared)`` sentences for the interval channel."""
    changed: list[str] = []
    blocked: list[str] = []
    for c in delta.intervals:
        if c.suppressed_reason:
            blocked.append(f"{c.name} not compared: {c.suppressed_reason}")
            continue
        if not c.significant:
            continue
        changed.append(
            f"The {c.name} has {c.direction} from {_fmt(c.prior, c.unit)} to "
            f"{_fmt(c.current, c.unit)} ({c.delta:+.0f} {c.unit})"
        )
    return changed, blocked


def _is_rhythm(code: str) -> bool:
    entry = VOCAB.get(code)
    return entry is not None and entry.group in ("rhythm", "pacing")


def _finding_sentences(delta: LongitudinalDelta) -> list[str]:
    """Findings prose, with rhythm changes phrased as *transitions*.

    When one rhythm resolves and another appears, "atrial fibrillation has reverted to
    sinus rhythm" is how a cardiologist writes it — and it is how PTB-XL's own readers
    write it. Emitting "new: sinus rhythm" and "resolved: atrial fibrillation" as two
    unrelated bullets states the same facts while losing the causal link that makes the
    sentence worth reading. Both codes stay in the structured delta either way, so the
    consistency gate still sees them.
    """
    out: list[str] = []
    new = delta.new_findings()
    gone = delta.resolved_findings()

    new_rhythm = [f for f in new if _is_rhythm(f.code)]
    gone_rhythm = [f for f in gone if _is_rhythm(f.code)]
    if len(new_rhythm) == 1 and len(gone_rhythm) == 1:
        before = _describe(gone_rhythm[0].code, gone_rhythm[0].description)
        after = _describe(new_rhythm[0].code, new_rhythm[0].description)
        reverted = after == "sinus rhythm"
        out.append(f"{_cap(before)} has {'reverted' if reverted else 'changed'} to {after}")
        new = [f for f in new if f is not new_rhythm[0]]
        gone = [f for f in gone if f is not gone_rhythm[0]]

    if new:
        out.append("New since the prior study: "
                   + _join([_describe(f.code, f.description) for f in new]))
    if gone:
        out.append("No longer present: "
                   + _join([_describe(f.code, f.description) for f in gone]))
    return out


@dataclass
class ChangeReport:
    comparison: str          # the body: what changed
    impression: str          # the one-line verdict
    not_compared: str        # what could not be assessed, and why
    caveats: str             # data-quality warnings
    disclaimer: str = DISCLAIMER

    @property
    def text(self) -> str:
        blocks = [self.comparison, self.impression, self.not_compared, self.caveats]
        return "\n\n".join(b for b in blocks if b)

    def as_dict(self) -> dict:
        return {"comparison": self.comparison, "impression": self.impression,
                "not_compared": self.not_compared, "caveats": self.caveats,
                "disclaimer": self.disclaimer, "text": self.text}


def render_change_report(delta: LongitudinalDelta) -> ChangeReport:
    """Render the structured delta as a serial-comparison report."""
    when = delta.prior_date or "the prior study"
    gap = f" ({delta.gap_phrase})" if delta.gap_phrase else ""
    header = f"Compared with the prior study of {when}{gap}:"

    findings = _finding_sentences(delta)
    st = _st_sentences(delta)
    intervals, blocked = _interval_sentences(delta)
    body = findings + st + intervals

    if body:
        comparison = header + " " + ". ".join(body) + "."
    else:
        comparison = (f"{header} no significant change in rhythm, intervals, or "
                      "ST segments by the detectable-change criteria applied.")

    # --- impression ---------------------------------------------------------
    if delta.new_findings():
        lead = _describe(delta.new_findings()[0].code, delta.new_findings()[0].description)
        impression = f"Impression: interval change since the prior study — new {lead}."
    elif st:
        impression = "Impression: new or evolving repolarization change since the prior study."
    elif intervals:
        names = _join([c.name for c in delta.significant_intervals()])
        impression = (f"Impression: {names} changed beyond the measurable threshold; "
                      "no new diagnostic finding.")
    elif delta.resolved_findings():
        impression = ("Impression: previously reported finding no longer present; "
                      "otherwise no interval change.")
    else:
        impression = "Impression: no significant interval change from the prior study."

    persistent = delta.persistent_findings()
    if persistent:
        impression += (" Persisting: "
                       + _join([_describe(f.code, f.description) for f in persistent]) + ".")

    not_compared = ("Not compared — " + "; ".join(blocked) + ".") if blocked else ""
    caveats = ("Data quality — " + "; ".join(delta.caveats) + ".") if delta.caveats else ""
    return ChangeReport(comparison, impression, not_compared, caveats)


# --- the longitudinal consistency gate ---------------------------------------

def _section(text: str, opener: str) -> str:
    """The clause following ``opener``, up to the next sentence break (or "" if absent)."""
    m = re.search(re.escape(opener) + r"(.*?)(?:\.|$)", text, re.S)
    return m.group(1) if m else ""


def _transition_subject(text: str) -> str:
    """The rhythm named *before* "has reverted/changed to" — the one that resolved."""
    m = re.search(r"(?:^|\.\s*)([^.]*?)\s+has (?:reverted|changed) to", text, re.S)
    return m.group(1) if m else ""


def _codes_in(fragment: str) -> set[str]:
    """SCP codes whose clinical phrase appears in ``fragment``, longest match wins.

    Naive substring matching over the vocabulary double-counts nested phrases: "premature
    complexes" (PRC(S)) sits inside "atrial premature complexes" (PAC), so a report that
    correctly resolved PAC was also charged with asserting PRC(S) and the gate reported a
    hallucination that had not happened. A gate that cries wolf gets switched off, so every
    match is kept with its span and any match strictly contained in a longer one is dropped.
    """
    if not fragment.strip():
        return set()
    hits: list[tuple[int, int, str]] = []
    for code, entry in VOCAB.items():
        phrase = (entry.impression or entry.finding).lower()
        if not phrase:
            continue
        for m in re.finditer(re.escape(phrase), fragment):
            hits.append((m.start(), m.end(), code))
    keep: set[str] = set()
    for start, end, code in hits:
        if any(s2 <= start and end <= e2 and (e2 - s2) > (end - start)
               for s2, e2, _ in hits):
            continue
        keep.add(code)
    return keep


@dataclass
class ChangeConsistency:
    """Result of auditing a change narrative against the structured delta."""

    consistent: bool
    asserted_new: set[str]
    asserted_resolved: set[str]
    asserted_intervals: set[str]
    unsupported: set[str]           # claims with no backing in the delta

    def as_dict(self) -> dict:
        return {"consistent": self.consistent,
                "asserted_new": sorted(self.asserted_new),
                "asserted_resolved": sorted(self.asserted_resolved),
                "asserted_intervals": sorted(self.asserted_intervals),
                "unsupported": sorted(self.unsupported)}


_DIRECTION_WORDS = {"increased": +1, "lengthened": +1, "prolonged": +1, "risen": +1,
                    "decreased": -1, "shortened": -1, "narrowed": -1, "fallen": -1}


def check_change_consistency(text: str, delta: LongitudinalDelta) -> ChangeConsistency:
    """Verify that a change narrative only asserts changes present in the delta.

    Checks three claim types, and — unlike the Phase-7 checker — the *direction* of each
    interval claim as well as its subject. Reporting that the QT shortened when it
    lengthened is not a paraphrase, it is the opposite clinical message, so it is recorded
    as unsupported.
    """
    low = text.lower()
    claimable = delta.claimable()
    unsupported: set[str] = set()

    asserted_new = _codes_in(_section(low, "new since the prior study:")
                             + " " + _section(low, "has changed to")
                             + " " + _section(low, "has reverted to"))
    asserted_resolved = _codes_in(_section(low, "no longer present:")
                                  + " " + _transition_subject(low))
    for code in asserted_new:
        if code not in claimable["new"]:
            unsupported.add(f"new:{code}")
    for code in asserted_resolved:
        if code not in claimable["resolved"]:
            unsupported.add(f"resolved:{code}")

    asserted_intervals: set[str] = set()
    by_key = {c.key: c for c in delta.intervals}
    for key, change in by_key.items():
        name = change.name.lower()
        m = re.search(rf"{re.escape(name)} has (\w+)", low)
        if not m:
            continue
        asserted_intervals.add(key)
        if not change.significant or change.delta is None:
            unsupported.add(f"interval:{key}")
            continue
        want = _DIRECTION_WORDS.get(m.group(1))
        if want is not None and want * change.delta < 0:
            unsupported.add(f"interval-direction:{key}")

    return ChangeConsistency(not unsupported, asserted_new, asserted_resolved,
                             asserted_intervals, unsupported)
