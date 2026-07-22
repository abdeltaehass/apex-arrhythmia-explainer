"""Turn a set of SCP codes (+ context) into a structured input and a two-section report.

Two jobs:

1. :func:`build_structured_input` — assemble the *structured detection output* the
   generator consumes: ordered findings with confidence and localizing leads, plus
   rate/rhythm/demographic context. This is the same shape whether it comes from the
   detector at inference or from ground-truth labels when building training data.

2. :func:`render_report` — deterministically render that structured input into the
   target **Findings / Impression** text in cardiologist register. This template output
   is both the fine-tuning *target* and the "what a human would write" *reference* the
   generated text is compared against. Because it only ever states the findings it is
   given, the template is consistency-clean by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.vocab import GROUP_ORDER, ISCHEMIC_CODES, VOCAB, leads_for

# Codes that don't contradict a "Normal ECG" read when paired with NORM — physiological
# rate/rhythm variants, not pathology. Anything else co-occurring with NORM (AFIB, PVC,
# an MI code, ...) must still be named in the Impression; see `render_report`.
NORM_COMPATIBLE = {"NORM", "SR", "SBRAD", "STACH", "SARRH"}


@dataclass
class Finding:
    code: str
    description: str
    group: str
    confidence: float | None = None
    leads: list[str] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < 0.5


@dataclass
class StructuredInput:
    findings: list[Finding]
    record_id: int | None = None
    age: int | None = None
    sex: str | None = None
    heart_rate_bpm: float | None = None
    heart_axis: str | None = None
    review_threshold: float = 0.5
    original_report: str | None = None  # PTB-XL's own (terse) report, for provenance

    def codes(self) -> list[str]:
        return [f.code for f in self.findings]


def _group_rank(group: str) -> int:
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)


def build_structured_input(
    codes,
    confidences: dict[str, float] | None = None,
    leads_by_code: dict[str, list[str]] | None = None,
    descriptions: dict[str, str] | None = None,
    review_threshold: float = 0.5,
    **context,
) -> StructuredInput:
    """Assemble a :class:`StructuredInput` from present SCP codes.

    ``confidences`` / ``leads_by_code`` come from the detector + grounding layers at
    inference; when building training data they are derived from PTB-XL (likelihood and
    standard territory leads). Findings are ordered by clinical group then code. Unknown
    codes are skipped. ``context`` accepts ``record_id/age/sex/heart_rate_bpm/heart_axis/
    original_report``.
    """
    confidences = confidences or {}
    leads_by_code = leads_by_code or {}
    descriptions = descriptions or {}
    findings: list[Finding] = []
    for code in codes:
        e = VOCAB.get(code)
        if e is None:
            continue
        findings.append(Finding(
            code=code,
            description=descriptions.get(code, ""),
            group=e.group,
            confidence=confidences.get(code),
            leads=leads_by_code.get(code) or leads_for(code),
        ))
    findings.sort(key=lambda f: (_group_rank(f.group), f.code))
    valid = {"record_id", "age", "sex", "heart_rate_bpm", "heart_axis", "original_report"}
    return StructuredInput(
        findings=findings,
        review_threshold=review_threshold,
        **{k: v for k, v in context.items() if k in valid},
    )


def _lead_clause(leads: list[str], territory: str | None) -> str:
    if not leads:
        return ""
    joined = _join(leads)
    return f" in the {territory} leads ({joined})" if territory else f" in leads {joined}"


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _rate_phrase(hr: float | None, rhythm_codes: set[str]) -> str | None:
    if hr is None:
        return None
    label = "Ventricular rate" if rhythm_codes & {"AFIB", "AFLT"} else "Rate"
    return f"{label} approximately {round(hr)} bpm"


def _findings_sentences(si: StructuredInput) -> list[str]:
    by_group: dict[str, list[Finding]] = {}
    for f in si.findings:
        by_group.setdefault(f.group, []).append(f)
    rhythm_codes = {f.code for f in si.findings if f.group in ("rhythm", "pacing")}

    sentences: list[str] = []
    # Rhythm + rate lead the Findings section.
    if "rhythm" in by_group or "pacing" in by_group:
        lead_rhythm = (by_group.get("rhythm") or by_group.get("pacing"))[0]
        e = VOCAB[lead_rhythm.code]
        sentences.append(_flag(e.finding + _lead_clause(lead_rhythm.leads, e.territory),
                               lead_rhythm))
    rate = _rate_phrase(si.heart_rate_bpm, rhythm_codes)
    if rate:
        sentences.append(rate)
    if si.heart_axis and str(si.heart_axis).lower() not in ("nan", "none", ""):
        axis = str(si.heart_axis)
        sentences.append(_cap(axis if "axis" in axis.lower() else f"{axis} axis"))

    for group in GROUP_ORDER:
        if group in ("rhythm", "pacing", "normal"):
            continue
        items = by_group.get(group, [])
        if not items:
            continue
        if group in ("repolarization", "infarction"):
            sentences.extend(_merge_localized(items))
        else:
            for f in items:
                e = VOCAB[f.code]
                sentences.append(_flag(e.finding + _lead_clause(f.leads, e.territory), f))
    # Purely normal study.
    if "normal" in by_group and len(si.findings) == len(by_group["normal"]):
        sentences.append(VOCAB["NORM"].finding)
    return sentences


def _merge_localized(items: list[Finding]) -> list[str]:
    """Merge same-observation findings across territories into one sentence.

    e.g. two `T-wave inversion` codes in different territories become
    "T-wave inversion in the anterior (V2, V3, V4) and inferior (II, III, aVF) leads".
    """
    order: list[str] = []
    grouped: dict[str, list[Finding]] = {}
    for f in items:
        core = VOCAB[f.code].finding
        if core not in grouped:
            order.append(core)
        grouped.setdefault(core, []).append(f)
    out: list[str] = []
    for core in order:
        fs = grouped[core]
        terr = [(VOCAB[f.code].territory, f.leads, f) for f in fs if f.leads]
        if terr:
            parts = [f"{t} ({_join(ld)})" for t, ld, _ in terr]
            sentence = f"{core} in the {_join(parts)} leads"
            low = any(f.low_confidence for _, _, f in terr)
            out.append(_flag_raw(sentence, low))
        else:
            for f in fs:
                out.append(_flag(core, f))
    return out


def _flag(text: str, f: Finding) -> str:
    return _flag_raw(text, f.low_confidence)


def _flag_raw(text: str, low: bool) -> str:
    return f"{text} (requires clinician confirmation)" if low else text


def _impression_phrases(si: StructuredInput) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for f in sorted(si.findings, key=lambda f: (_group_rank(f.group), f.code)):
        imp = VOCAB[f.code].impression
        if imp and imp not in seen:
            seen.add(imp)
            phrases.append(imp)
    return phrases


def render_report(si: StructuredInput) -> dict[str, str]:
    """Render the structured input into ``{"findings": ..., "impression": ...}`` text."""
    codes = set(si.codes())
    # NORM collapses the Impression to "Normal ECG" only alongside codes that don't
    # contradict a normal read — physiological rate/rhythm variants (SR, sinus brady-
    # /tachycardia, sinus arrhythmia). NORM co-occurring with anything else (PTB-XL
    # does pair NORM with e.g. AFIB or PVC on occasion) must still name that finding —
    # collapsing to a bare "Normal ECG" would silently drop it from the Impression
    # even though Findings still describes it, a real gap this fixes.
    is_normal = "NORM" in codes and not (codes - NORM_COMPATIBLE)

    findings = ". ".join(_cap(s) for s in _findings_sentences(si))
    findings = (findings + ".") if findings else "No structured findings provided."

    phrases = _impression_phrases(si)
    if is_normal:
        extra = [VOCAB[f.code].impression for f in si.findings
                if f.code in NORM_COMPATIBLE - {"NORM", "SR"} and VOCAB[f.code].impression]
        impression = (f"Normal ECG, with {', '.join(extra)}. No acute abnormality."
                     if extra else "Normal ECG. No acute abnormality.")
    elif phrases:
        impression = f"Findings consistent with {phrases[0]}."
        if len(phrases) > 1:
            impression += " " + " ".join(f"{_cap(p)}." for p in phrases[1:])
        if not (codes & ISCHEMIC_CODES):
            impression += " No acute ischemic changes identified."
    else:
        impression = "Non-specific findings, as above."
        if not (codes & ISCHEMIC_CODES):
            impression += " No acute ischemic changes identified."
    return {"findings": findings, "impression": impression}
