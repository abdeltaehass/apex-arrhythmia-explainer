"""Phase 27 — one renderer, any language.

Reproduces `src.generation.templater.render_report` exactly for English, and produces the
same report in Spanish from :data:`~src.i18n.languages.SPANISH`. Structure, ordering, and
merging logic are shared; only the phrase bank changes.

Sharing the code path is the whole design. The obvious alternative — a second renderer for
Spanish — starts identical and drifts: a fix to the English merge logic silently does not
reach Spanish, and the two languages quietly stop saying the same thing about the same ECG.
A test asserts English output here is byte-identical to the Phase-6 templater, so any
divergence is a build failure rather than a discovery months later.
"""

from __future__ import annotations

from src.generation.templater import NORM_COMPATIBLE, StructuredInput
from src.generation.vocab import GROUP_ORDER, ISCHEMIC_CODES
from src.i18n.languages import PhraseBank, get_language


def _group_rank(group: str) -> int:
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)


def _join(items: list[str], lang: PhraseBank) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}{lang.two_joiner}{items[1]}"
    return lang.list_separator.join(items[:-1]) + f"{lang.serial_joiner}{items[-1]}"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _lead_clause(leads: list[str], territory: str | None, lang: PhraseBank) -> str:
    if not leads:
        return ""
    joined = _join(leads, lang)
    if territory:
        return lang.lead_clause_territory.format(
            territory=lang.territories.get(territory, territory), leads=joined)
    return lang.lead_clause_plain.format(leads=joined)


def _flag(text: str, low: bool, lang: PhraseBank) -> str:
    return f"{text} {lang.low_confidence}" if low else text


def _rate_phrase(hr: float | None, rhythm_codes: set[str], lang: PhraseBank) -> str | None:
    if hr is None:
        return None
    label = lang.ventricular_rate_label if rhythm_codes & {"AFIB", "AFLT"} else lang.rate_label
    return lang.rate_template.format(label=label, bpm=round(hr))


def _merge_localized(items, lang: PhraseBank) -> list[str]:
    order: list[str] = []
    grouped: dict[str, list] = {}
    for f in items:
        core = lang.vocab[f.code].finding
        if core not in grouped:
            order.append(core)
        grouped.setdefault(core, []).append(f)

    out: list[str] = []
    for core in order:
        fs = grouped[core]
        terr = [(lang.vocab[f.code].territory, f.leads, f) for f in fs if f.leads]
        if terr:
            parts = [
                f"{lang.territories.get(t, t)} ({_join(ld, lang)})" for t, ld, _ in terr
            ]
            sentence = lang.merged_territory_clause.format(core=core,
                                                          parts=_join(parts, lang))
            out.append(_flag(sentence, any(f.low_confidence for _, _, f in terr), lang))
        else:
            for f in fs:
                out.append(_flag(core, f.low_confidence, lang))
    return out


def _findings_sentences(si: StructuredInput, lang: PhraseBank) -> list[str]:
    by_group: dict[str, list] = {}
    for f in si.findings:
        by_group.setdefault(f.group, []).append(f)
    rhythm_codes = {f.code for f in si.findings if f.group in ("rhythm", "pacing")}

    sentences: list[str] = []
    if "rhythm" in by_group or "pacing" in by_group:
        lead_rhythm = (by_group.get("rhythm") or by_group.get("pacing"))[0]
        e = lang.vocab[lead_rhythm.code]
        sentences.append(_flag(e.finding + _lead_clause(lead_rhythm.leads, e.territory, lang),
                               lead_rhythm.low_confidence, lang))
    rate = _rate_phrase(si.heart_rate_bpm, rhythm_codes, lang)
    if rate:
        sentences.append(rate)
    if si.heart_axis and str(si.heart_axis).lower() not in ("nan", "none", ""):
        axis = str(si.heart_axis)
        sentences.append(_cap(axis if lang.axis_word in axis.lower()
                              else f"{axis} {lang.axis_word}"))

    for group in GROUP_ORDER:
        if group in ("rhythm", "pacing", "normal"):
            continue
        items = by_group.get(group, [])
        if not items:
            continue
        if group in ("repolarization", "infarction"):
            sentences.extend(_merge_localized(items, lang))
        else:
            for f in items:
                e = lang.vocab[f.code]
                sentences.append(_flag(e.finding + _lead_clause(f.leads, e.territory, lang),
                                       f.low_confidence, lang))
    if "normal" in by_group and len(si.findings) == len(by_group["normal"]):
        sentences.append(lang.vocab["NORM"].finding)
    return sentences


def _impression_phrases(si: StructuredInput, lang: PhraseBank) -> list[str]:
    seen: set[str] = set()
    phrases: list[str] = []
    for f in sorted(si.findings, key=lambda f: (_group_rank(f.group), f.code)):
        imp = lang.vocab[f.code].impression
        if imp and imp not in seen:
            seen.add(imp)
            phrases.append(imp)
    return phrases


def render_report(si: StructuredInput, lang: str | PhraseBank = "en") -> dict[str, str]:
    """Render the structured input as ``{"findings": ..., "impression": ...}``."""
    bank = lang if isinstance(lang, PhraseBank) else get_language(lang)
    codes = set(si.codes())
    is_normal = "NORM" in codes and not (codes - NORM_COMPATIBLE)

    findings = ". ".join(_cap(s) for s in _findings_sentences(si, bank))
    findings = (findings + ".") if findings else bank.no_findings

    phrases = _impression_phrases(si, bank)
    if is_normal:
        extra = [bank.vocab[f.code].impression for f in si.findings
                 if f.code in NORM_COMPATIBLE - {"NORM", "SR"} and bank.vocab[f.code].impression]
        impression = (bank.normal_with_extra.format(extra=bank.extra_separator.join(extra))
                      if extra else bank.normal_only)
    elif phrases:
        impression = bank.consistent_with.format(term=phrases[0])
        if len(phrases) > 1:
            impression += " " + " ".join(f"{_cap(p)}." for p in phrases[1:])
        if not (codes & ISCHEMIC_CODES):
            impression += " " + bank.no_acute_ischemia
    else:
        impression = bank.non_specific
        if not (codes & ISCHEMIC_CODES):
            impression += " " + bank.no_acute_ischemia
    return {"findings": findings, "impression": impression}


def render_text(si: StructuredInput, lang: str | PhraseBank = "en") -> str:
    """The full two-section report, with localized headers."""
    bank = lang if isinstance(lang, PhraseBank) else get_language(lang)
    parts = render_report(si, bank)
    return (f"{bank.findings_header}:\n{parts['findings']}\n\n"
            f"{bank.impression_header}:\n{parts['impression']}")
