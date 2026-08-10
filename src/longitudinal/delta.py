"""Phase 22 — the comparison engine: what actually changed between two ECGs.

Given a prior and a current recording this produces a :class:`LongitudinalDelta`: a
structured, fully auditable account of the difference. Everything the narrative in
:mod:`src.longitudinal.report` is allowed to say comes from here, and nothing else does —
the same discipline Phase 7 imposes on single-record explanations, carried over to change.

Two independent channels feed it.

**The measurement channel** — PR, QRS, QT/QTc, heart rate, and per-lead ST level, from
:mod:`src.longitudinal.intervals`. Continuous, physical, and reproducible.

**The diagnostic channel** — the detector's calibrated per-label probabilities on each
recording, differenced to give new / resolved / persistent findings.

**Why a noise floor is not optional.** A difference of two noisy measurements is noisier
than either. Reporting "PR increased from 160 ms to 174 ms" is worse than saying nothing
when the measurement repeats to ±29 ms, because it manufactures a trend out of jitter and
a clinician cannot tell the difference. Every change therefore has to clear a **minimum
detectable change** (MDC) before it is allowed to be stated.

**Where the MDC comes from, and the mistake I made getting there.** The obvious source is
PTB-XL's 327 same-day pairs: recorded hours apart, too soon for real remodelling, so the
spread of their differences should be pure noise. It isn't. Their measured spread is
*larger* than that of pairs a year apart (QRS SD 40.8 ms vs 27.1 ms) — impossible for
noise, which cannot shrink as time passes. The cause is selection: you only get two ECGs
within a day if something is acutely wrong. Acute or arrhythmic codes appear in 63% of
same-day pairs against 34% of >1-year pairs (both-normal: 16.5% vs 46.3%). The same-day
cohort is not a null, it is the sickest cohort in the dataset.

So the floor is estimated two other ways, and reported as a bracket:

1. **Within-record split-half** (:func:`~src.longitudinal.intervals.split_half`) — one
   recording measured twice from alternating beats. No disease change is even possible, so
   this is pure instrument noise, and it is a *lower* bound because it excludes everything
   that varies between sessions (electrode placement, posture, autonomic state).
2. **Between-session, label-stable pairs** — two sessions whose annotated code set is
   identical. This includes session-to-session variation and is what the thresholds are
   actually fitted on. Crucially, its RC is flat across every gap bucket once labels are
   held stable (QRS 7.3 / 9.6 / 7.3 / 7.0 ms from same-day out to >1 year), which is the
   evidence that it measures noise rather than elapsed-time-dependent drift.

The fitted floors, on folds 1-8: heart rate 20 bpm, PR 30 ms, QRS 10 ms, QT 40 ms,
QTc(Fridericia) 30 ms, and 0.07 mV per lead for ST.

Spread is summarised robustly (``1.96 x 1.4826 x MAD``) rather than as ``1.96 x SD``. A
few percent of records suffer gross delineation failure, and those outliers inflate the SD
by an order of magnitude — split-half QRS has SD 31.3 ms against a robust SD of 2.5 ms.
Using the SD would set the bar so high that nothing short of bundle branch block could ever
be reported.

Thresholds are fitted on **folds 1-8 only** and loaded from
``outputs/longitudinal_thresholds.json``; the built-in defaults below are what that fit
produced, so the module works before the script is ever run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import ROOT
from src.longitudinal.intervals import IntervalSet

THRESHOLD_PATH = ROOT / "outputs" / "longitudinal_thresholds.json"

# Fitted on fold 1-8 label-stable pairs (robust repeatability coefficient), rounded up to a
# reportable granularity. Overridden by THRESHOLD_PATH when scripts/longitudinal_eval.py
# has been run.
DEFAULT_MDC: dict[str, float] = {
    "heart_rate": 20.0,        # bpm
    "pr": 30.0,                # ms
    "qrs": 10.0,               # ms
    "qt": 40.0,                # ms
    "qtc_bazett": 35.0,        # ms
    "qtc_fridericia": 30.0,    # ms
    "st_level": 0.07,          # mV, per lead (Bonferroni-corrected across 8 leads)
}

# The ST channel tests eight leads at once. At a per-lead 95% bar, the chance of at least
# one false positive somewhere in an unchanged tracing is 1 - 0.95**8 = 34% — so a third of
# all comparisons would invent a regional ST change. The per-lead threshold is therefore
# widened to hold the *family-wise* error at 5%: z rises from 1.96 to 2.73 (the 1-0.05/8
# quantile), scaling the fitted robust spread by 1.39. This is why st_level's default is
# 0.07 mV rather than the ~0.05 mV its raw repeatability implies.
ST_LEADS_TESTED = 8
Z_95 = 1.959964
Z_FAMILYWISE = 2.734368      # normal quantile at 1 - 0.05 / (2 * 8)

# Probability change a label must move by before "new" or "resolved" is asserted; see
# scripts/longitudinal_eval.py, which fits it the same way on label-stable pairs.
DEFAULT_PROB_MDC = 0.25

INTERVAL_META: dict[str, tuple[str, str]] = {
    "heart_rate": ("heart rate", "bpm"),
    "pr": ("PR interval", "ms"),
    "qrs": ("QRS duration", "ms"),
    "qt": ("QT interval", "ms"),
    "qtc_bazett": ("QTc (Bazett)", "ms"),
    "qtc_fridericia": ("QTc (Fridericia)", "ms"),
}

# Rhythms in which PR is undefined rather than merely unmeasured, so no number may be
# emitted. Two distinct reasons, both fatal to a PR interval:
#   - no P wave exists at all (atrial fibrillation, atrial flutter);
#   - P waves exist but do not conduct, so the P-to-QRS distance is an accident of where
#     the two independent rhythms happen to fall (second- and third-degree AV block).
# The second case is the more dangerous one, because a P wave *is* detected and a
# plausible-looking number does come out: gold pair 7658 -> 7688 developed complete heart
# block, and the module cheerfully reported "PR decreased from 278 ms to 180 ms" for a
# patient whose atria and ventricles had stopped talking to each other altogether.
NO_P_RHYTHMS = {"AFIB": "atrial fibrillation", "AFLT": "atrial flutter"}
DISSOCIATED_RHYTHMS = {"3AVB": "third-degree AV block", "2AVB": "second-degree AV block"}

PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")


def load_mdc(path: Path = THRESHOLD_PATH) -> dict[str, float]:
    """Fitted MDC thresholds, falling back to :data:`DEFAULT_MDC`."""
    mdc = dict(DEFAULT_MDC)
    if path.exists():
        blob = json.loads(path.read_text())
        mdc.update({k: float(v) for k, v in blob.get("mdc", {}).items()})
    return mdc


def load_prob_mdc(path: Path = THRESHOLD_PATH) -> float:
    if path.exists():
        blob = json.loads(path.read_text())
        return float(blob.get("prob_mdc", DEFAULT_PROB_MDC))
    return DEFAULT_PROB_MDC


@dataclass
class IntervalChange:
    """One interval, then and now, with the verdict on whether the move is real."""

    key: str
    name: str
    unit: str
    prior: float | None
    current: float | None
    delta: float | None
    threshold: float
    significant: bool
    direction: str                  # "increased" | "decreased" | "unchanged" | "unavailable"
    suppressed_reason: str | None = None   # set when the comparison is not meaningful


@dataclass
class STChange:
    lead: str
    prior: float
    current: float
    delta: float
    threshold: float
    significant: bool
    direction: str                  # "elevated" | "depressed" | "unchanged"


@dataclass
class FindingChange:
    code: str
    description: str
    status: str                     # "new" | "resolved" | "persistent"
    prior_prob: float
    current_prob: float
    threshold: float

    @property
    def delta_prob(self) -> float:
        return self.current_prob - self.prior_prob


@dataclass
class LongitudinalDelta:
    """Everything that changed, and everything that was checked and did not."""

    prior_id: int | None = None
    current_id: int | None = None
    interval_days: int | None = None
    gap_phrase: str = ""
    prior_date: str = ""
    current_date: str = ""

    intervals: list[IntervalChange] = field(default_factory=list)
    st_changes: list[STChange] = field(default_factory=list)
    findings: list[FindingChange] = field(default_factory=list)

    prior_quality: str = "ok"
    current_quality: str = "ok"
    caveats: list[str] = field(default_factory=list)

    # --- convenience views the renderer and the consistency checker use --------
    def significant_intervals(self) -> list[IntervalChange]:
        return [c for c in self.intervals if c.significant]

    def significant_st(self) -> list[STChange]:
        return [c for c in self.st_changes if c.significant]

    def new_findings(self) -> list[FindingChange]:
        return [f for f in self.findings if f.status == "new"]

    def resolved_findings(self) -> list[FindingChange]:
        return [f for f in self.findings if f.status == "resolved"]

    def persistent_findings(self) -> list[FindingChange]:
        return [f for f in self.findings if f.status == "persistent"]

    @property
    def any_change(self) -> bool:
        return bool(self.significant_intervals() or self.significant_st()
                    or self.new_findings() or self.resolved_findings())

    def claimable(self) -> dict[str, set[str]]:
        """The complete set of assertions the narrative is permitted to make.

        Consumed by :func:`src.longitudinal.report.check_change_consistency`, which is the
        longitudinal analogue of the Phase-7 consistency gate: any change the text states
        that is not in here is a fabrication.
        """
        return {
            "new": {f.code for f in self.new_findings()},
            "resolved": {f.code for f in self.resolved_findings()},
            "persistent": {f.code for f in self.persistent_findings()},
            "intervals": {c.key for c in self.significant_intervals()},
            "leads": {c.lead for c in self.significant_st()},
        }

    def as_dict(self) -> dict:
        return {
            "prior_id": self.prior_id, "current_id": self.current_id,
            "interval_days": self.interval_days, "gap_phrase": self.gap_phrase,
            "prior_date": self.prior_date, "current_date": self.current_date,
            "intervals": [vars(c) for c in self.intervals],
            "st_changes": [vars(c) for c in self.st_changes],
            "findings": [{**vars(f), "delta_prob": round(f.delta_prob, 4)} for f in self.findings],
            "prior_quality": self.prior_quality, "current_quality": self.current_quality,
            "caveats": list(self.caveats),
            "any_change": self.any_change,
        }


def _direction(delta: float, significant: bool) -> str:
    if not significant:
        return "unchanged"
    return "increased" if delta > 0 else "decreased"


def compare_intervals(prior: IntervalSet, current: IntervalSet,
                      mdc: dict[str, float] | None = None,
                      rhythm_codes: set[str] | None = None) -> list[IntervalChange]:
    """Interval-by-interval comparison, each gated by its own MDC.

    Two suppression rules stop the output from asserting nonsense:

    - **PR in the absence of a P wave.** If either study has no detectable P wave the PR
      interval does not exist and no number is emitted. Atrial fibrillation is the usual
      reason and is named explicitly, because "PR interval unchanged" would be actively
      misleading in a patient who is in AF.
    - **Bazett when the rate moved.** Bazett's correction divides by the square root of RR,
      so a heart-rate change alone shifts QTc(Bazett) even when QT is untouched. Measured
      here: between sessions its repeatability is 36.0 ms against Fridericia's 29.3 ms,
      while within a single recording — where the rate cannot vary — the two are identical
      at 8.4 ms. The whole excess is rate-induced. Bazett is therefore suppressed whenever
      the rate itself changed significantly, and Fridericia carries the QTc verdict.
    """
    mdc = mdc or load_mdc()
    rhythm_codes = rhythm_codes or set()
    out: list[IntervalChange] = []

    hr_changed = False
    hr_a, hr_b = prior.heart_rate, current.heart_rate
    if hr_a is not None and hr_b is not None:
        hr_changed = abs(hr_b - hr_a) > mdc["heart_rate"]

    for key, (name, unit) in INTERVAL_META.items():
        thr = mdc.get(key, 0.0)
        a, b = getattr(prior, key), getattr(current, key)
        reason = None

        if key == "pr":
            dissociated = sorted(DISSOCIATED_RHYTHMS[c] for c in rhythm_codes
                                 if c in DISSOCIATED_RHYTHMS)
            missing = [lbl for lbl, s in (("prior", prior), ("current", current))
                       if not s.p_detected]
            if dissociated:
                reason = (f"{dissociated[0]} — P waves are dissociated from the QRS, so the "
                          "PR interval is undefined regardless of what can be measured")
            elif missing:
                named = sorted(NO_P_RHYTHMS[c] for c in rhythm_codes if c in NO_P_RHYTHMS)
                why = f" ({named[0]})" if named else ""
                reason = (f"no P wave detected in the {' and '.join(missing)} study{why}"
                          " — PR interval is undefined")
        if key in ("qt", "qtc_bazett") and hr_changed:
            # The QT interval shortens as the rate rises — that is physiology, not
            # pathology, and it is the entire reason rate correction exists. Reporting the
            # raw shift as a finding turns an expected consequence of tachycardia into an
            # apparent event: gold pair 8450 -> 8475 sped up from 72 to 128 bpm and the raw
            # QT duly "decreased by 69 ms" while the corrected value barely moved. Bazett is
            # suppressed for the same reason one step removed — its sqrt(RR) over-corrects,
            # so it manufactures change in the opposite direction. When the rate moves,
            # Fridericia carries the verdict alone.
            which = "the QT interval is rate-dependent" if key == "qt" else (
                "Bazett's correction is rate-dependent")
            reason = (f"heart rate changed by more than the detectable threshold and "
                      f"{which}, so QTc (Fridericia) is reported instead")

        if reason is not None or a is None or b is None:
            if reason is None:
                reason = "not measurable in one or both studies"
            out.append(IntervalChange(key, name, unit, a, b, None, thr, False,
                                      "unavailable", reason))
            continue

        d = b - a
        sig = abs(d) > thr
        out.append(IntervalChange(key, name, unit, a, b, round(d, 1), thr, sig,
                                  _direction(d, sig)))
    return out


def compare_st(prior: IntervalSet, current: IntervalSet,
               mdc: dict[str, float] | None = None) -> list[STChange]:
    """Per-lead ST-level comparison at J+60 ms, gated by the ST MDC."""
    mdc = mdc or load_mdc()
    thr = mdc.get("st_level", DEFAULT_MDC["st_level"])
    out: list[STChange] = []
    for lead, a in prior.st_level.items():
        if lead not in current.st_level:
            continue
        b = current.st_level[lead]
        d = b - a
        sig = abs(d) > thr
        out.append(STChange(lead, a, b, round(d, 4), thr, sig,
                            "elevated" if d > 0 else "depressed" if sig else "unchanged"))
    return out


def compare_findings(prior_probs: dict[str, float], current_probs: dict[str, float],
                     descriptions: dict[str, str] | None = None,
                     decision_threshold: float = 0.5,
                     prob_mdc: float | None = None) -> list[FindingChange]:
    """New / resolved / persistent findings from two probability vectors.

    A label is called **new** only if it is above the decision threshold now, was below it
    before, *and* the probability moved by more than ``prob_mdc``. That third condition is
    the one that matters: without it, any label sitting near 0.5 flickers between studies
    on noise alone and generates a spurious "new onset" — the double-thresholding problem,
    where differencing two independent decisions roughly doubles the error rate. Requiring
    the underlying probability to actually move suppresses exactly those borderline flips.
    """
    descriptions = descriptions or {}
    prob_mdc = load_prob_mdc() if prob_mdc is None else prob_mdc
    out: list[FindingChange] = []
    for code in sorted(set(prior_probs) | set(current_probs)):
        pa = float(prior_probs.get(code, 0.0))
        pb = float(current_probs.get(code, 0.0))
        was, now = pa >= decision_threshold, pb >= decision_threshold
        if now and not was:
            status = "new" if (pb - pa) > prob_mdc else "persistent"
        elif was and not now:
            status = "resolved" if (pa - pb) > prob_mdc else "persistent"
        elif was and now:
            status = "persistent"
        else:
            continue
        if status == "persistent" and not (was and now):
            continue      # a sub-threshold flicker: neither a change nor a standing finding
        out.append(FindingChange(code, descriptions.get(code, ""), status, pa, pb, prob_mdc))
    return out


def group_leads(leads: list[str]) -> str:
    """Format a lead list the way a report reads: contiguous precordials become a range.

    ``["V4","V5","V6"]`` -> ``"V4-V6"``; ``["I","aVL","V5","V6"]`` -> ``"I, aVL, V5-V6"``.
    """
    limb = [x for x in leads if x not in PRECORDIAL]
    prec = sorted((x for x in leads if x in PRECORDIAL), key=PRECORDIAL.index)
    parts: list[str] = list(limb)
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        parts.append(run[0] if len(run) == 1 else
                     f"{run[0]}-{run[1]}" if len(run) == 2 else f"{run[0]}-{run[-1]}")
        run.clear()

    for lead in prec:
        if run and PRECORDIAL.index(lead) == PRECORDIAL.index(run[-1]) + 1:
            run.append(lead)
        else:
            flush()
            run.append(lead)
    flush()
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + (" and " if len(parts) == 2 else ", and ") + parts[-1]


def build_delta(prior: IntervalSet, current: IntervalSet,
                prior_probs: dict[str, float] | None = None,
                current_probs: dict[str, float] | None = None,
                descriptions: dict[str, str] | None = None,
                pair=None, mdc: dict[str, float] | None = None,
                decision_threshold: float = 0.5) -> LongitudinalDelta:
    """Assemble the full structured comparison.

    ``pair`` is an optional :class:`~src.longitudinal.pairs.ECGPair` supplying identifiers
    and the elapsed-time phrasing. Probability dicts are optional — with none supplied the
    delta carries the measurement channel alone, which is exactly what the offline
    interval-only evaluation uses.
    """
    mdc = mdc or load_mdc()
    surfaced_now = {c for c, p in (current_probs or {}).items() if p >= decision_threshold}
    surfaced_before = {c for c, p in (prior_probs or {}).items() if p >= decision_threshold}

    delta = LongitudinalDelta(
        intervals=compare_intervals(prior, current, mdc,
                                    rhythm_codes=surfaced_now | surfaced_before),
        st_changes=compare_st(prior, current, mdc),
        findings=(compare_findings(prior_probs, current_probs, descriptions,
                                   decision_threshold)
                  if prior_probs is not None and current_probs is not None else []),
        prior_quality=prior.quality, current_quality=current.quality,
    )
    if pair is not None:
        delta.prior_id, delta.current_id = pair.prior_id, pair.current_id
        delta.interval_days = pair.interval_days
        delta.gap_phrase = pair.describe_gap()
        delta.prior_date = str(pair.prior_date.date())
        delta.current_date = str(pair.current_date.date())

    for label, s in (("prior", prior), ("current", current)):
        if s.quality != "ok":
            delta.caveats.append(f"{label} study flagged '{s.quality}': {'; '.join(s.notes) or 'see notes'}")
    if prior.n_beats and prior.n_beats < 5:
        delta.caveats.append(f"prior median beat built from only {prior.n_beats} beats")
    if current.n_beats and current.n_beats < 5:
        delta.caveats.append(f"current median beat built from only {current.n_beats} beats")
    return delta
