"""Phase 28 — presenting an ECG to a model that cannot read a waveform.

A text LLM cannot consume a ``(12, 1000)`` array, and a multimodal one reading a rendered
strip is doing image interpretation rather than signal analysis. Both are legitimate
protocols and they measure different things, so the protocol has to be stated rather than
assumed.

This module implements the **measurement-mediated** protocol: the recording is reduced to
the numbers a cardiologist would read off it — rate, PR, QRS, QT/QTc, and per-lead ST
levels, all from Phase 22's delineator — and those are handed to the model as text.

Why this and not the rendered image:

- It is **reproducible without a vision model**, so the same protocol runs against a local
  text model, GPT-4o, or Claude, and the arms stay comparable.
- It **isolates the question worth asking**. APEX's advantage over a generalist could be
  either that it extracts better features from the signal, or that it reasons better about
  the features. Handing the generalist the *same* measurements APEX's own delineator
  produces removes the first explanation, so what remains is the second.
- It is **generous to the generalist**, which matters for an honest comparison. The model
  is not asked to find a QRS complex in a picture; it is given the interval already
  measured. Any gap that survives that handicap is a real one.

The obvious objection is that measurements discard morphology — a Q wave's shape, an rSR'
pattern, the exact contour of a T wave — and that is true. The protocol therefore
under-states what a strong multimodal model could do from the image, and the report says so
rather than treating this as the last word on generalist ECG reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.longitudinal.intervals import IntervalSet, measure

# The five PTB-XL diagnostic superclasses, the shared prediction target.
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

SUPERCLASS_GLOSS = {
    "NORM": "normal ECG",
    "MI": "myocardial infarction (old or acute)",
    "STTC": "ST/T change (ischemia, repolarization abnormality)",
    "CD": "conduction disturbance (AV block, bundle branch block, fascicular block)",
    "HYP": "hypertrophy (ventricular or atrial enlargement)",
}

SYSTEM_PROMPT = (
    "You are an experienced cardiologist reading a 12-lead ECG. You are given the "
    "standard measurements. Answer only in the requested format, with no preamble."
)

PROMPT_TEMPLATE = """12-lead ECG measurements:
{measurements}

Rate the likelihood of each diagnostic category on a 0-10 scale (0 = certainly absent,
10 = certainly present). More than one may be present. Categories:
{categories}

Reply in EXACTLY this format and nothing else:
NORM: <number>
MI: <number>
STTC: <number>
CD: <number>
HYP: <number>
INTERPRETATION: <one sentence>"""


@dataclass
class ECGFeatures:
    """The measured view of one recording, plus its rendered prompt."""

    intervals: IntervalSet
    text: str
    measurable: bool


def _fmt(value: float | None, unit: str, digits: int = 0) -> str:
    if value is None:
        return "not measurable"
    return f"{value:.{digits}f} {unit}"


def describe(intervals: IntervalSet) -> str:
    """The measurement block, in the register a report would use.

    Unmeasurable quantities are stated as such rather than omitted. A missing PR interval
    in atrial fibrillation is clinically informative, and silently dropping the line would
    let the model read the absence as normality — the same reasoning as Phase 22's
    "not compared" lines.
    """
    lines = [
        f"Heart rate: {_fmt(intervals.heart_rate, 'bpm')}",
        f"PR interval: {_fmt(intervals.pr, 'ms')}",
        f"QRS duration: {_fmt(intervals.qrs, 'ms')}",
        f"QT interval: {_fmt(intervals.qt, 'ms')}",
        f"QTc (Fridericia): {_fmt(intervals.qtc_fridericia, 'ms')}",
    ]
    if not intervals.p_detected:
        lines.append("P waves: not detectable")
    if intervals.st_level:
        st = ", ".join(f"{lead} {value:+.2f}" for lead, value in intervals.st_level.items())
        lines.append(f"ST level at J+60 ms (mV): {st}")
    if intervals.t_amplitude:
        t = ", ".join(f"{lead} {value:+.2f}" for lead, value in intervals.t_amplitude.items())
        lines.append(f"T-wave amplitude (mV): {t}")
    if intervals.quality != "ok":
        lines.append(f"Signal quality: {intervals.quality}")
    return "\n".join(lines)


def build_prompt(intervals: IntervalSet) -> str:
    categories = "\n".join(f"- {code}: {SUPERCLASS_GLOSS[code]}" for code in SUPERCLASSES)
    return PROMPT_TEMPLATE.format(measurements=describe(intervals), categories=categories)


def extract(signal: np.ndarray, fs: int = 100) -> ECGFeatures:
    """Measure a raw ``(12, T)`` recording and render it as a prompt."""
    intervals = measure(np.asarray(signal, dtype=float), fs)
    return ECGFeatures(intervals=intervals, text=build_prompt(intervals),
                       measurable=intervals.measurable)
