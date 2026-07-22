"""PTB-XL row -> one supervised fine-tuning example.

Turns a PTB-XL record (its SCP-code label set + demographics + a real measured heart
rate) into a :class:`~src.generation.templater.StructuredInput` and its target
Findings/Impression text, i.e. one (input, target) pair for `train_lora.py`. Ground
truth confidences are synthetically sampled per finding (mostly high, occasionally
sub-threshold) so the model sees the "(requires clinician confirmation)" phrasing
during training even though PTB-XL labels themselves are binary presence, not
graded confidence — the target text is always rendered from the *same* sampled
confidence, so every training pair stays internally consistent by construction.

``heart_axis`` mapping: PTB-XL's own data dictionary does not spell these codes out
inline; the mapping below is corroborated against the published demographics (52%
male / 48% female matches ``sex==0`` male in this data) and secondary descriptions of
the extraction tool's axis categories. Only the two unambiguous *abnormal* buckets
(LAD/RAD and their extreme forms) are surfaced into the report — axis orientation
codes of uncertain clinical framing (AXL/AXR/SAG) and the normal bucket (MID) are
tracked but not templated into text, to avoid asserting something not confidently known.
"""

from __future__ import annotations

import numpy as np

from src.generation.templater import StructuredInput, build_structured_input, render_report
from src.generation.vocab import VOCAB

SEX_MAP = {0: "M", 1: "F"}

# code -> (spoken phrase, surface it in the report?)
AXIS_MAP: dict[str, tuple[str, bool]] = {
    "LAD": ("left axis deviation", True),
    "RAD": ("right axis deviation", True),
    "ALAD": ("extreme left axis deviation", True),
    "ARAD": ("extreme right axis deviation", True),
    "MID": ("normal axis", False),
    "AXL": ("horizontal axis", False),
    "AXR": ("vertical axis", False),
    "SAG": ("sagittal (S1S2S3) axis pattern", False),
}

LOW_CONF_PROB = 0.15   # fraction of findings trained with a sub-threshold confidence
HIGH_CONF_RANGE = (0.82, 0.99)
LOW_CONF_RANGE = (0.30, 0.49)


def present_codes_in_vocab(scp_codes: dict[str, float]) -> list[str]:
    """Present SCP codes restricted to ones the report vocabulary knows how to phrase."""
    return sorted(c for c in scp_codes if c in VOCAB)


def sample_confidences(codes: list[str], rng: np.random.Generator,
                       low_conf_prob: float = LOW_CONF_PROB) -> dict[str, float]:
    """Per-finding confidence: mostly high (ground truth), occasionally sub-threshold.

    The sub-threshold minority teaches the "(requires clinician confirmation)" clause
    without which the model would never see it in training (PTB-XL labels carry no
    natural confidence signal on their own).
    """
    out = {}
    for c in codes:
        lo, hi = LOW_CONF_RANGE if rng.random() < low_conf_prob else HIGH_CONF_RANGE
        out[c] = float(rng.uniform(lo, hi))
    return out


def axis_phrase(heart_axis) -> str | None:
    if not isinstance(heart_axis, str) or heart_axis not in AXIS_MAP:
        return None
    phrase, surface = AXIS_MAP[heart_axis]
    return phrase if surface else None


def row_to_structured_input(
    row,
    rng: np.random.Generator,
    heart_rate_bpm: float | None = None,
    review_threshold: float = 0.5,
) -> StructuredInput | None:
    """One `ptbxl_database.csv` row (+ a measured rate) -> a :class:`StructuredInput`.

    Returns ``None`` if the record has no SCP code the vocabulary recognizes (should not
    happen for the 71-code label space, but guards malformed rows).
    """
    codes = present_codes_in_vocab(row["scp_codes"])
    if not codes:
        return None
    confidences = sample_confidences(codes, rng)
    sex = SEX_MAP.get(int(row["sex"])) if row.get("sex") is not None else None
    age = row.get("age")
    age = int(age) if age is not None and age == age and age < 200 else None  # NaN + PTB-XL's 300 sentinel
    return build_structured_input(
        codes,
        confidences=confidences,
        review_threshold=review_threshold,
        record_id=int(row.name) if hasattr(row, "name") else None,
        age=age,
        sex=sex,
        heart_rate_bpm=heart_rate_bpm,
        heart_axis=axis_phrase(row.get("heart_axis")),
        original_report=row.get("report"),
    )


def build_example(row, rng: np.random.Generator, heart_rate_bpm: float | None = None,
                  review_threshold: float = 0.5) -> dict | None:
    """Full training example: structured input + rendered target, JSONL-ready.

    Returns ``None`` when the row has no recognized finding (mirrors
    :func:`row_to_structured_input`).
    """
    si = row_to_structured_input(row, rng, heart_rate_bpm, review_threshold)
    if si is None:
        return None
    target = render_report(si)
    return {
        "ecg_id": si.record_id,
        "codes": si.codes(),
        "age": si.age,
        "sex": si.sex,
        "heart_rate_bpm": si.heart_rate_bpm,
        "heart_axis": si.heart_axis,
        "confidences": {f.code: f.confidence for f in si.findings},
        "leads": {f.code: f.leads for f in si.findings},
        "review_threshold": si.review_threshold,
        "original_report": si.original_report,
        "findings": target["findings"],
        "impression": target["impression"],
    }


def example_to_structured_input(example: dict) -> StructuredInput:
    """Rebuild the `StructuredInput` a stored JSONL example was built from.

    Round-trips `build_example`'s flat record back into the object `prompts.py` and
    `train_lora.py` expect, so the dataset file is the single source of truth (no need
    to re-derive it from the raw PTB-XL row at train/inference time).
    """
    return build_structured_input(
        example["codes"],
        confidences=example.get("confidences"),
        leads_by_code=example.get("leads"),
        review_threshold=example.get("review_threshold", 0.5),
        record_id=example.get("ecg_id"),
        age=example.get("age"),
        sex=example.get("sex"),
        heart_rate_bpm=example.get("heart_rate_bpm"),
        heart_axis=example.get("heart_axis"),
        original_report=example.get("original_report"),
    )
