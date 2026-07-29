"""Phase 13 — adversarial & edge-case curation and failure analysis.

Deployment-relevant behaviour lives at the edges: noisy recordings, findings sitting on
the confidence threshold, rare labels, records carrying several conditions at once, and
the plain-normal ECG the system must *not* alarm on. This module holds the reusable,
model-free logic for building those cohorts from PTB-XL metadata + detector outputs and
for scoring what the system got wrong on each record. The heavy lifting (running the
detector, running the full report pipeline on concrete examples) lives in
``scripts/edge_case_report.py``; everything here operates on plain arrays / dicts so it
is testable without torch or the dataset.

Two failure axes are tracked throughout, because they carry very different clinical cost:

- a **miss** — a label that is present in the ground truth but the system did not surface
  (a silent false negative). A miss of an *urgent* code is a "dangerous miss".
- an **over-flag** — a label the system surfaced that is not present (a false positive),
  the driver of alarm fatigue.

Everything is measured at the **deployed surfacing rule** (probability ≥
``review_threshold``, 0.5) — i.e. what the shipped pipeline actually says — not at a
post-hoc F1-tuned threshold, so the numbers describe real behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- PTB-XL artifact metadata ------------------------------------------------
# PTB-XL annotates signal-quality problems per record in four free-text columns
# (baseline_drift, static_noise, burst_noise, electrodes_problems). The value lists the
# affected leads; the German token "alles" means "all leads" — whole-record corruption,
# the worst case.
def parse_affected(value) -> list[str]:
    """Parse one PTB-XL noise-annotation cell into a list of affected-lead tokens.

    ``["ALL"]`` marks whole-record noise (PTB-XL's "alles"). Empty / missing -> ``[]``.
    """
    if value is None or not isinstance(value, str):
        return []
    s = value.strip().lower()
    if not s:
        return []
    if "alles" in s:
        return ["ALL"]
    return [tok.strip().upper() for tok in s.replace(",", " ").split() if tok.strip()]


@dataclass
class ArtifactProfile:
    """Signal-quality summary for one record, pooled over the four noise columns."""

    whole_record: bool          # any column marked "alles" (all leads corrupted)
    n_types: int                # how many of the 4 noise types are annotated
    burst_or_electrode: bool    # burst noise or an electrode problem (the coarse artifacts)
    affected: dict[str, list[str]] = field(default_factory=dict)

    @property
    def any_noise(self) -> bool:
        return self.n_types > 0

    @property
    def significant(self) -> bool:
        """Artifact bad enough to expect it to move the model: whole-record noise, two or
        more noise types at once, or a burst / electrode problem."""
        return self.whole_record or self.n_types >= 2 or self.burst_or_electrode


def artifact_profile(baseline_drift=None, static_noise=None, burst_noise=None,
                     electrodes_problems=None) -> ArtifactProfile:
    """Pool the four PTB-XL noise columns of one record into an :class:`ArtifactProfile`."""
    cols = {
        "baseline_drift": parse_affected(baseline_drift),
        "static_noise": parse_affected(static_noise),
        "burst_noise": parse_affected(burst_noise),
        "electrodes_problems": parse_affected(electrodes_problems),
    }
    affected = {k: v for k, v in cols.items() if v}
    whole = any("ALL" in v for v in affected.values())
    return ArtifactProfile(
        whole_record=whole,
        n_types=len(affected),
        burst_or_electrode=bool(cols["burst_noise"] or cols["electrodes_problems"]),
        affected=affected,
    )


# --- Per-record outcome ------------------------------------------------------
def surfaced_from_probs(
    probs_row: np.ndarray, label_space: list[str], threshold: float = 0.5,
) -> tuple[list[str], dict[str, float]]:
    """The codes the deployed pipeline surfaces (prob ≥ threshold) + their confidences."""
    surfaced, confidences = [], {}
    for j, code in enumerate(label_space):
        p = float(probs_row[j])
        if p >= threshold:
            surfaced.append(code)
            confidences[code] = p
    return surfaced, confidences


@dataclass
class RecordOutcome:
    """What the system got right and wrong on one record, at the deployed threshold."""

    ecg_id: int
    present: list[str]           # ground-truth present codes (in the 71-label space)
    surfaced: list[str]          # codes surfaced at prob ≥ threshold
    misses: list[str]            # present but not surfaced (silent false negatives)
    false_positives: list[str]   # surfaced but not present (over-flags)
    missed_urgent: list[str]     # present urgent codes not surfaced (dangerous misses)
    near_miss: list[str]         # missed present codes whose prob sat just below threshold
    max_prob: float              # strongest surfaced-or-not probability on the record
    top_miss_prob: float         # highest prob among the missed present codes (0 if none)

    @property
    def n_present(self) -> int:
        return len(self.present)

    @property
    def correct_silent(self) -> bool:
        """Nothing present was missed and nothing absent was surfaced."""
        return not self.misses and not self.false_positives


def evaluate_record(
    ecg_id: int,
    present: list[str],
    probs_row: np.ndarray,
    label_space: list[str],
    urgent: frozenset[str] | set[str] = frozenset(),
    threshold: float = 0.5,
    near_band: float = 0.15,
) -> RecordOutcome:
    """Score one record's detector output against its ground-truth present codes.

    ``present`` are the codes truly present (restricted here to the label space).
    ``near_band`` defines a "near miss": a present code that was missed but whose
    probability sat within ``near_band`` below the threshold (e.g. 0.35–0.5) — a finding
    the system came within a hair of surfacing, then dropped silently.
    """
    idx = {c: i for i, c in enumerate(label_space)}
    present = [c for c in present if c in idx]
    surfaced, _ = surfaced_from_probs(probs_row, label_space, threshold)
    surfaced_set = set(surfaced)
    present_set = set(present)

    misses = sorted(present_set - surfaced_set)
    false_positives = sorted(surfaced_set - present_set)
    missed_urgent = sorted(m for m in misses if m in urgent)
    near_miss = sorted(
        m for m in misses if threshold - near_band <= float(probs_row[idx[m]]) < threshold
    )
    miss_probs = [float(probs_row[idx[m]]) for m in misses]
    return RecordOutcome(
        ecg_id=int(ecg_id),
        present=present,
        surfaced=surfaced,
        misses=misses,
        false_positives=false_positives,
        missed_urgent=missed_urgent,
        near_miss=near_miss,
        max_prob=float(probs_row.max()),
        top_miss_prob=max(miss_probs) if miss_probs else 0.0,
    )


# --- Cohort selection (pure; operate on arrays / lists) ----------------------
def select_multicondition(present_lists: list[list[str]], min_codes: int = 4) -> list[int]:
    """Indices of records carrying at least ``min_codes`` present labels at once."""
    return [i for i, codes in enumerate(present_lists) if len(codes) >= min_codes]


def select_borderline(
    probs: np.ndarray, present_lists: list[list[str]], label_space: list[str],
    threshold: float = 0.5, band: float = 0.1,
) -> list[int]:
    """Indices where a *present* label's probability sits within ``band`` of ``threshold``.

    These are the knife's-edge records: a small shift in the score flips a correct call
    to a miss (or vice-versa), so they are exactly where a fixed threshold is fragile.
    """
    idx = {c: i for i, c in enumerate(label_space)}
    lo, hi = threshold - band, threshold + band
    out = []
    for r, codes in enumerate(present_lists):
        for c in codes:
            j = idx.get(c)
            if j is not None and lo <= float(probs[r, j]) <= hi:
                out.append(r)
                break
    return out


def rarest_labels(train_counts: dict[str, int], k: int = 8, min_count: int = 1) -> list[str]:
    """The ``k`` rarest labels that still have at least ``min_count`` training examples."""
    eligible = [(c, n) for c, n in train_counts.items() if n >= min_count]
    eligible.sort(key=lambda cn: cn[1])
    return [c for c, _ in eligible[:k]]


def select_carrying(present_lists: list[list[str]], labels: set[str]) -> list[int]:
    """Indices of records carrying at least one of ``labels`` as a present code."""
    return [i for i, codes in enumerate(present_lists) if labels & set(codes)]


# --- Cohort-level aggregation ------------------------------------------------
def cohort_metrics(outcomes: list[RecordOutcome]) -> dict:
    """Aggregate failure rates over a cohort of :class:`RecordOutcome`.

    - ``label_recall`` — fraction of present labels the system surfaced (1 − miss rate).
    - ``overflag_per_record`` — mean count of surfaced-but-absent labels per record.
    - ``dangerous_miss_rate`` — fraction of records with a missed *urgent* code.
    - ``clean_silent_rate`` — fraction of records with no miss and no over-flag.
    - ``near_miss_records`` — records with a present code dropped just below threshold.
    """
    n = len(outcomes)
    if n == 0:
        return {"n": 0}
    total_present = sum(o.n_present for o in outcomes)
    total_missed = sum(len(o.misses) for o in outcomes)
    return {
        "n": n,
        "n_present_labels": total_present,
        "n_missed_labels": total_missed,
        "label_recall": round(1 - total_missed / total_present, 4) if total_present else None,
        "miss_rate": round(total_missed / total_present, 4) if total_present else None,
        "overflag_per_record": round(sum(len(o.false_positives) for o in outcomes) / n, 4),
        "records_with_any_miss": sum(bool(o.misses) for o in outcomes),
        "records_with_any_overflag": sum(bool(o.false_positives) for o in outcomes),
        "dangerous_miss_records": sum(bool(o.missed_urgent) for o in outcomes),
        "dangerous_miss_rate": round(sum(bool(o.missed_urgent) for o in outcomes) / n, 4),
        "near_miss_records": sum(bool(o.near_miss) for o in outcomes),
        "clean_silent_rate": round(sum(o.correct_silent for o in outcomes) / n, 4),
    }
