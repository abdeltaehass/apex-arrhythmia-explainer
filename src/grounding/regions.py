"""Map a saliency trace onto ECG wave regions (P / QRS / ST / T).

To sanity-check grounding against clinical intuition we need to know *which part of the
cardiac cycle* a saliency peak lands on. Given R-peak indices (from
`src/preprocessing/segmentation.pan_tompkins_rpeaks`) we carve fixed, beat-relative
windows around each beat and measure how much of the saliency mass falls in each:

    P    wave : R - 200 ms  ..  R - 80 ms   (atrial depolarization; before QRS)
    QRS      : R -  50 ms  ..  R + 50 ms    (ventricular depolarization; the spike)
    ST  segment: R + 80 ms  ..  R + 200 ms  (early repolarization; J-point onward)
    T    wave : R + 200 ms  ..  R + 400 ms  (ventricular repolarization)

These are approximate windows at normal rates, not fiducial-point detections — enough to
answer "did the model look at the ST/T region or the P wave?" without a full delineator.
The ``baseline`` region is everything not covered by any window (the T-P interval), which
matters for rhythm findings whose evidence is *between* beats (irregular RR, absent P).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Region name -> (start, end) offset from the R-peak, in seconds.
WAVE_WINDOWS: dict[str, tuple[float, float]] = {
    "P": (-0.20, -0.08),
    "QRS": (-0.05, 0.05),
    "ST": (0.08, 0.20),
    "T": (0.20, 0.40),
}
REGIONS = (*WAVE_WINDOWS.keys(), "baseline")


@dataclass
class RegionMass:
    """How a 1-D saliency trace distributes across the wave regions.

    ``fractions`` sums to 1 across ``REGIONS`` (P, QRS, ST, T, baseline); each value is
    the share of total saliency mass in that region. ``n_beats`` is how many R-peaks
    contributed, ``rr_cv`` the coefficient of variation of the RR intervals (a rhythm-
    regularity measure), and ``bpm`` the mean heart rate.
    """

    fractions: dict[str, float]
    n_beats: int
    rr_cv: float
    bpm: float

    def dominant(self) -> str:
        return max(self.fractions, key=self.fractions.get)

    def repolarization(self) -> float:
        """Combined ST + T share — the repolarization segment, where ST/T findings live."""
        return self.fractions.get("ST", 0.0) + self.fractions.get("T", 0.0)


def _to_1d(saliency: np.ndarray, lead: int | None) -> np.ndarray:
    """Collapse a ``(12, T)`` per-lead trace to ``(T,)`` (a chosen lead, or lead-max)."""
    s = np.asarray(saliency, dtype=float)
    if s.ndim == 1:
        return s
    if s.ndim == 2:
        return s[lead] if lead is not None else s.max(axis=0)
    raise ValueError(f"saliency must be (T,) or (12, T), got shape {s.shape}")


# When windows from neighbouring beats overlap (fast rates), a sample is assigned to
# the single most reliably-identifiable wave: QRS first, then the repolarization
# segment (ST, T), and the P wave last. Resolving overlaps toward repolarization rather
# than P keeps a prior beat's T wave from masquerading as the next beat's P wave.
_PRIORITY = ("P", "T", "ST", "QRS")  # assigned low->high; later writes win


def region_masks(n: int, rpeaks: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    """Mutually exclusive masks over ``n`` samples for each wave region + baseline.

    Every sample is assigned to at most one region (a partition), so downstream mass
    fractions sum to 1 even when beats are close enough that windows would overlap.
    Overlaps are resolved by ``_PRIORITY``; ``baseline`` is whatever no window covers.
    """
    assigned = np.full(n, "baseline", dtype=object)
    for name in _PRIORITY:
        lo_s, hi_s = WAVE_WINDOWS[name]
        for r in rpeaks:
            lo = max(0, int(round(r + lo_s * fs)))
            hi = min(n, int(round(r + hi_s * fs)))
            if hi > lo:
                assigned[lo:hi] = name
    return {name: (assigned == name) for name in REGIONS}


def rr_coefficient_of_variation(rpeaks: np.ndarray, fs: int) -> float:
    """CV (std/mean) of RR intervals — 0 for a metronome, high for irregular rhythms.

    Atrial fibrillation is "irregularly irregular": CV typically well above the ~0.05
    of normal sinus rhythm. NaN if fewer than 3 beats.
    """
    if len(rpeaks) < 3:
        return float("nan")
    rr = np.diff(np.sort(rpeaks)) / fs
    return float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else float("nan")


def saliency_by_region(
    saliency: np.ndarray,
    rpeaks: np.ndarray,
    fs: int = 100,
    lead: int | None = None,
) -> RegionMass:
    """Distribute a saliency trace across P / QRS / ST / T / baseline regions.

    Args:
        saliency: ``(T,)`` temporal trace or ``(12, T)`` per-lead trace. For the latter,
            pass ``lead`` to analyze one lead, else the per-time max across leads is used.
        rpeaks: R-peak sample indices in the same time base as ``saliency``.
        fs: sampling rate (Hz).
        lead: optional lead index to select from a per-lead trace.

    Returns:
        A :class:`RegionMass`. If there are no beats, all mass is reported as ``baseline``.
    """
    s = _to_1d(saliency, lead)
    n = s.shape[-1]
    total = float(s.sum())
    from src.preprocessing.segmentation import heart_rate_bpm

    bpm = heart_rate_bpm(np.asarray(rpeaks), fs)
    rr_cv = rr_coefficient_of_variation(np.asarray(rpeaks), fs)

    if total <= 0 or len(rpeaks) == 0:
        fractions = {r: 0.0 for r in REGIONS}
        fractions["baseline"] = 1.0 if total <= 0 else 0.0
        return RegionMass(fractions, int(len(rpeaks)), rr_cv, bpm)

    masks = region_masks(n, np.asarray(rpeaks), fs)
    fractions = {name: float(s[mask].sum()) / total for name, mask in masks.items()}
    return RegionMass(fractions, int(len(rpeaks)), rr_cv, bpm)
