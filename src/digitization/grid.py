"""Recover the ECG grid from a rendered/photographed strip.

Two things the digitizer needs before it can read a trace:

- **the plotting rectangle** — where the grid actually is, so labels/margins are
  excluded. Found from the reddish grid ink's bounding box.
- **the grid pitch** (pixels per millimetre) — the calibration that turns pixel
  offsets into millimetres, and hence (with 25 mm/s, 10 mm/mV) into time and mV.
  Found from the *median spacing* between detected vertical grid lines: minor (1 mm)
  lines outnumber major (5 mm) ones, so adjacent-line spacing is the 1 mm box. (An
  autocorrelation peak, by contrast, lands on the stronger 5 mm major period.)

Everything is plain numpy so it runs without OpenCV / scikit-image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridInfo:
    x0: int
    y0: int
    x1: int
    y1: int
    px_per_mm: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def reddish_mask(rgb: np.ndarray, margin: int = 12, floor: int = 110) -> np.ndarray:
    """Boolean mask of pink/red grid pixels (R clearly above G and B)."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    return (r > g + margin) & (r > b + margin) & (r > floor)


def _extent(projection: np.ndarray, frac: float = 0.15) -> tuple[int, int]:
    """First/last index where a 1-D projection exceeds ``frac`` of its own max."""
    if projection.max() <= 0:
        return 0, len(projection) - 1
    hits = np.flatnonzero(projection > frac * projection.max())
    return int(hits[0]), int(hits[-1])


def _line_spacing(profile: np.ndarray, min_px: int, max_px: int) -> float | None:
    """Median spacing between grid-line peaks in a 1-D line-strength ``profile`` (px).

    Detects every vertical grid line (minor *and* major) as a peak, then takes the
    median consecutive spacing — the 1 mm minor pitch, since minor lines are adjacent
    to everything. Robust to a few missed lines and to the darker major lines.
    """
    from scipy.signal import find_peaks

    if profile.max() <= 0:
        return None
    peaks, _ = find_peaks(profile, distance=max(1, min_px - 1), height=0.25 * profile.max())
    if len(peaks) < 3:
        return None
    spacings = np.diff(peaks)
    spacings = spacings[(spacings >= min_px) & (spacings <= max_px)]
    return float(np.median(spacings)) if len(spacings) else None


def detect_grid(image, min_mm_px: int = 2, max_mm_px: int = 40) -> GridInfo:
    """Locate the plotting rectangle and the mm grid pitch of an ECG image.

    ``image`` is a PIL image or an ``(H, W, 3)`` uint8 array. ``min_mm_px``/``max_mm_px``
    bound the search for the 1 mm box size in pixels. Falls back to the full-image
    bounding box / a default pitch when the grid is too faint to measure.
    """
    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    mask = reddish_mask(rgb)
    col_proj = mask.sum(axis=0).astype(float)
    row_proj = mask.sum(axis=1).astype(float)
    x0, x1 = _extent(col_proj)
    y0, y1 = _extent(row_proj)

    # Pitch from the vertical grid lines' spacing, using the full plot height so faint
    # minor lines accumulate (traces only occupy a thin row band per lead).
    line_profile = mask[y0:y1 + 1, x0:x1 + 1].sum(axis=0).astype(float)
    period = _line_spacing(line_profile, min_mm_px, max_mm_px)
    if period is None:  # grid too faint to measure -> conservative default
        period = 5.0
    return GridInfo(x0=x0, y0=y0, x1=x1, y1=y1, px_per_mm=period)
