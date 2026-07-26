"""Digitize a paper-ECG image back into a ``(12, T)`` numeric signal.

The *inverse* of `digitization.render`, and the Phase-10 deliverable: an image in, a
signal tensor out that drops straight into the Phase-2 preprocessing pipeline. The
approach is classical computer vision (no training data needed, and none of paper-ECG
photos exists to train on):

    1. `grid.detect_grid`  -> plotting rectangle + px-per-mm calibration
    2. threshold           -> dark trace pixels, isolated from the lighter grid
    3. split the plot into 12 equal lead bands (the "stacked" layout)
    4. per column, take the darkness-weighted centroid row -> the waveform's y(x)
    5. px -> mV via the grid pitch and the standard 10 mm/mV gain; baseline removed
    6. resample each lead's per-column series to ``fs_out`` -> ``(12, T)``

A learned trace/grid segmentation model is the natural upgrade for messy real-world
photos (perspective, shadows, JPEG); the classical path is accurate and dependency-light
on clean scans/renders, which is what the round-trip evaluation exercises.
"""

from __future__ import annotations

import io

import numpy as np

from src.config import NUM_LEADS
from src.digitization.grid import GridInfo, detect_grid
from src.digitization.render import MM_PER_MV, MM_PER_S

_LUMA = np.array([0.299, 0.587, 0.114])


def _to_rgb(image) -> np.ndarray:
    return np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)[..., :3]


def _to_gray(image) -> np.ndarray:
    return _to_rgb(image).astype(np.float32) @ _LUMA


def _ink_weight(gray: np.ndarray) -> np.ndarray:
    """Darkness of the trace ink below an adaptive threshold.

    The cut-off is a fraction of the paper-white reference (the 92nd-percentile
    luminance), so dimming / JPEG shifts don't wipe the trace the way a fixed value
    does. The grid (even the darker 5 mm lines) is lighter than the black trace and
    sits above the threshold, so luminance alone separates them — a colour test was
    tried too but JPEG paints mildly-reddish halos onto the dark trace, so it wrongly
    erased ink; luminance-only is both simpler and more robust.
    """
    bg = float(np.percentile(gray, 92)) or 255.0
    thr = 0.55 * bg
    return np.clip(thr - gray, 0.0, None)


def _trace_centroids(dark_weight: np.ndarray) -> np.ndarray:
    """Per-column darkness-weighted row centroid; NaN where a column has no ink."""
    h = dark_weight.shape[0]
    rows = np.arange(h)[:, None]
    col_mass = dark_weight.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroids = (dark_weight * rows).sum(axis=0) / col_mass
    centroids[col_mass <= 0] = np.nan
    return centroids


def _fill_nans(y: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN gaps (and flat-fill any leading/trailing NaNs)."""
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if not good.any():
        return np.zeros_like(y)
    return np.interp(idx, idx[good], y[good])


def digitize_image(
    image,
    fs_out: int = 100,
    n_leads: int = NUM_LEADS,
    mm_per_mv: float = MM_PER_MV,
    grid: GridInfo | None = None,
) -> np.ndarray:
    """Paper-ECG image -> ``(n_leads, T)`` float32 signal in millivolts.

    ``image`` is a PIL image or ``(H, W, 3)`` array in the stacked 12-row layout. ``T``
    is derived from the plot width and the 25 mm/s paper speed, resampled to ``fs_out``.
    Per-lead baseline (isoelectric line) is removed; absolute gain assumes the standard
    ``mm_per_mv``.
    """
    gray = _to_gray(image)
    grid = grid or detect_grid(image)
    x0, y0, x1, y1 = grid.bbox
    ppm = grid.px_per_mm

    # Trace ink = dark pixels inside the plotting rectangle.
    dark_weight = _ink_weight(gray)[y0:y1 + 1, x0:x1 + 1]

    band_edges = np.linspace(0, dark_weight.shape[0], n_leads + 1).round().astype(int)
    duration_s = (x1 - x0) / ppm / MM_PER_S
    t_out = max(1, round(duration_s * fs_out))

    leads = np.empty((n_leads, t_out), dtype=np.float32)
    for i in range(n_leads):
        band = dark_weight[band_edges[i]:band_edges[i + 1], :]
        y_px = _fill_nans(_trace_centroids(band))
        baseline = np.median(y_px)                      # isoelectric line
        mv = (baseline - y_px) / ppm / mm_per_mv        # up (smaller y) -> positive mV
        # resample the per-column series (uniform in x, i.e. in time) to fs_out
        src_t = np.linspace(0.0, 1.0, len(mv))
        dst_t = np.linspace(0.0, 1.0, t_out)
        leads[i] = np.interp(dst_t, src_t, mv).astype(np.float32)
    return leads


class ImageDecodeError(ValueError):
    """The uploaded bytes could not be decoded as an image."""


def digitize_bytes(content: bytes, fs_out: int = 100) -> tuple[np.ndarray, int]:
    """Raw image bytes -> ``((12, T) float32, fs_out)`` for the upload path.

    Raises :class:`ImageDecodeError` if the bytes aren't a readable image, so the API
    layer can return a clear 4xx instead of a stack trace.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageDecodeError(f"could not decode uploaded image: {e}") from e
    return digitize_image(image, fs_out=fs_out), fs_out
