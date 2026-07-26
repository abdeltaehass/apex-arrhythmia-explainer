"""Render a ``(12, T)`` signal onto a realistic paper-ECG image.

This is the *forward* half of Phase 10 — and the source of paired (image, signal)
training/validation data, since no dataset of real paper-ECG photos with ground-truth
signals is available. Drawing is done directly with PIL for exact pixel control, using
the standard ECG paper conventions:

    time      25 mm/s   (1 small 1 mm box = 0.04 s)
    amplitude 10 mm/mV  (1 small box = 0.1 mV)

Two layouts:

- ``"stacked"`` (default) — 12 full-width rows, each showing the whole record. Every
  lead carries the full duration, so it round-trips losslessly through the digitizer;
  this is the format the demo path uses.
- ``"3x4"`` — the clinical 4-column x 3-row mosaic + a lead-II rhythm strip. More
  familiar to look at, but each mosaic cell only holds 2.5 s of its lead, so it can't
  reconstruct a full 10 s / 12-lead tensor (a property of the paper format, not the
  digitizer). Provided for display / realism.

The grid is drawn only inside the plotting rectangle; lead labels sit in a left margin
*outside* it, so `digitization.grid` can recover the plot area from the grid's own
extent without tripping over the text.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MM_PER_S = 25.0          # standard paper speed
MM_PER_MV = 10.0         # standard gain
DEFAULT_PX_PER_MM = 5

# Layout geometry, in millimetres.
LEFT_MARGIN_MM = 14.0    # lead labels live here, left of the grid
SIDE_MARGIN_MM = 5.0
BAND_MM = 18.0           # vertical space per lead in the stacked layout (+-0.9 mV visible)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

_GRID_MINOR = (255, 190, 190)   # light pink, every 1 mm
_GRID_MAJOR = (240, 120, 120)   # darker red, every 5 mm
_TRACE = (10, 10, 10)
_BG = (255, 255, 255)


def _font(px: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", px)
    except Exception:
        return ImageFont.load_default()


def _draw_grid(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, ppm) -> None:
    """Minor (1 mm) + major (5 mm) grid inside the rectangle ``(x0, y0)-(x1, y1)``."""
    w_mm = round((x1 - x0) / ppm)
    h_mm = round((y1 - y0) / ppm)
    for i in range(w_mm + 1):
        x = x0 + i * ppm
        draw.line([(x, y0), (x, y1)], fill=_GRID_MAJOR if i % 5 == 0 else _GRID_MINOR,
                  width=2 if i % 5 == 0 else 1)
    for j in range(h_mm + 1):
        y = y0 + j * ppm
        draw.line([(x0, y), (x1, y)], fill=_GRID_MAJOR if j % 5 == 0 else _GRID_MINOR,
                  width=2 if j % 5 == 0 else 1)


def render_ecg(
    signal: np.ndarray,
    fs: int = 100,
    px_per_mm: int = DEFAULT_PX_PER_MM,
    layout: str = "stacked",
    mm_per_mv: float = MM_PER_MV,
) -> Image.Image:
    """Render ``signal`` ``(12, T)`` as a paper-ECG :class:`PIL.Image.Image`.

    ``mm_per_mv`` is the gain (10 = standard); lower it (e.g. 5, "half gain") for
    large-amplitude records. The digitizer assumes the same standard gain, so keep it
    at the default unless you also tell the digitizer.
    """
    if layout == "3x4":
        return _render_3x4(signal, fs, px_per_mm, mm_per_mv)
    return _render_stacked(signal, fs, px_per_mm, mm_per_mv)


def _render_stacked(signal, fs, ppm, mm_per_mv) -> Image.Image:
    n_leads, T = signal.shape
    duration_s = T / fs
    plot_w_mm = duration_s * MM_PER_S
    plot_h_mm = n_leads * BAND_MM

    x0 = round(LEFT_MARGIN_MM * ppm)
    y0 = round(SIDE_MARGIN_MM * ppm)
    x1 = x0 + round(plot_w_mm * ppm)
    y1 = y0 + round(plot_h_mm * ppm)
    W = x1 + round(SIDE_MARGIN_MM * ppm)
    H = y1 + round(SIDE_MARGIN_MM * ppm)

    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, x0, y0, x1, y1, ppm)
    font = _font(max(10, round(3.2 * ppm)))

    t_ms = np.arange(T) / fs            # seconds
    xs = x0 + t_ms * MM_PER_S * ppm     # pixel x per sample
    for i in range(n_leads):
        baseline = y0 + (i + 0.5) * BAND_MM * ppm
        ys = baseline - signal[i] * mm_per_mv * ppm
        # clip into the band so a big QRS doesn't bleed into the neighbour's row
        lo, hi = y0 + i * BAND_MM * ppm, y0 + (i + 1) * BAND_MM * ppm
        ys = np.clip(ys, lo, hi)
        pts = list(zip(xs.tolist(), ys.tolist(), strict=True))
        draw.line(pts, fill=_TRACE, width=2)
        draw.text((round(2 * ppm), round(baseline - 2 * ppm)),
                  LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"L{i}", fill=_TRACE, font=font)
    return img


def _render_3x4(signal, fs, ppm, mm_per_mv) -> Image.Image:
    """Clinical 4x3 mosaic + lead-II rhythm strip (display only — 2.5 s per cell)."""
    n_leads, T = signal.shape
    seg = T // 4                       # 2.5 s per column at 10 s total
    col_w_mm = (T / fs / 4) * MM_PER_S
    row_h_mm = BAND_MM
    rhythm_mm = BAND_MM

    x0 = round(LEFT_MARGIN_MM * ppm)
    y0 = round(SIDE_MARGIN_MM * ppm)
    plot_w_mm = 4 * col_w_mm
    plot_h_mm = 3 * row_h_mm + rhythm_mm
    x1 = x0 + round(plot_w_mm * ppm)
    y1 = y0 + round(plot_h_mm * ppm)
    W, H = x1 + round(SIDE_MARGIN_MM * ppm), y1 + round(SIDE_MARGIN_MM * ppm)

    img = Image.new("RGB", (W, H), _BG)
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, x0, y0, x1, y1, ppm)
    font = _font(max(9, round(3.0 * ppm)))

    order = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    for r, row in enumerate(order):
        for c, name in enumerate(row):
            idx = LEAD_NAMES.index(name)
            s = signal[idx, c * seg:(c + 1) * seg]
            baseline = y0 + (r + 0.5) * row_h_mm * ppm
            cx0 = x0 + c * col_w_mm * ppm
            xs = cx0 + (np.arange(len(s)) / fs) * MM_PER_S * ppm
            ys = np.clip(baseline - s * mm_per_mv * ppm,
                        y0 + r * row_h_mm * ppm, y0 + (r + 1) * row_h_mm * ppm)
            draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=_TRACE, width=2)
            draw.text((round(cx0 + 1 * ppm), round(baseline - 8 * ppm)), name, fill=_TRACE, font=font)
    # full-width lead-II rhythm strip along the bottom
    baseline = y0 + (3 + 0.5) * row_h_mm * ppm
    xs = x0 + (np.arange(T) / fs) * MM_PER_S * ppm
    ys = np.clip(baseline - signal[1] * mm_per_mv * ppm,
                y0 + 3 * row_h_mm * ppm, y1)
    draw.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=_TRACE, width=2)
    draw.text((round(2 * ppm), round(baseline - 8 * ppm)), "II", fill=_TRACE, font=font)
    return img
