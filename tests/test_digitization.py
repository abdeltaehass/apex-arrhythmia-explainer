"""Phase-10 tests: render -> digitize round-trip, grid detection, upload routing.

All synthetic (no PTB-XL, no torch): a smooth multi-sine 12-lead signal round-trips
through the paper-image renderer and back, so the numbers are deterministic and the
suite runs anywhere.
"""

import io

import numpy as np
import pytest

from src.digitization import LEAD_NAMES, detect_grid, digitize_image, render_ecg
from src.digitization.augment import photograph
from src.digitization.digitize import ImageDecodeError, digitize_bytes
from src.digitization.render import DEFAULT_PX_PER_MM


def _smooth_signal(T: int = 1000, fs: int = 100) -> np.ndarray:
    """A gentle multi-sine per lead — smooth enough to round-trip near-losslessly."""
    t = np.arange(T) / fs
    return np.stack([0.5 * np.sin(2 * np.pi * (0.7 + 0.12 * i) * t + 0.3 * i)
                    for i in range(12)]).astype(np.float32)


def _fidelity(orig: np.ndarray, recon: np.ndarray) -> float:
    """Mean per-lead correlation, resampled to a common length + best small-lag align."""
    T = orig.shape[1]
    corrs = []
    for i in range(12):
        ri = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, recon.shape[1]), recon[i])
        best = -1.0
        for lag in range(-4, 5):
            b = np.roll(ri, lag)[6:T - 6]
            a = orig[i][6:T - 6]
            a, b = a - a.mean(), b - b.mean()
            best = max(best, float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        corrs.append(best)
    return float(np.mean(corrs))


# --- render ------------------------------------------------------------------
def test_render_produces_rgb_image():
    img = render_ecg(_smooth_signal(), fs=100)
    assert img.mode == "RGB"
    assert img.width > img.height          # 10 s at 25 mm/s is wide
    assert min(img.size) > 100


def test_render_3x4_layout_smoke():
    img = render_ecg(_smooth_signal(), fs=100, layout="3x4")
    assert img.mode == "RGB" and min(img.size) > 100


# --- grid detection ----------------------------------------------------------
def test_detect_grid_recovers_pitch_and_bbox():
    img = render_ecg(_smooth_signal(), fs=100, px_per_mm=DEFAULT_PX_PER_MM)
    grid = detect_grid(img)
    assert abs(grid.px_per_mm - DEFAULT_PX_PER_MM) <= 1.0     # ~5 px/mm recovered
    x0, y0, x1, y1 = grid.bbox
    assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height


def test_detect_grid_scales_with_render_resolution():
    img = render_ecg(_smooth_signal(), fs=100, px_per_mm=8)
    assert abs(detect_grid(img).px_per_mm - 8) <= 1.5


# --- round trip --------------------------------------------------------------
def test_round_trip_shape_matches_duration():
    recon = digitize_image(render_ecg(_smooth_signal(T=1000, fs=100), fs=100), fs_out=100)
    assert recon.shape[0] == 12
    assert abs(recon.shape[1] - 1000) <= 20         # ~10 s recovered


def test_round_trip_high_fidelity():
    orig = _smooth_signal()
    recon = digitize_image(render_ecg(orig, fs=100), fs_out=100)
    assert _fidelity(orig, recon) > 0.9             # smooth signal round-trips cleanly


def test_round_trip_recovers_amplitude_scale():
    orig = _smooth_signal()
    recon = digitize_image(render_ecg(orig, fs=100), fs_out=100)
    # per-lead std should match within tolerance (standard 10 mm/mV gain both ways)
    o_std = orig.std(axis=1)
    r_std = recon[:, :orig.shape[1]].std(axis=1)
    assert np.allclose(o_std, r_std, atol=0.08)


def test_round_trip_degrades_gracefully_under_photo_noise():
    orig = _smooth_signal()
    img = photograph(render_ecg(orig, fs=100), level=0.5, rng=np.random.default_rng(0))
    assert _fidelity(orig, digitize_image(img)) > 0.7


# --- bytes / decode ----------------------------------------------------------
def test_digitize_bytes_from_png():
    img = render_ecg(_smooth_signal(), fs=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    signal, fs = digitize_bytes(buf.getvalue())
    assert signal.shape[0] == 12 and fs == 100


def test_digitize_bytes_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        digitize_bytes(b"\x89PNG\r\n this is not an image")


# --- upload routing ----------------------------------------------------------
def test_parse_signal_upload_routes_image_to_digitizer():
    from src.serving.loaders import parse_signal_upload

    img = render_ecg(_smooth_signal(), fs=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    signal, sr = parse_signal_upload("ecg.png", buf.getvalue())
    assert signal.shape[0] == 12 and sr == 100


def test_lead_names_are_twelve():
    assert len(LEAD_NAMES) == 12
