"""Unit tests for the Phase 2 preprocessing chain (synthetic signals, no data/torch)."""

import numpy as np
import pytest

from src.preprocessing.filters import bandpass
from src.preprocessing.normalization import zscore
from src.preprocessing.pipeline import Preprocessor, preprocess
from src.preprocessing.resample import resample_signal
from src.preprocessing.segmentation import (
    heart_rate_bpm,
    pan_tompkins_rpeaks,
    segment_beats,
)

FS = 100
T = 10 * FS  # 10 s


def _synthetic_ecg(fs=FS, seconds=10, bpm=60, noise=0.01, seed=0):
    """12-lead-ish signal: a QRS-like spike train at a fixed rate + baseline wander."""
    rng = np.random.default_rng(seed)
    n = fs * seconds
    t = np.arange(n) / fs
    period = int(round(fs * 60 / bpm))
    peaks = np.arange(period // 2, n, period)
    beat = np.zeros(n)
    for p in peaks:
        lo, hi = max(0, p - 3), min(n, p + 4)
        beat[lo:hi] += np.hanning(hi - lo)  # narrow QRS bump
    wander = 0.5 * np.sin(2 * np.pi * 0.15 * t)  # 0.15 Hz baseline wander
    lead = beat + wander + noise * rng.standard_normal(n)
    sig = np.stack([lead] * 12)  # (12, n)
    return sig.astype(np.float32), peaks


# --- resample ---------------------------------------------------------------
def test_resample_identity_when_equal():
    sig, _ = _synthetic_ecg()
    out = resample_signal(sig, FS, FS)
    assert out.shape == sig.shape


def test_resample_500_to_100_length():
    sig = np.random.default_rng(0).standard_normal((12, 5000)).astype(np.float32)
    out = resample_signal(sig, 500, 100)
    assert out.shape == (12, 1000)


def test_resample_preserves_low_freq_sine():
    t = np.arange(5000) / 500
    sig = np.stack([np.sin(2 * np.pi * 5 * t)] * 12).astype(np.float32)  # 5 Hz
    out = resample_signal(sig, 500, 100)
    # 5 Hz is well below the 50 Hz Nyquist of 100 Hz -> amplitude ~preserved
    assert 0.8 < np.abs(out).max() < 1.2


# --- bandpass ---------------------------------------------------------------
def test_bandpass_removes_baseline_wander():
    sig, _ = _synthetic_ecg(noise=0.0)
    bp = bandpass(sig, FS)

    # sub-0.5 Hz energy (baseline wander) should collapse
    def low_freq_energy(x):
        return np.abs(np.fft.rfft(x[0]))[:3].sum()

    assert low_freq_energy(bp) < 0.1 * low_freq_energy(sig)


# --- normalization ----------------------------------------------------------
def test_zscore_per_lead():
    sig = np.random.default_rng(1).standard_normal((12, T)).astype(np.float32) * 5 + 3
    z = zscore(sig)
    assert np.allclose(z.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(z.std(axis=-1), 1, atol=1e-3)


# --- Pan-Tompkins -----------------------------------------------------------
def test_pan_tompkins_finds_expected_beats():
    sig, peaks = _synthetic_ecg(bpm=60)
    det = pan_tompkins_rpeaks(sig[1], FS)  # lead II
    assert abs(len(det) - len(peaks)) <= 1  # ~10 beats over 10 s
    # each detection is close to a true peak
    for d in det:
        assert np.min(np.abs(peaks - d)) <= 8


def test_pan_tompkins_flat_signal_no_peaks():
    assert pan_tompkins_rpeaks(np.zeros(T), FS).size == 0


def test_heart_rate_from_peaks():
    peaks = np.arange(50, T, 100)  # exactly 60 bpm at 100 Hz
    assert heart_rate_bpm(peaks, FS) == pytest.approx(60.0, abs=0.5)
    assert np.isnan(heart_rate_bpm(np.array([5]), FS))


# --- segmentation -----------------------------------------------------------
def test_segment_beats_shape_and_edge_drop():
    sig, _ = _synthetic_ecg(bpm=60)
    rp = pan_tompkins_rpeaks(sig[1], FS)
    beats = segment_beats(sig, rp, FS, pre_s=0.25, post_s=0.45)
    assert beats.ndim == 3 and beats.shape[1:] == (12, 70)
    assert beats.shape[0] <= len(rp)  # edge beats dropped


# --- end-to-end pipeline ----------------------------------------------------
def test_preprocess_returns_clean_and_rpeaks():
    sig, _ = _synthetic_ecg(bpm=60)
    clean, rp = preprocess(sig, fs_in=FS)
    assert clean.shape == (12, T) and clean.dtype == np.float32
    assert np.allclose(clean.mean(axis=-1), 0, atol=1e-5)
    assert len(rp) >= 8


def test_preprocessor_callable_matches_shape():
    sig, _ = _synthetic_ecg()
    out = Preprocessor(fs_in=FS, fs_out=FS)(sig)
    assert out.shape == (12, T) and out.dtype == np.float32
