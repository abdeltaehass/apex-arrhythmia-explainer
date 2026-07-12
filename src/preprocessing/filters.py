"""Signal filtering for 12-lead ECG.

Standard clinical preprocessing: baseline-wander removal (high-pass ~0.5 Hz),
powerline notch (50/60 Hz), and optional low-pass anti-noise. Operates on
arrays shaped (leads, time).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
except ImportError:
    butter = iirnotch = sosfiltfilt = tf2sos = None  # type: ignore


def bandpass(signal: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass. `signal` is (leads, time)."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def notch(signal: np.ndarray, fs: int, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove powerline interference (set freq=60 for North America)."""
    b, a = iirnotch(freq, q, fs)
    return sosfiltfilt(tf2sos(b, a), signal, axis=-1)


def clean(signal: np.ndarray, fs: int, powerline: float = 50.0) -> np.ndarray:
    return notch(bandpass(signal, fs), fs, freq=powerline)
