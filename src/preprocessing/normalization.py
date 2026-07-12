"""Amplitude normalization for ECG signals (per-lead)."""

from __future__ import annotations

import numpy as np


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-lead z-score. `signal` is (leads, time)."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / (std + eps)


def robust_scale(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-lead median/IQR scaling — less sensitive to spikes than z-score."""
    median = np.median(signal, axis=-1, keepdims=True)
    q75, q25 = np.percentile(signal, [75, 25], axis=-1, keepdims=True)
    iqr = q75 - q25
    return (signal - median) / (iqr + eps)
