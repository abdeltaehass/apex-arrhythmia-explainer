"""Segmentation / windowing.

PTB-XL records are a fixed 10 s, so v1 uses the whole strip. This module holds
optional windowing (e.g. sliding windows for augmentation, or R-peak-centred beat
extraction) for later experiments.
"""

from __future__ import annotations

import numpy as np


def fixed_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or right-pad (leads, time) to exactly `target_len` samples."""
    t = signal.shape[-1]
    if t == target_len:
        return signal
    if t > target_len:
        start = (t - target_len) // 2
        return signal[:, start : start + target_len]
    pad = target_len - t
    return np.pad(signal, ((0, 0), (0, pad)), mode="constant")


def sliding_windows(signal: np.ndarray, win: int, stride: int) -> list[np.ndarray]:
    """Overlapping windows for augmentation. Returns a list of (leads, win)."""
    t = signal.shape[-1]
    return [signal[:, s : s + win] for s in range(0, max(1, t - win + 1), stride)]
