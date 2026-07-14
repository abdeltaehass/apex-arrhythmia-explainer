"""Resampling to a common sampling rate.

PTB-XL ships every record at both 100 Hz and 500 Hz. We standardise on 100 Hz
(``src.config.DEFAULT_SAMPLING_RATE``): loading the 100 Hz file makes this a no-op,
but the function also handles 500 Hz (or any rate) so a single code path works for
downsampled 500 Hz data, external recordings, or digitized paper ECGs.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(signal: np.ndarray, fs_in: int, fs_out: int = 100) -> np.ndarray:
    """Resample (leads, time) from `fs_in` to `fs_out` Hz.

    Uses polyphase resampling (``resample_poly``), which applies an anti-aliasing
    FIR filter — important when downsampling 500 Hz -> 100 Hz to avoid folding
    high-frequency content back into the ECG band.
    """
    if fs_in == fs_out:
        return signal.astype(np.float32, copy=False)
    g = gcd(int(fs_in), int(fs_out))
    up, down = fs_out // g, fs_in // g
    out = resample_poly(signal, up, down, axis=-1)
    return out.astype(np.float32, copy=False)
