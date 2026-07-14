"""The APEX preprocessing chain, in the order Phase 2 specifies:

    1. resample to 100 Hz            (src.preprocessing.resample)
    2. band-pass 0.5-40 Hz           (src.preprocessing.filters)  -- baseline + EMG
    3. R-peak detection (Pan-Tompkins) for beat segmentation / grounding
    4. per-lead z-score normalization (src.preprocessing.normalization)

The detector is trained on the whole 10 s strip, so the tensor handed to the model is
the resampled/filtered/normalized ``(12, T)`` signal (steps 1-2-4). R-peaks (step 3)
are computed from the *filtered, pre-normalization* signal and returned alongside —
they drive beat windowing and the grounding overlay, not the input tensor itself.

All arrays are shaped ``(leads, time)`` with the PTB-XL lead order
``I, II, III, aVR, aVL, aVF, V1..V6`` (lead II, index 1, is used for R-peak detection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_SAMPLING_RATE
from src.preprocessing.filters import bandpass
from src.preprocessing.normalization import zscore
from src.preprocessing.resample import resample_signal
from src.preprocessing.segmentation import pan_tompkins_rpeaks

LEAD_II = 1  # index of lead II in the PTB-XL lead order


@dataclass
class Preprocessor:
    """Callable preprocessing chain; use as the ``transform`` of ``PTBXLDataset``.

    Produces the model input tensor: resample -> band-pass -> per-lead z-score.
    """

    fs_in: int = DEFAULT_SAMPLING_RATE
    fs_out: int = DEFAULT_SAMPLING_RATE
    low: float = 0.5
    high: float = 40.0
    normalize: bool = True

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        sig = resample_signal(signal, self.fs_in, self.fs_out)
        sig = bandpass(sig, self.fs_out, self.low, self.high)
        if self.normalize:
            sig = zscore(sig)
        return np.ascontiguousarray(sig, dtype=np.float32)


def filtered_signal(signal: np.ndarray, fs_in: int, fs_out: int = DEFAULT_SAMPLING_RATE,
                    low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Resample + band-pass only (no normalization) — the domain for R-peak detection."""
    sig = resample_signal(signal, fs_in, fs_out)
    return bandpass(sig, fs_out, low, high)


def preprocess(signal: np.ndarray, fs_in: int, fs_out: int = DEFAULT_SAMPLING_RATE,
               detect_rpeaks: bool = True, rpeak_lead: int = LEAD_II
               ) -> tuple[np.ndarray, np.ndarray]:
    """Run the full chain on a raw ``(12, T)`` record.

    Returns ``(clean, rpeaks)`` where ``clean`` is the normalized model-input tensor
    and ``rpeaks`` are R-peak sample indices (in ``fs_out`` time) detected on lead II
    of the filtered-but-unnormalized signal. ``rpeaks`` is empty if ``detect_rpeaks``
    is False.
    """
    filt = filtered_signal(signal, fs_in, fs_out)
    rpeaks = (
        pan_tompkins_rpeaks(filt[rpeak_lead], fs_out)
        if detect_rpeaks else np.array([], dtype=int)
    )
    clean = zscore(filt)
    return np.ascontiguousarray(clean, dtype=np.float32), rpeaks
