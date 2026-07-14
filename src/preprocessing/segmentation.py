"""Segmentation / windowing.

PTB-XL records are a fixed 10 s. The detector consumes the whole strip, but we also
detect R-peaks (Pan-Tompkins) so we can (a) extract beat-centred windows and (b) show
where the heartbeats are in the grounding/sanity views. This module holds the QRS
detector plus fixed-length, sliding, and beat-centred windowing helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


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


# --- Pan-Tompkins QRS / R-peak detection -----------------------------------
# Classic Pan & Tompkins (1985): bandpass -> derivative -> square -> moving-window
# integration -> adaptive thresholding, then refine each detection to the R-peak on
# the bandpassed signal. Operates on a single lead (lead II by default).

def _qrs_bandpass(x: np.ndarray, fs: int, low: float = 5.0, high: float = 15.0) -> np.ndarray:
    """Band-pass that isolates QRS energy (~5-15 Hz)."""
    high = min(high, fs / 2 * 0.99)  # keep below Nyquist for low fs
    sos = butter(2, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def pan_tompkins_rpeaks(x: np.ndarray, fs: int, refractory_s: float = 0.20) -> np.ndarray:
    """R-peak sample indices in single-lead signal `x` via Pan-Tompkins.

    `refractory_s` is the physiological minimum RR interval (200 ms ~= 300 bpm max).
    Returns a sorted int array of R-peak indices (possibly empty for flat signals).
    """
    x = np.asarray(x, dtype=float)
    if x.size < fs // 2:  # too short to contain a beat
        return np.array([], dtype=int)

    filtered = _qrs_bandpass(x, fs)
    deriv = np.ediff1d(filtered, to_begin=0.0)          # slope
    squared = deriv ** 2                                 # nonlinear amplification
    win = max(1, int(round(0.150 * fs)))                 # 150 ms integration window
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # Candidate peaks in the integrated signal, spaced by the refractory period.
    min_dist = max(1, int(round(refractory_s * fs)))
    cand, _ = find_peaks(integrated, distance=min_dist)
    if cand.size == 0:
        return np.array([], dtype=int)

    # Pan-Tompkins adaptive thresholding on the integrated waveform. Seed the running
    # signal/noise peak estimates from the first 2 s (or the whole strip if shorter).
    init = integrated[: 2 * fs] if integrated.size >= 2 * fs else integrated
    spki = 0.25 * float(init.max())
    npki = 0.5 * float(init.mean())
    qrs: list[int] = []
    for p in cand:
        thr = npki + 0.25 * (spki - npki)
        if integrated[p] > thr:
            qrs.append(int(p))
            spki = 0.125 * integrated[p] + 0.875 * spki
        else:
            npki = 0.125 * integrated[p] + 0.875 * npki

    # Refine: the integration/derivative delays the peak, so snap each detection to the
    # largest local deflection on the QRS-bandpassed signal within +/-100 ms.
    rad = max(1, int(round(0.100 * fs)))
    rpeaks: list[int] = []
    for p in qrs:
        lo, hi = max(0, p - rad), min(len(filtered), p + rad)
        r = lo + int(np.argmax(np.abs(filtered[lo:hi])))
        if not rpeaks or (r - rpeaks[-1]) >= min_dist:
            rpeaks.append(r)
    return np.asarray(rpeaks, dtype=int)


def heart_rate_bpm(rpeaks: np.ndarray, fs: int) -> float:
    """Mean heart rate (bpm) from R-peak indices; NaN if <2 peaks."""
    if len(rpeaks) < 2:
        return float("nan")
    rr = np.diff(rpeaks) / fs           # seconds between beats
    return float(60.0 / np.mean(rr))


def segment_beats(
    signal: np.ndarray, rpeaks: np.ndarray, fs: int, pre_s: float = 0.25, post_s: float = 0.45
) -> np.ndarray:
    """Extract R-peak-centred beat windows from (leads, time).

    Returns (n_beats, leads, pre+post samples); beats whose window would fall off
    either end of the strip are dropped so every returned beat is full-length.
    """
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    t = signal.shape[-1]
    beats = [signal[:, r - pre : r + post] for r in rpeaks if r - pre >= 0 and r + post <= t]
    if not beats:
        return np.empty((0, signal.shape[0], pre + post), dtype=signal.dtype)
    return np.stack(beats)
