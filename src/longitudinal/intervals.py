"""Phase 22 — ECG interval and ST-level measurement.

To say "the PR interval has increased from 160 ms to 210 ms" the system has to *measure*
PR, and PTB-XL ships no interval annotations — only SCP statements. So this module
delineates the wave boundaries itself and derives the standard intervals plus per-lead ST
levels. Everything downstream (:mod:`src.longitudinal.delta`) consumes these numbers.

The pipeline, and why each step is the way it is:

**1. Baseline removal by median filtering, not the 0.5 Hz high-pass.** APEX's normal
preprocessing band-passes 0.5-40 Hz, which is right for the detector and wrong here: a
high-pass with a corner that high measurably distorts the ST segment, the exact quantity
this module reports. Instead the baseline is estimated with the classical two-stage median
filter (200 ms to swallow the QRS, then 600 ms to swallow P and T) and subtracted. That
removes wander without touching ST morphology.

**2. A median beat, not a single beat.** Beats are aligned on their R peaks and reduced
per-lead by the *median* across beats. Noise falls as roughly 1/sqrt(n) while the repeating
morphology survives, and the median (not the mean) discards the occasional ectopic or
motion-corrupted beat instead of smearing it across the template. Beats that correlate
poorly with a first-pass template, or whose RR is far from the local median, are dropped
before averaging.

**3. Interpolation to 500 Hz is legitimate here, not resolution laundering.** The records
are sampled at 100 Hz — 10 ms per sample, which is coarse next to a 40 ms change in PR.
But the signal has already been low-passed at 40 Hz, comfortably under the 50 Hz Nyquist
limit, so the continuous waveform is fully determined by its samples and cubic-spline
interpolation recovers intermediate values rather than inventing them. Combined with the
median beat's noise suppression, sub-sample boundary estimates are real. This does *not*
recover information a true 500 Hz recording would have (genuine 40-150 Hz QRS content is
gone for good, so absolute QRS duration is biased short) — see the honesty note below.

**4. Asymmetric onset/offset thresholds.** QRS onset is a sharp event; QRS offset is not,
and treating them symmetrically systematically truncates the wide QRS complexes that matter
most. See :data:`QRS_OFFSET_FRAC` for the measurement that forced this and the ceiling that
bounds it.

**5. Global boundaries across leads, the way ECG machines do it.** A wave does not start at
the same instant in every lead; the true onset is the *earliest* across leads and the true
offset the *latest*. Measuring in one lead systematically underestimates every duration.
Boundaries are therefore found on a spatial-magnitude signal built from the eight
independent leads (I, II, V1-V6 — the other four are exact linear combinations and would
only reweight the same information).

**Honesty about absolute accuracy.** These are 100 Hz recordings low-passed at 40 Hz, so
absolute durations carry a bias — QRS onset/offset in particular are softened by the
filter, biasing QRS duration. This module is nonetheless fit for its purpose, because a
comparison reports a *difference* between two recordings measured by the identical
algorithm, and a constant bias cancels exactly in the subtraction. What does *not* cancel
is the random error, which is why :mod:`src.longitudinal.delta` refuses to report any
change smaller than the noise floor measured on same-day pairs. Absolute values are still
validated against label-derived ground truth (1AVB implies PR > 200 ms, bundle branch block
implies QRS >= 120 ms) in ``scripts/longitudinal_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

from src.preprocessing.segmentation import pan_tompkins_rpeaks

# PTB-XL lead order.
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# The eight linearly independent leads; III/aVR/aVL/aVF are derivable from I and II.
INDEPENDENT = (0, 1, 6, 7, 8, 9, 10, 11)
LEAD_II = 1

UPSAMPLE = 5                 # 100 Hz -> 500 Hz effective (see module docstring, note 3)
ST_OFFSET_MS = 60.0          # ST level is read at J + 60 ms (standard practice)
MIN_BEATS = 3                # fewer clean beats than this -> no median beat, no measurement
# Physiologic plausibility windows. A value outside these is not a surprising patient, it
# is a failed delineation, and emitting it would put a fabricated number into a change
# report — the exact failure this phase is built to avoid. Rejected measurements become
# ``None`` (reported as "not compared") rather than being silently clipped into range.
PR_PLAUSIBLE_MS = (80.0, 400.0)
QT_PLAUSIBLE_MS = (200.0, 700.0)
# Rate-corrected, this catches the failures a raw QT window cannot: at 130 bpm a 197 ms QT
# is inside the raw window but implies a QTcF of 256 ms, which no heart produces.
QTC_PLAUSIBLE_MS = (280.0, 700.0)

# Velocity thresholds (as a fraction of the QRS velocity peak) for the wave boundaries.
# Asymmetric on purpose. QRS *onset* is a sharp event and 8% finds it cleanly, but the
# terminal forces at *offset* are slow and low-amplitude — most of all in right bundle
# branch block, whose late slurred S wave is exactly what defines it. A symmetric 8% cut
# the QRS short there: median QRS in CRBBB came out at 121 ms against a diagnosis that
# requires >=120 ms, with only 52% of RBBB records clearing the bar (AUROC 0.782). Halving
# the offset threshold to 4% raises that to 138 ms / 80% / AUROC 0.885 while leaving normal
# QRS at a textbook 85 ms. Going lower still keeps helping RBBB but starts running the
# offset into the T wave — at 2% the median LBBB "QRS" reaches an impossible 256 ms — so 4%
# is the last value where every diagnostic group stays physiologically plausible.
# Chosen on folds 1-8 only and confirmed on the held-out fold; see docs/longitudinal/report.md.
QRS_ONSET_FRAC = 0.08
QRS_OFFSET_FRAC = 0.04


@dataclass
class IntervalSet:
    """Global intervals (ms) and per-lead ST levels (mV) for one recording.

    ``None`` means *not measurable*, which is a real answer and never a zero. The common
    case is PR in atrial fibrillation: there is no P wave, so there is no PR interval, and
    emitting a number there would be a fabrication of exactly the kind Phase 7 exists to
    prevent.
    """

    heart_rate: float | None = None       # bpm, from the median RR
    rr: float | None = None               # ms
    pr: float | None = None               # ms, P onset -> QRS onset
    qrs: float | None = None              # ms, QRS onset -> QRS offset
    qt: float | None = None               # ms, QRS onset -> T offset
    qtc_bazett: float | None = None       # ms, QT / sqrt(RR in s)
    qtc_fridericia: float | None = None   # ms, QT / cbrt(RR in s)
    st_level: dict[str, float] = field(default_factory=dict)   # lead -> mV at J+60ms
    t_amplitude: dict[str, float] = field(default_factory=dict)  # lead -> mV at T peak

    n_beats: int = 0                      # beats surviving quality control
    n_beats_total: int = 0                # beats detected before quality control
    p_detected: bool = False
    quality: str = "ok"                   # "ok" | "noisy" | "unmeasurable"
    notes: list[str] = field(default_factory=list)

    @property
    def measurable(self) -> bool:
        return self.quality != "unmeasurable"

    def as_dict(self) -> dict:
        return {
            "heart_rate": self.heart_rate, "rr": self.rr, "pr": self.pr, "qrs": self.qrs,
            "qt": self.qt, "qtc_bazett": self.qtc_bazett, "qtc_fridericia": self.qtc_fridericia,
            "st_level": dict(self.st_level), "t_amplitude": dict(self.t_amplitude),
            "n_beats": self.n_beats, "n_beats_total": self.n_beats_total,
            "p_detected": self.p_detected, "quality": self.quality, "notes": list(self.notes),
        }


# --- conditioning ------------------------------------------------------------
def remove_baseline(signal: np.ndarray, fs: int) -> np.ndarray:
    """Two-stage median-filter baseline removal (200 ms then 600 ms), per lead.

    Preferred over a high-pass filter because it leaves the ST segment undistorted; a
    0.5 Hz high-pass — what the detector's preprocessing applies — visibly bends ST levels
    and this module's whole job is reading them.
    """
    w1 = max(3, int(round(0.200 * fs)) | 1)   # force odd
    w2 = max(3, int(round(0.600 * fs)) | 1)
    out = np.empty_like(signal, dtype=float)
    for i, lead in enumerate(np.asarray(signal, dtype=float)):
        base = median_filter(lead, size=w1, mode="nearest")
        base = median_filter(base, size=w2, mode="nearest")
        out[i] = lead - base
    return out


def _lowpass(signal: np.ndarray, fs: int, cutoff: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(cutoff, nyq * 0.95)
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def condition(signal: np.ndarray, fs: int) -> np.ndarray:
    """Baseline-removed, 40 Hz low-passed ``(12, T)`` signal — the measurement domain."""
    return _lowpass(remove_baseline(signal, fs), fs)


# --- median beat -------------------------------------------------------------
def _quality_filter(beats: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Boolean mask over beats: keep those that look like the dominant morphology.

    Two rejections, both aimed at ectopy and motion artefact:
    ``|RR - median(RR)| > 20%`` (a premature or escape beat), and correlation with a
    first-pass template below 0.8 (a beat whose shape does not belong).
    """
    keep = np.ones(len(beats), dtype=bool)
    if len(rr) == len(beats) and len(rr) > 2:
        med = np.median(rr)
        if med > 0:
            keep &= np.abs(rr - med) / med <= 0.20

    if keep.sum() >= 2:
        template = np.median(beats[keep], axis=0)         # (leads, win)
        flat_t = template.ravel()
        if np.std(flat_t) > 0:
            for i, b in enumerate(beats):
                flat_b = b.ravel()
                if np.std(flat_b) == 0:
                    keep[i] = False
                    continue
                r = float(np.corrcoef(flat_b, flat_t)[0, 1])
                if not np.isfinite(r) or r < 0.80:
                    keep[i] = False
    return keep


def median_beat(signal: np.ndarray, rpeaks: np.ndarray, fs: int,
                pre_s: float = 0.35, post_s: float = 0.55) -> tuple[np.ndarray | None, int, int, int]:
    """Median beat across quality-filtered beats.

    Returns ``(beat, r_index_within_beat, n_kept, n_total)``; ``beat`` is ``None`` when
    fewer than :data:`MIN_BEATS` usable beats remain. The window is asymmetric because the
    T wave extends much further after R than the P wave does before it.
    """
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    t = signal.shape[-1]
    usable = [r for r in rpeaks if r - pre >= 0 and r + post <= t]
    n_total = len(rpeaks)
    if len(usable) < MIN_BEATS:
        return None, pre, 0, n_total

    beats = np.stack([signal[:, r - pre:r + post] for r in usable])
    rr = np.diff(np.asarray(usable, dtype=float)) / fs * 1000.0
    rr_per_beat = np.concatenate([rr[:1], rr]) if len(rr) else np.array([])
    keep = _quality_filter(beats, rr_per_beat)
    if keep.sum() < MIN_BEATS:
        keep = np.ones(len(beats), dtype=bool)   # fall back rather than refuse outright

    return np.median(beats[keep], axis=0), pre, int(keep.sum()), n_total


def upsample_beat(beat: np.ndarray, factor: int = UPSAMPLE) -> np.ndarray:
    """Cubic-spline interpolation of ``(leads, n)`` by ``factor`` (see docstring note 3)."""
    n = beat.shape[-1]
    x = np.arange(n, dtype=float)
    xi = np.linspace(0.0, n - 1.0, (n - 1) * factor + 1)
    return CubicSpline(x, np.asarray(beat, dtype=float), axis=-1)(xi)


# --- delineation -------------------------------------------------------------
def _spatial_magnitude(beat: np.ndarray) -> np.ndarray:
    """Root-sum-square across the eight independent leads."""
    return np.sqrt(np.sum(beat[list(INDEPENDENT)] ** 2, axis=0))


def _spatial_velocity(beat: np.ndarray) -> np.ndarray:
    """Root-sum-square of the per-lead first derivative — the standard onset/offset cue."""
    d = np.diff(beat[list(INDEPENDENT)], axis=-1, prepend=beat[list(INDEPENDENT)][:, :1])
    return np.sqrt(np.sum(d ** 2, axis=0))


def _cross_below(curve: np.ndarray, start: int, thresh: float, direction: int,
                 limit: int) -> float:
    """Walk from ``start`` until ``curve`` drops below ``thresh``; linearly interpolate.

    Returns a fractional sample index so the boundary is not pinned to the sample grid.
    ``direction`` is -1 (leftward, for onsets) or +1 (rightward, for offsets).
    """
    i = start
    while 0 < i < len(curve) - 1 and abs(i - start) < limit:
        nxt = i + direction
        if curve[nxt] < thresh:
            span = curve[i] - curve[nxt]
            frac = (curve[i] - thresh) / span if span > 0 else 0.0
            return float(i + direction * frac)
        i = nxt
    return float(i)


@dataclass
class Fiducials:
    """Wave boundaries as fractional sample indices into the upsampled median beat."""

    p_onset: float | None
    p_peak: float | None
    p_offset: float | None
    qrs_onset: float
    r_peak: float
    qrs_offset: float
    t_peak: float | None
    t_offset: float | None


def delineate(beat_up: np.ndarray, fs_up: float, r_idx: float,
              rr_ms: float | None) -> Fiducials | None:
    """Locate P/QRS/T boundaries on an upsampled median beat.

    QRS bounds come from where spatial velocity falls under 8% of its QRS peak; the P wave
    is sought in the 300 ms before QRS onset and accepted only if it clears a
    noise-referenced amplitude bar; T offset uses the tangent (Lepeschkin) method — the
    steepest descent after the T peak, extrapolated to baseline — which is the standard
    manual technique and far more stable than chasing the point where T meets baseline.
    """
    ms = fs_up / 1000.0                       # samples per millisecond
    vel = _spatial_velocity(beat_up)
    mag = _spatial_magnitude(beat_up)
    n = len(vel)

    r = int(round(r_idx))
    search = max(1, int(40 * ms))
    lo, hi = max(0, r - search), min(n, r + search)
    r = lo + int(np.argmax(mag[lo:hi]))       # snap R to the spatial-magnitude peak

    qrs_win = max(1, int(80 * ms))
    qrs_vel_peak = float(np.max(vel[max(0, r - qrs_win):min(n, r + qrs_win)]))
    if qrs_vel_peak <= 0:
        return None
    qrs_onset = _cross_below(vel, r, QRS_ONSET_FRAC * qrs_vel_peak, -1, int(120 * ms))
    qrs_offset = _cross_below(vel, r, QRS_OFFSET_FRAC * qrs_vel_peak, +1, int(200 * ms))

    # --- P wave: search the 300 ms before QRS onset -------------------------
    p_hi = int(qrs_onset - 20 * ms)
    p_lo = max(0, int(qrs_onset - 300 * ms))
    p_onset = p_peak = p_offset = None
    if p_hi - p_lo > int(20 * ms):
        seg = mag[p_lo:p_hi]
        # Take the most *prominent local maximum*, not the largest sample. The QRS upslope
        # bleeds into the end of this window and can out-amplitude a real P wave by a hair
        # (record 2: a 0.1702 mV ramp 22 ms before QRS onset beating the true 0.168 mV P
        # wave 120 ms out, yielding an impossible 42 ms PR). A P wave is a discrete bump
        # with a decline on both sides; a monotonic ramp into the window edge is not a
        # local maximum at all, so requiring one rejects it structurally.
        floor = float(np.percentile(seg, 20))
        min_amp = max(3.0 * floor, 0.02)                 # 20 uV absolute minimum
        peaks, props = find_peaks(seg, height=min_amp, prominence=max(0.5 * floor, 0.01))
        if len(peaks):
            rel = int(peaks[int(np.argmax(props["prominences"]))])
            cand = p_lo + rel
            p_vel_peak = float(np.max(vel[p_lo:p_hi])) or 1.0
            p_thresh = 0.15 * p_vel_peak
            p_onset = _cross_below(vel, cand, p_thresh, -1, int(120 * ms))
            p_offset = _cross_below(vel, cand, p_thresh, +1, int(120 * ms))
            p_peak = float(cand)
            pr_ms = (qrs_onset - p_onset) / ms
            if p_onset >= qrs_onset or not (PR_PLAUSIBLE_MS[0] <= pr_ms <= PR_PLAUSIBLE_MS[1]):
                p_onset = p_peak = p_offset = None       # implausible -> report nothing

    # --- T wave: tangent method ---------------------------------------------
    t_peak = t_offset = None
    limit = int(min(n - 1, qrs_offset + (0.6 * rr_ms * ms if rr_ms else 450 * ms)))
    t_lo = int(qrs_offset + 60 * ms)
    if limit - t_lo > int(40 * ms):
        # Search on lead-space magnitude, but take the tangent on the signed lead with the
        # largest T deflection — the tangent is only meaningful on a signed waveform.
        seg = mag[t_lo:limit]
        tp = t_lo + int(np.argmax(seg))
        t_peak = float(tp)
        lead = int(np.argmax(np.abs(beat_up[list(INDEPENDENT), tp])))
        trace = beat_up[INDEPENDENT[lead]]
        sign = np.sign(trace[tp]) or 1.0
        deriv = np.gradient(trace)
        d_lo, d_hi = tp, min(n - 1, limit)
        if d_hi - d_lo > 2:
            # steepest return toward baseline after the T peak
            k = d_lo + int(np.argmin(sign * deriv[d_lo:d_hi]))
            slope = float(deriv[k])
            if abs(slope) > 1e-9:
                # tangent at k extrapolated to zero (baseline is 0 after baseline removal)
                t_offset = float(k - trace[k] / slope)
                t_offset = float(np.clip(t_offset, tp, n - 1))

    return Fiducials(p_onset, p_peak, p_offset, float(qrs_onset), float(r),
                     float(qrs_offset), t_peak, t_offset)


# --- top level ---------------------------------------------------------------
def measure(signal: np.ndarray, fs: int) -> IntervalSet:
    """Measure intervals and ST levels on a raw ``(12, T)`` recording.

    ``signal`` is the raw record in mV (not the detector's z-scored tensor — normalization
    would destroy the ST amplitudes this reads).
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 2 or sig.shape[0] < 12:
        return IntervalSet(quality="unmeasurable", notes=[f"expected (12, T), got {sig.shape}"])

    clean = condition(sig, fs)
    rpeaks = pan_tompkins_rpeaks(clean[LEAD_II], fs)
    return measure_from_beats(clean, rpeaks, fs)


def split_half(signal: np.ndarray, fs: int) -> tuple[IntervalSet, IntervalSet]:
    """Measure the *same* recording twice, from disjoint alternating beats.

    Odd-numbered beats form one median beat, even-numbered beats the other, and each is
    delineated independently. Because both halves come from the same ten seconds of the
    same patient, the difference between them contains no disease progression, no change in
    electrode placement, and no cohort selection — only measurement error plus beat-to-beat
    physiological variation. That makes it the clean repeatability estimate that a
    between-session cohort cannot provide (see :mod:`src.longitudinal.delta`).

    It is a *lower* bound on true test-retest error, since it excludes everything that
    varies between recording sessions; the phase brackets it from above with a
    stable-label long-gap cohort rather than presenting it as the whole story.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.ndim != 2 or sig.shape[0] < 12:
        bad = IntervalSet(quality="unmeasurable", notes=["bad shape"])
        return bad, bad
    clean = condition(sig, fs)
    rpeaks = pan_tompkins_rpeaks(clean[LEAD_II], fs)
    return (measure_from_beats(clean, rpeaks[0::2], fs),
            measure_from_beats(clean, rpeaks[1::2], fs))


def measure_from_beats(clean: np.ndarray, rpeaks: np.ndarray, fs: int) -> IntervalSet:
    """Core measurement, given an already-conditioned signal and a chosen set of R-peaks.

    Split out from :func:`measure` so :func:`split_half` can re-measure the same recording
    from disjoint beat subsets without re-conditioning or re-detecting.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return IntervalSet(quality="unmeasurable", n_beats_total=len(rpeaks),
                           notes=["fewer than 2 R-peaks detected"])

    rr_ms = float(np.median(np.diff(rpeaks)) / fs * 1000.0)
    out = IntervalSet(rr=round(rr_ms, 1), heart_rate=round(60000.0 / rr_ms, 1))

    beat, r_in_beat, n_kept, n_total = median_beat(clean, rpeaks, fs)
    out.n_beats, out.n_beats_total = n_kept, n_total
    if beat is None:
        out.quality = "unmeasurable"
        out.notes.append(f"only {n_total} beat(s) fully inside the strip; need {MIN_BEATS}")
        return out
    if n_kept < MIN_BEATS + 1:
        out.quality = "noisy"
        out.notes.append(f"median beat built from only {n_kept} beats")

    beat_up = upsample_beat(beat)
    fs_up = fs * UPSAMPLE
    fid = delineate(beat_up, fs_up, r_in_beat * UPSAMPLE, rr_ms)
    if fid is None:
        out.quality = "unmeasurable"
        out.notes.append("QRS delineation failed (no spatial velocity peak)")
        return out

    per_ms = fs_up / 1000.0
    out.qrs = round((fid.qrs_offset - fid.qrs_onset) / per_ms, 1)
    if fid.p_onset is not None:
        out.pr = round((fid.qrs_onset - fid.p_onset) / per_ms, 1)
        out.p_detected = True
    else:
        out.notes.append("no P wave detected — PR not reported")
    if fid.t_offset is not None:
        qt = (fid.t_offset - fid.qrs_onset) / per_ms
        rr_s = rr_ms / 1000.0
        qtcf = qt / np.cbrt(rr_s)
        if not (QT_PLAUSIBLE_MS[0] <= qt <= QT_PLAUSIBLE_MS[1]):
            out.notes.append(f"QT {qt:.0f} ms outside physiologic range — not reported")
        elif not (QTC_PLAUSIBLE_MS[0] <= qtcf <= QTC_PLAUSIBLE_MS[1]):
            out.notes.append(f"QTc {qtcf:.0f} ms outside physiologic range — QT not reported")
        else:
            out.qt = round(qt, 1)
            out.qtc_bazett = round(qt / np.sqrt(rr_s), 1)
            out.qtc_fridericia = round(qtcf, 1)
    else:
        out.notes.append("T offset not resolvable — QT not reported")

    # --- per-lead ST level and T amplitude ----------------------------------
    # Isoelectric reference is the PQ segment (20 ms ending at QRS onset); with the
    # baseline already removed this is a small correction, but it is the clinical
    # convention and it absorbs any residual per-lead offset.
    pq_hi = int(fid.qrs_onset)
    pq_lo = max(0, int(fid.qrs_onset - 20 * per_ms))
    j_plus = int(fid.qrs_offset + ST_OFFSET_MS * per_ms)
    for idx in INDEPENDENT:
        name = LEAD_NAMES[idx]
        iso = float(np.mean(beat_up[idx, pq_lo:pq_hi])) if pq_hi > pq_lo else 0.0
        if 0 <= j_plus < beat_up.shape[-1]:
            out.st_level[name] = round(float(beat_up[idx, j_plus]) - iso, 4)
        if fid.t_peak is not None:
            out.t_amplitude[name] = round(float(beat_up[idx, int(fid.t_peak)]) - iso, 4)

    return out
