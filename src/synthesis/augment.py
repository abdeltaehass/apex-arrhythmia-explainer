"""Phase 25 — classical signal-space augmentation.

The control that any generative method has to beat. It is cheap, it has no training
procedure to go wrong, and in small-data regimes it is frequently the thing that actually
works — so a synthetic-data result reported only against "no augmentation" tells you
almost nothing.

Every transform here preserves the *diagnostic* content of the tracing, which is a stronger
constraint than it sounds. Standard image augmentations have no valid analogue on an ECG:

- **No horizontal flip.** Time reversal turns a QRS complex into something no heart
  produces, and inverts the entire T-wave relationship.
- **No lead shuffling.** Lead identity *is* the localizing information — swapping V1 and V6
  would relabel an anteroseptal infarct as lateral.
- **No aggressive time scaling.** Stretching the time axis changes the heart rate, and rate
  is itself a label (``STACH``, ``SBRAD``). Warping is therefore small and local rather than
  a global resample.

What is left is the set of nuisance variation a real recording session actually contains:
gain differences between machines, baseline wander from respiration, electrode noise,
powerline interference, and the occasional bad lead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    """Per-transform probability and magnitude. Defaults are deliberately mild."""

    p_scale: float = 0.5
    scale_range: tuple[float, float] = (0.85, 1.15)     # recording gain
    p_baseline: float = 0.5
    baseline_amp: float = 0.10                           # mV of wander
    baseline_hz: tuple[float, float] = (0.05, 0.5)       # respiratory band
    p_noise: float = 0.5
    noise_sd: float = 0.02                               # mV of broadband noise
    p_powerline: float = 0.2
    powerline_amp: float = 0.03
    powerline_hz: float = 50.0
    p_warp: float = 0.3
    warp_strength: float = 0.05                          # +/-5% local time warp
    p_lead_dropout: float = 0.15
    max_dropped_leads: int = 1
    p_shift: float = 0.5
    max_shift_s: float = 0.25                            # circular time shift


def _baseline_wander(n: int, fs: int, cfg: AugmentConfig, rng) -> np.ndarray:
    t = np.arange(n) / fs
    freq = rng.uniform(*cfg.baseline_hz)
    phase = rng.uniform(0, 2 * np.pi)
    return cfg.baseline_amp * rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * freq * t + phase)


def _time_warp(x: np.ndarray, cfg: AugmentConfig, rng) -> np.ndarray:
    """Smooth, small, *local* warp of the time axis.

    Implemented as a cumulative random walk over a handful of knots, normalized back to the
    original duration, so overall heart rate is preserved while beat-to-beat spacing
    wobbles the way real RR intervals do.
    """
    n = x.shape[-1]
    knots = rng.normal(0.0, cfg.warp_strength, size=8)
    grid = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(knots)), knots)
    warped = np.cumsum(1.0 + grid)
    warped = (warped - warped[0]) / (warped[-1] - warped[0]) * (n - 1)
    idx = np.arange(n, dtype=float)
    return np.stack([np.interp(idx, warped, lead) for lead in x])


def augment(signal: np.ndarray, fs: int = 100, cfg: AugmentConfig | None = None,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply a random subset of the transforms to a ``(12, T)`` recording."""
    cfg = cfg or AugmentConfig()
    rng = rng or np.random.default_rng()
    x = np.asarray(signal, dtype=np.float32).copy()
    n = x.shape[-1]

    if rng.random() < cfg.p_scale:
        x *= rng.uniform(*cfg.scale_range)
    if rng.random() < cfg.p_warp:
        x = _time_warp(x, cfg, rng).astype(np.float32)
    if rng.random() < cfg.p_shift:
        # Circular, because the strip is a 10 s window of a continuous rhythm — where the
        # window starts carries no diagnostic meaning.
        x = np.roll(x, int(rng.integers(-int(cfg.max_shift_s * fs), int(cfg.max_shift_s * fs))),
                    axis=-1)
    if rng.random() < cfg.p_baseline:
        wander = _baseline_wander(n, fs, cfg, rng)
        x += wander[None, :] * rng.uniform(0.5, 1.0, size=(x.shape[0], 1))
    if rng.random() < cfg.p_powerline:
        t = np.arange(n) / fs
        x += cfg.powerline_amp * np.sin(2 * np.pi * cfg.powerline_hz * t
                                        + rng.uniform(0, 2 * np.pi))[None, :]
    if rng.random() < cfg.p_noise:
        x += rng.normal(0.0, cfg.noise_sd, size=x.shape)
    if rng.random() < cfg.p_lead_dropout:
        # A single failed electrode. More than one at a time is a rejected recording, not
        # an augmentation, and the Phase-8 input gate would refuse it anyway.
        for lead in rng.choice(x.shape[0], size=int(rng.integers(1, cfg.max_dropped_leads + 1)),
                               replace=False):
            x[lead] = 0.0
    return x.astype(np.float32)


def augment_batch(signals: np.ndarray, fs: int = 100, cfg: AugmentConfig | None = None,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Independently augment each recording in an ``(N, 12, T)`` batch."""
    rng = rng or np.random.default_rng()
    return np.stack([augment(s, fs, cfg, rng) for s in signals])
