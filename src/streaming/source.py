"""Replay a recorded ECG as if it were arriving live.

PTB-XL records are 10 seconds long, which is exactly one analysis window — so a single
record played once yields one decision, not a stream. Two honest ways to get a continuous
signal out of finite recordings:

- **loop** a record (the default): the same rhythm continues indefinitely, which is what
  you want for "monitor a stable patient" demos;
- **playlist**: concatenate several records so the rhythm *changes* mid-stream (normal →
  atrial fibrillation), which is the clinically interesting case — does the monitor notice?

Both create joins that do not exist in a real continuous recording. A hard splice is a step
discontinuity, and a step is a broadband transient that the band-pass filter will ring on —
an artifact the model would (reasonably) react to. :func:`build_playlist` therefore applies
a short raised-cosine crossfade at each join and the report says so, rather than pretending
the seam is not there. Windows spanning a join are genuinely part-one-record,
part-another; that is a property of the simulation, not a bug being hidden.

Pacing is wall-clock by default so a demo runs in real time; ``speed=0`` disables sleeping
entirely so evaluation and tests run as fast as the CPU allows.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import CFG, PTBXL_DIR


@dataclass
class StreamChunk:
    """One delivery from the stream: samples plus where they sit on the stream clock."""

    samples: np.ndarray     # (n_leads, k)
    t_start_s: float        # stream time of the chunk's first sample
    fs: int
    source: str = ""        # which underlying record these samples came from

    @property
    def duration_s(self) -> float:
        return self.samples.shape[1] / self.fs


def load_record_signal(ecg_id: int, ptbxl_dir: Path = PTBXL_DIR,
                       fs: int = 100) -> tuple[np.ndarray, int]:
    """Load one PTB-XL record as a raw ``(12, T)`` float32 signal at ``fs``."""
    import wfdb

    from src.data.labels import load_database

    df = load_database(ptbxl_dir)
    col = "filename_lr" if fs == 100 else "filename_hr"
    sig, _ = wfdb.rdsamp(str(ptbxl_dir / df.loc[ecg_id, col]))
    return sig.T.astype(np.float32), fs


def _taper(n: int) -> np.ndarray:
    """Raised-cosine ramp from 0 to 1 over ``n`` samples."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    return (0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))).astype(np.float32)


def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Concatenate ``a`` and ``b`` with an ``n``-sample raised-cosine crossfade.

    Blends the tail of ``a`` into the head of ``b`` so the join is continuous instead of a
    step. The result is ``a.shape[1] + b.shape[1] - n`` samples long.
    """
    n = int(min(n, a.shape[1], b.shape[1]))
    if n <= 0:
        return np.concatenate([a, b], axis=1)
    ramp = _taper(n)
    blend = a[:, a.shape[1] - n:] * (1 - ramp) + b[:, :n] * ramp
    return np.concatenate([a[:, :a.shape[1] - n], blend, b[:, n:]], axis=1)


def build_playlist(signals: list[np.ndarray], fs: int = 100,
                   crossfade_s: float = 0.25) -> np.ndarray:
    """Concatenate signals into one continuous stream with crossfaded joins.

    Every signal must have the same lead count. See the module docstring on why the joins
    are tapered and why they are still a simulation artifact.
    """
    if not signals:
        raise ValueError("need at least one signal")
    leads = {s.shape[0] for s in signals}
    if len(leads) != 1:
        raise ValueError(f"all signals must have the same lead count, got {sorted(leads)}")
    n = int(round(crossfade_s * fs))
    out = signals[0].astype(np.float32)
    for s in signals[1:]:
        out = crossfade(out, s.astype(np.float32), n)
    return out


class ReplaySource:
    """Yield :class:`StreamChunk`s from a recorded signal, paced like a live monitor.

    ``chunk_s`` is the delivery granularity (a real device pushes small packets, not whole
    windows). ``speed`` multiplies real time: ``1.0`` is true real-time, ``4.0`` is a
    4x-fast demo, and ``0`` means "no sleeping at all" for evaluation. ``loop`` repeats the
    signal indefinitely, crossfading the wrap so the loop point is not a step.
    """

    def __init__(self, signal: np.ndarray, fs: int = CFG.sampling_rate,
                 chunk_s: float = 0.25, speed: float = 1.0, loop: bool = True,
                 max_duration_s: float | None = None, source: str = "",
                 loop_crossfade_s: float = 0.25):
        sig = np.asarray(signal, dtype=np.float32)
        if sig.ndim != 2:
            raise ValueError(f"expected a (leads, T) signal, got {tuple(sig.shape)}")
        self.fs = int(fs)
        self.chunk = max(1, int(round(chunk_s * self.fs)))
        self.speed = float(speed)
        self.loop = bool(loop)
        self.source = source
        self.max_samples = (None if max_duration_s is None
                            else int(round(max_duration_s * self.fs)))
        # Pre-crossfade the wrap so looping doesn't inject a step every pass.
        if self.loop:
            n = int(round(loop_crossfade_s * self.fs))
            self._sig = crossfade(sig, sig, n) if n > 0 else sig
        else:
            self._sig = sig

    @property
    def n_leads(self) -> int:
        return self._sig.shape[0]

    def __iter__(self) -> Iterator[StreamChunk]:
        total = self._sig.shape[1]
        pos, emitted = 0, 0
        t0 = time.monotonic()
        while True:
            if pos >= total:
                if not self.loop:
                    return
                pos = 0
            k = min(self.chunk, total - pos)
            if self.max_samples is not None:
                k = min(k, self.max_samples - emitted)
                if k <= 0:
                    return
            chunk = self._sig[:, pos:pos + k]
            t_start = emitted / self.fs
            if self.speed > 0:                      # pace against the wall clock
                target = t0 + t_start / self.speed
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield StreamChunk(samples=chunk, t_start_s=t_start, fs=self.fs,
                              source=self.source)
            pos += k
            emitted += k
