"""Fixed-size ring buffer for the rolling analysis window.

A live monitor streams forever, so the buffer must be O(window) in memory and O(chunk)
per push — no reallocating or re-concatenating a growing array on every sample. This is a
preallocated ``(n_leads, capacity)`` array with a write cursor that wraps.

The only subtlety is that :meth:`RingBuffer.window` must return samples in *chronological*
order regardless of where the cursor happens to sit, which means stitching the two halves
when a read spans the wrap point.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Rolling multi-lead sample buffer holding the most recent ``capacity`` samples."""

    def __init__(self, n_leads: int, capacity: int, dtype=np.float32):
        if n_leads <= 0 or capacity <= 0:
            raise ValueError("n_leads and capacity must be positive")
        self.n_leads = int(n_leads)
        self.capacity = int(capacity)
        self._buf = np.zeros((self.n_leads, self.capacity), dtype=dtype)
        self._cursor = 0        # next write position
        self._written = 0       # total samples ever written (monotonic)

    @property
    def total_written(self) -> int:
        """Every sample ever pushed — the stream clock, in samples."""
        return self._written

    @property
    def filled(self) -> int:
        """How many valid samples the buffer currently holds (caps at ``capacity``)."""
        return min(self._written, self.capacity)

    def ready(self, n: int | None = None) -> bool:
        """Whether at least ``n`` samples (default: a full buffer) are available."""
        return self.filled >= (self.capacity if n is None else n)

    def push(self, chunk: np.ndarray) -> None:
        """Append ``(n_leads, k)`` samples, overwriting the oldest once full.

        A chunk longer than the capacity keeps only its final ``capacity`` samples — the
        buffer's whole contract is "the most recent window", so silently dropping the
        older part is correct rather than an error.
        """
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_leads:
            raise ValueError(
                f"expected a ({self.n_leads}, k) chunk, got {tuple(chunk.shape)}"
            )
        k = chunk.shape[1]
        if k == 0:
            return
        if k >= self.capacity:                     # keep only the newest capacity samples
            self._buf[:] = chunk[:, -self.capacity:]
            self._cursor = 0
            self._written += k
            return

        end = self._cursor + k
        if end <= self.capacity:
            self._buf[:, self._cursor:end] = chunk
        else:                                       # wraps: split across the boundary
            first = self.capacity - self._cursor
            self._buf[:, self._cursor:] = chunk[:, :first]
            self._buf[:, :k - first] = chunk[:, first:]
        self._cursor = end % self.capacity
        self._written += k

    def window(self, n: int | None = None) -> np.ndarray:
        """The most recent ``n`` samples in chronological order, ``(n_leads, n)``.

        Raises if fewer than ``n`` samples have been written — callers should gate on
        :meth:`ready` rather than analyzing a partially-warm buffer, since a
        zero-padded window is not a real recording.
        """
        n = self.capacity if n is None else int(n)
        if n <= 0 or n > self.capacity:
            raise ValueError(f"n must be in 1..{self.capacity}, got {n}")
        if self.filled < n:
            raise ValueError(f"buffer holds {self.filled} samples, need {n}")
        start = (self._cursor - n) % self.capacity
        if start + n <= self.capacity:
            return self._buf[:, start:start + n].copy()
        first = self.capacity - start
        return np.concatenate([self._buf[:, start:], self._buf[:, :n - first]], axis=1)

    def clear(self) -> None:
        """Reset to empty (keeps the allocation)."""
        self._buf.fill(0)
        self._cursor = 0
        self._written = 0
