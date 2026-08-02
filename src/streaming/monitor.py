"""Streaming analyzer: rolling-window detection with temporal persistence.

Batch APEX makes one decision per recording. A monitor makes a new decision every hop —
which changes the error profile completely. Phase 13 measured **5.09 spurious labels per
record** at the shipped 0.5 threshold; at a 1 s hop that is a fresh handful of spurious
labels *every second*, so a naive live panel flickers so badly it is unusable. The batch
metrics do not transfer.

The fix here is **temporal persistence (N-of-M)**: a finding is only *confirmed* once it
has appeared in at least ``confirm_k`` of the last ``confirm_m`` windows, and is only
*released* after it has been absent from all of the last ``confirm_m``. Two different bars
for entering and leaving is deliberate hysteresis — a single symmetric threshold makes
findings flap on and off at the boundary.

This trades detection **latency** for stability: a true finding needs ``confirm_k`` hops
before it is announced. That cost is measured rather than assumed — see
``docs/streaming/report.md`` for the suppression/latency trade-off across the test split.

The window itself stays at 10 s to match how the detector was trained. The architecture
(global average pooling) would accept shorter windows, but that is out of distribution and
unvalidated, so it is not the default.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.config import CFG
from src.serving.severity import URGENT_CODES


@dataclass
class FindingState:
    """A code's current standing in the stream."""

    code: str
    confidence: float                 # most recent window probability
    hits: int                         # windows in the persistence memory that fired
    window_of: int                    # size of that memory
    confirmed: bool
    first_seen_s: float               # stream time it first cleared threshold
    confirmed_at_s: float | None      # stream time it was confirmed (None if pending)
    last_seen_s: float

    @property
    def urgent(self) -> bool:
        return self.code in URGENT_CODES

    def releasing(self, confirm_k: int) -> bool:
        """Confirmed, but no longer holding — on its way out under hysteresis.

        Worth surfacing distinctly: a confirmed finding stays confirmed until it is absent
        from the *whole* memory, so between "stopped firing" and "released" it would
        otherwise render as a normal confirmed finding with a collapsed confidence, which
        reads as a bug to anyone watching the panel.
        """
        return self.confirmed and self.hits < confirm_k


@dataclass
class StreamEvent:
    """A transition worth telling the user about."""

    kind: str          # "onset" (confirmed) | "offset" (released)
    code: str
    t_s: float
    confidence: float = 0.0

    @property
    def urgent(self) -> bool:
        return self.code in URGENT_CODES


@dataclass
class StreamUpdate:
    """One analysis tick: what the findings panel should show right now."""

    t_s: float                              # stream time at the window's trailing edge
    confirmed: list[FindingState] = field(default_factory=list)
    pending: list[FindingState] = field(default_factory=list)   # seen, not yet persistent
    events: list[StreamEvent] = field(default_factory=list)
    latency_ms: float = 0.0                 # inference cost for this window
    window_s: float = 10.0
    # Codes over threshold in *this* window alone — i.e. what an undebounced panel would
    # show. Kept so the persistence layer's effect can be measured, not just asserted.
    fired: list[str] = field(default_factory=list)

    confirm_k: int = 3                      # so consumers can render the releasing state

    @property
    def urgent_active(self) -> list[str]:
        return sorted(f.code for f in self.confirmed if f.urgent)

    @property
    def alarm(self) -> bool:
        return bool(self.urgent_active)


class StreamMonitor:
    """Consume stream chunks, emit an update once per hop.

    Typical use::

        mon = StreamMonitor()
        for chunk in ReplaySource(signal):
            update = mon.push(chunk.samples)
            if update:
                render(update)

    ``push`` returns ``None`` while the buffer is still filling or between hops, so the
    caller can feed it arbitrarily-sized chunks without worrying about alignment.
    """

    def __init__(
        self,
        fs: int = CFG.sampling_rate,
        window_s: float = 10.0,
        hop_s: float = 1.0,
        n_leads: int = 12,
        threshold: float = CFG.review_threshold,
        confirm_k: int = 3,
        confirm_m: int = 5,
        checkpoint=None,
        device: str = "cpu",
        preprocess: bool = True,
    ):
        if not 1 <= confirm_k <= confirm_m:
            raise ValueError("need 1 <= confirm_k <= confirm_m")
        from src.streaming.buffer import RingBuffer

        self.fs = int(fs)
        self.window_samples = int(round(window_s * self.fs))
        self.hop_samples = max(1, int(round(hop_s * self.fs)))
        self.window_s = float(window_s)
        self.threshold = float(threshold)
        self.confirm_k = int(confirm_k)
        self.confirm_m = int(confirm_m)
        self.preprocess = preprocess
        self._checkpoint, self._device = checkpoint, device
        self._buf = RingBuffer(n_leads, self.window_samples)
        self._next_hop = self.window_samples          # first analysis at a full window
        self._history: dict[str, deque] = {}
        self._state: dict[str, FindingState] = {}
        self._model = self._label_space = None

    # --- model plumbing ------------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            from src.serving.model_cache import get_detector

            self._model, self._label_space, _ = get_detector(self._checkpoint,
                                                             device=self._device)
        return self._model, self._label_space

    def _classify(self, window: np.ndarray) -> np.ndarray:
        """Window -> per-code probabilities."""
        import torch

        from src.preprocessing.pipeline import preprocess as prep

        model, _ = self._ensure_model()
        x = prep(window, fs_in=self.fs, fs_out=self.fs)[0] if self.preprocess else window
        with torch.no_grad():
            logits = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
                           .unsqueeze(0).to(self._device))
        return torch.sigmoid(logits)[0].cpu().numpy()

    # --- persistence ---------------------------------------------------------
    def _update_states(self, probs: np.ndarray,
                       t_s: float) -> tuple[list[StreamEvent], set[str]]:
        _, label_space = self._ensure_model()
        fired = {label_space[j] for j in np.flatnonzero(probs >= self.threshold)}
        events: list[StreamEvent] = []

        # every code we are tracking, plus anything that just fired
        tracked = set(self._history) | fired
        for code in tracked:
            hist = self._history.setdefault(code, deque(maxlen=self.confirm_m))
            hist.append(code in fired)
            hits = sum(hist)
            conf = float(probs[label_space.index(code)])
            st = self._state.get(code)

            if st is None:
                if code not in fired:
                    continue
                st = FindingState(code=code, confidence=conf, hits=hits,
                                  window_of=len(hist), confirmed=False,
                                  first_seen_s=t_s, confirmed_at_s=None, last_seen_s=t_s)
                self._state[code] = st
            else:
                st.confidence, st.hits, st.window_of = conf, hits, len(hist)
                if code in fired:
                    st.last_seen_s = t_s

            if not st.confirmed and hits >= self.confirm_k:
                st.confirmed = True
                st.confirmed_at_s = t_s
                events.append(StreamEvent("onset", code, t_s, conf))
            elif st.confirmed and hits == 0:
                # released only once absent from the whole memory (hysteresis)
                events.append(StreamEvent("offset", code, t_s, conf))
                self._state.pop(code, None)
                self._history.pop(code, None)
            elif not st.confirmed and hits == 0:
                self._state.pop(code, None)
                self._history.pop(code, None)
        return events, fired

    # --- public API ----------------------------------------------------------
    def push(self, chunk: np.ndarray) -> StreamUpdate | None:
        """Feed samples; return an update when this chunk completes a hop."""
        self._buf.push(chunk)
        if self._buf.total_written < self._next_hop:
            return None
        # Skip any hops the chunk overshot — a late/large chunk must not queue up stale
        # analyses; a monitor always reports on the newest data available.
        while self._buf.total_written >= self._next_hop + self.hop_samples:
            self._next_hop += self.hop_samples
        self._next_hop += self.hop_samples

        t_s = self._buf.total_written / self.fs
        t0 = time.perf_counter()
        probs = self._classify(self._buf.window(self.window_samples))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        events, fired = self._update_states(probs, t_s)
        states = sorted(self._state.values(), key=lambda s: -s.confidence)
        return StreamUpdate(
            t_s=t_s,
            confirmed=[s for s in states if s.confirmed],
            pending=[s for s in states if not s.confirmed],
            events=events,
            latency_ms=latency_ms,
            window_s=self.window_s,
            fired=sorted(fired),
            confirm_k=self.confirm_k,
        )

    def reset(self) -> None:
        """Clear buffer and all finding state (e.g. new patient)."""
        self._buf.clear()
        self._history.clear()
        self._state.clear()
        self._next_hop = self.window_samples
