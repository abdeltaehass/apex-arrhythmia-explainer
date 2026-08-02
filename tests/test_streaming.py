"""Phase 16 — tests for the streaming buffer, replay source, and persistence logic.

Data-independent: synthetic signals, and a stubbed classifier so the persistence state
machine is tested without torch or a checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.streaming.buffer import RingBuffer
from src.streaming.monitor import StreamMonitor
from src.streaming.source import ReplaySource, build_playlist, crossfade


# --- ring buffer -------------------------------------------------------------
def test_buffer_window_is_chronological_across_wrap():
    b = RingBuffer(2, 5)
    b.push(np.arange(8, dtype=np.float32).reshape(1, -1).repeat(2, axis=0))
    # capacity 5, wrote 8 -> holds the last five samples, oldest first
    assert b.window(5)[0].tolist() == [3, 4, 5, 6, 7]
    assert b.filled == 5 and b.total_written == 8


def test_buffer_partial_reads_take_the_newest():
    b = RingBuffer(1, 6)
    b.push(np.arange(6, dtype=np.float32).reshape(1, -1))
    assert b.window(3)[0].tolist() == [3, 4, 5]


def test_buffer_ready_and_filled():
    b = RingBuffer(1, 4)
    assert not b.ready() and b.filled == 0
    b.push(np.zeros((1, 3), dtype=np.float32))
    assert not b.ready() and b.ready(3) and b.filled == 3
    b.push(np.zeros((1, 1), dtype=np.float32))
    assert b.ready()


def test_buffer_oversized_chunk_keeps_newest():
    b = RingBuffer(1, 3)
    b.push(np.arange(10, dtype=np.float32).reshape(1, -1))
    assert b.window(3)[0].tolist() == [7, 8, 9]


def test_buffer_rejects_bad_shapes_and_reads():
    b = RingBuffer(2, 4)
    with pytest.raises(ValueError):
        b.push(np.zeros((3, 2)))              # wrong lead count
    with pytest.raises(ValueError):
        b.window(2)                            # not enough samples yet
    b.push(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        b.window(99)                           # larger than capacity


def test_buffer_clear_resets():
    b = RingBuffer(1, 3)
    b.push(np.ones((1, 3), dtype=np.float32))
    b.clear()
    assert b.filled == 0 and b.total_written == 0


def test_buffer_push_empty_is_noop():
    b = RingBuffer(1, 3)
    b.push(np.zeros((1, 0), dtype=np.float32))
    assert b.total_written == 0


# --- source ------------------------------------------------------------------
def test_crossfade_is_continuous_and_right_length():
    a = np.ones((1, 20), dtype=np.float32)
    b = np.zeros((1, 20), dtype=np.float32)
    out = crossfade(a, b, 10)
    assert out.shape == (1, 30)
    # no step: consecutive differences stay small through the blend
    assert np.abs(np.diff(out[0])).max() < 0.35


def test_build_playlist_lengths_and_validation():
    a = np.ones((12, 100), dtype=np.float32)
    b = np.zeros((12, 100), dtype=np.float32)
    out = build_playlist([a, b], fs=100, crossfade_s=0.1)   # 10-sample fade
    assert out.shape == (12, 190)
    with pytest.raises(ValueError):
        build_playlist([])
    with pytest.raises(ValueError):
        build_playlist([a, np.zeros((3, 10), dtype=np.float32)])


def test_replay_source_emits_expected_duration_without_sleeping():
    sig = np.zeros((12, 500), dtype=np.float32)
    src = ReplaySource(sig, fs=100, chunk_s=0.5, speed=0, loop=True, max_duration_s=3.0)
    chunks = list(src)
    assert sum(c.samples.shape[1] for c in chunks) == 300      # 3 s at 100 Hz
    assert chunks[0].t_start_s == 0.0
    assert all(c.samples.shape[0] == 12 for c in chunks)


def test_replay_source_no_loop_stops_at_end():
    sig = np.zeros((12, 250), dtype=np.float32)
    total = sum(c.samples.shape[1]
                for c in ReplaySource(sig, fs=100, chunk_s=1.0, speed=0, loop=False))
    assert total == 250


def test_replay_source_rejects_1d_signal():
    with pytest.raises(ValueError):
        ReplaySource(np.zeros(100, dtype=np.float32))


# --- persistence state machine (stubbed classifier) --------------------------
class _StubMonitor(StreamMonitor):
    """StreamMonitor with the model replaced by a scripted probability sequence."""

    def __init__(self, script, labels=("A", "B"), **kw):
        super().__init__(preprocess=False, **kw)
        self._script = list(script)
        self._labels = list(labels)
        self._i = 0

    def _ensure_model(self):
        return None, self._labels

    def _classify(self, window):
        row = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return np.array([row.get(c, 0.0) for c in self._labels], dtype=np.float32)


def _drive(mon, n_analyses):
    """Feed zeros until `n_analyses` windows have been analyzed; return every update.

    Filling the buffer is itself the first analysis, so that update is captured too.
    """
    updates = []
    u = mon.push(np.zeros((12, mon.window_samples), dtype=np.float32))
    if u:
        updates.append(u)
    for _ in range(n_analyses - 1):
        u = mon.push(np.zeros((12, mon.hop_samples), dtype=np.float32))
        if u:
            updates.append(u)
    return updates


def test_persistence_requires_k_of_m_before_confirming():
    # A fires every window; with k=3 it must not confirm until the third hop
    mon = _StubMonitor([{"A": 0.9}] * 10, confirm_k=3, confirm_m=5)
    ups = _drive(mon, 4)
    confirmed_at = [i for i, u in enumerate(ups) if any(f.code == "A" for f in u.confirmed)]
    assert confirmed_at and confirmed_at[0] == 2          # 3rd analysis (0-indexed)
    assert ups[0].pending and ups[0].pending[0].code == "A"


def test_single_window_blip_never_confirms():
    # A fires once then stops — exactly the flicker persistence is meant to suppress
    script = [{"A": 0.9}] + [{}] * 8
    mon = _StubMonitor(script, confirm_k=3, confirm_m=5)
    ups = _drive(mon, 6)
    assert all(not u.confirmed for u in ups)
    assert not any(e.kind == "onset" for u in ups for e in u.events)


def test_onset_and_offset_events_fire_once():
    script = [{"A": 0.9}] * 4 + [{}] * 6
    mon = _StubMonitor(script, confirm_k=2, confirm_m=3)
    ups = _drive(mon, 9)
    kinds = [(e.kind, e.code) for u in ups for e in u.events]
    assert kinds.count(("onset", "A")) == 1
    assert kinds.count(("offset", "A")) == 1
    assert kinds.index(("onset", "A")) < kinds.index(("offset", "A"))


def test_hysteresis_holds_through_a_gap():
    # confirmed, then one missing window: must stay confirmed (not flap)
    script = [{"A": 0.9}] * 3 + [{}] + [{"A": 0.9}] * 3
    mon = _StubMonitor(script, confirm_k=2, confirm_m=4)
    ups = _drive(mon, 7)
    after_gap = ups[3]
    assert any(f.code == "A" for f in after_gap.confirmed)
    assert not any(e.kind == "offset" for u in ups for e in u.events)


def test_releasing_flag_marks_a_fading_finding():
    # 3 hits then silence: hits decay out of the m=5 memory until they fall below k=3,
    # at which point the finding is still confirmed (hysteresis) but visibly fading
    script = [{"A": 0.9}] * 3 + [{}] * 4
    mon = _StubMonitor(script, confirm_k=3, confirm_m=5)
    ups = _drive(mon, 6)
    last = ups[-1]
    fading = [f for f in last.confirmed if f.releasing(last.confirm_k)]
    assert fading and fading[0].code == "A"


def test_fired_reports_the_raw_per_window_set():
    mon = _StubMonitor([{"A": 0.9, "B": 0.6}], confirm_k=3, confirm_m=5)
    ups = _drive(mon, 1)
    assert ups[0].fired == ["A", "B"]        # raw, before any persistence


def test_urgent_code_raises_alarm():
    mon = _StubMonitor([{"STE_": 0.9}] * 5, labels=("STE_", "A"), confirm_k=2, confirm_m=3)
    ups = _drive(mon, 3)
    assert ups[-1].alarm and ups[-1].urgent_active == ["STE_"]


def test_no_update_before_the_window_is_full():
    mon = _StubMonitor([{"A": 0.9}] * 5, confirm_k=1, confirm_m=3)
    assert mon.push(np.zeros((12, 100), dtype=np.float32)) is None     # 1 s of a 10 s window


def test_oversized_chunk_does_not_queue_stale_hops():
    """A late/large chunk must analyze once on the newest data, not replay every hop."""
    mon = _StubMonitor([{"A": 0.9}] * 20, confirm_k=1, confirm_m=3)
    mon.push(np.zeros((12, mon.window_samples), dtype=np.float32))
    u = mon.push(np.zeros((12, mon.hop_samples * 5), dtype=np.float32))
    assert u is not None
    # the stream clock jumped 5 hops but only one analysis ran
    assert mon._i <= 2


def test_reset_clears_state():
    mon = _StubMonitor([{"A": 0.9}] * 6, confirm_k=1, confirm_m=3)
    _drive(mon, 2)
    mon.reset()
    assert not mon._state and mon._buf.filled == 0


def test_invalid_confirm_params_rejected():
    with pytest.raises(ValueError):
        StreamMonitor(confirm_k=4, confirm_m=3)
