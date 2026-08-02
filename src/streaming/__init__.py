"""Phase 16 — real-time streaming: turn APEX from batch analysis into live monitoring.

A wearable or bedside monitor does not hand you a tidy 10-second recording; it hands you
samples, forever. This package adapts the batch pipeline to that shape:

- :mod:`src.streaming.buffer` — a fixed-size ring buffer holding the rolling window.
- :mod:`src.streaming.source` — replay a PTB-XL record (or a playlist of them) at
  wall-clock pace, the way a monitor would deliver it.
- :mod:`src.streaming.monitor` — the streaming analyzer: re-classify the rolling window
  every hop, then apply **temporal persistence** so a one-window blip is not an alarm.

The persistence layer is the reason this is more than a demo. Phase 13 measured 5.09
spurious labels per record at the shipped threshold; in a stream that becomes a *new*
spurious decision every hop, so the naive per-window panel flickers constantly. Requiring
a finding to hold across several consecutive windows suppresses most of that at a small,
measurable cost in detection latency — see ``docs/streaming/report.md``.
"""

from src.streaming.buffer import RingBuffer
from src.streaming.monitor import FindingState, StreamEvent, StreamMonitor, StreamUpdate
from src.streaming.source import ReplaySource, StreamChunk, build_playlist, load_record_signal

__all__ = [
    "RingBuffer",
    "ReplaySource",
    "StreamChunk",
    "StreamMonitor",
    "StreamUpdate",
    "StreamEvent",
    "FindingState",
    "build_playlist",
    "load_record_signal",
]
