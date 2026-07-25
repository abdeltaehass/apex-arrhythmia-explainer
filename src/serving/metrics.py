"""In-process request metrics for the API's ``GET /metrics`` endpoint.

A thread-safe collector that records per-request latency and status, and reports
count, error count, uptime, throughput, and p50/p95/p99 latency since process startup.
Latencies are kept in a bounded ring buffer so memory stays flat under load; the
percentiles are therefore over the most recent ``maxlen`` requests, which is what you
want for a "current performance" readout anyway.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated ``q``-percentile (0..100) of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[lo] + frac * (sorted_values[lo + 1] - sorted_values[lo])


@dataclass
class MetricsSnapshot:
    request_count: int
    error_count: int
    uptime_s: float
    requests_per_s: float          # request_count / uptime
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    max_latency_ms: float
    window_size: int               # how many recent latencies the percentiles cover

    def to_dict(self) -> dict:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "uptime_s": round(self.uptime_s, 3),
            "requests_per_s": round(self.requests_per_s, 4),
            "latency_ms": {
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
                "mean": round(self.mean_latency_ms, 2),
                "max": round(self.max_latency_ms, 2),
            },
            "latency_window_size": self.window_size,
        }


class MetricsCollector:
    def __init__(self, maxlen: int = 2000):
        self._lat_ms: deque[float] = deque(maxlen=maxlen)
        self._count = 0
        self._errors = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()

    def record(self, latency_s: float, ok: bool = True) -> None:
        with self._lock:
            self._lat_ms.append(latency_s * 1000.0)
            self._count += 1
            if not ok:
                self._errors += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            lat = sorted(self._lat_ms)
            count, errors = self._count, self._errors
            uptime = max(time.monotonic() - self._start, 1e-9)
        return MetricsSnapshot(
            request_count=count,
            error_count=errors,
            uptime_s=uptime,
            requests_per_s=count / uptime,
            p50_latency_ms=percentile(lat, 50),
            p95_latency_ms=percentile(lat, 95),
            p99_latency_ms=percentile(lat, 99),
            mean_latency_ms=(sum(lat) / len(lat)) if lat else 0.0,
            max_latency_ms=lat[-1] if lat else 0.0,
            window_size=len(lat),
        )

    def reset(self) -> None:
        with self._lock:
            self._lat_ms.clear()
            self._count = 0
            self._errors = 0
            self._start = time.monotonic()


# Process-global collector the API middleware records into.
METRICS = MetricsCollector()
