"""API-key authentication and a basic in-memory rate limiter.

Deliberately dependency-free (no Redis / slowapi): a single-process fixed-window
counter, keyed by API key when present else client IP. Good enough to protect a
single-node decision-support service and to demonstrate the mechanism; a multi-replica
deployment would swap the store for a shared one.

:class:`RateLimiter` is a plain class (framework-free, unit-tested directly). The two
FastAPI dependencies read :data:`src.serving.settings.SETTINGS`:

- :func:`require_api_key` — 401 if auth is enabled and the ``X-API-Key`` header is
  missing/unknown. When ``SETTINGS.api_keys`` is empty the service runs open (dev mode)
  and it is a no-op.
- :func:`rate_limit` — 429 once a client exceeds ``rate_limit`` requests within
  ``rate_window_s``.
"""

from __future__ import annotations

import threading
import time

from fastapi import Header, HTTPException, Request

from src.serving.settings import SETTINGS


class RateLimiter:
    """Fixed-window counter: ``limit`` requests per ``window_s`` per client key."""

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._windows: dict[str, tuple[int, float]] = {}  # key -> (count, window_start)
        self._lock = threading.Lock()

    def check(self, client_key: str) -> tuple[bool, int, float]:
        """Register a hit. Returns ``(allowed, remaining, retry_after_s)``."""
        now = time.monotonic()
        with self._lock:
            count, start = self._windows.get(client_key, (0, now))
            if now - start >= self.window_s:
                count, start = 0, now  # window elapsed -> reset
            count += 1
            self._windows[client_key] = (count, start)
            allowed = count <= self.limit
            remaining = max(0, self.limit - count)
            retry_after = 0.0 if allowed else (start + self.window_s - now)
            return allowed, remaining, max(0.0, retry_after)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


# Process-global limiter, configured from settings (tests mutate .limit / call .reset()).
LIMITER = RateLimiter(SETTINGS.rate_limit, SETTINGS.rate_window_s)


def _client_key(request: Request, api_key: str | None) -> str:
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency: validate the ``X-API-Key`` header when auth is enabled."""
    if not SETTINGS.auth_enabled:
        return None
    if x_api_key is None or x_api_key not in SETTINGS.api_keys:
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    return x_api_key


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> int:
    """FastAPI dependency: enforce the per-client rate limit, 429 when exceeded."""
    allowed, remaining, retry_after = LIMITER.check(_client_key(request, x_api_key))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1),
                     "X-RateLimit-Limit": str(LIMITER.limit)},
        )
    return remaining
