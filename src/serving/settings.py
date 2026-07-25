"""Runtime settings for the API, read from the environment.

Kept out of `src/config.py` (which is the static, committed source of truth for paths /
targets) because these are *deployment* knobs — API keys, rate limits, device — that
vary per host and must never be hard-coded. A single mutable :data:`SETTINGS` instance
is created at import; tests tweak its fields directly rather than re-reading the env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.config import DEFAULT_SAMPLING_RATE


def _parse_keys(raw: str | None) -> frozenset[str]:
    return frozenset(k.strip() for k in (raw or "").split(",") if k.strip())


@dataclass
class Settings:
    # Comma-separated in APEX_API_KEYS. Empty => auth disabled (open, dev mode).
    api_keys: frozenset[str] = field(default_factory=lambda: _parse_keys(os.environ.get("APEX_API_KEYS")))
    api_key_header: str = os.environ.get("APEX_API_KEY_HEADER", "X-API-Key")
    # Fixed-window rate limit per client (API key, else source IP).
    rate_limit: int = int(os.environ.get("APEX_RATE_LIMIT", "60"))
    rate_window_s: float = float(os.environ.get("APEX_RATE_WINDOW_S", "60"))
    # Detector device + explanation backend defaults for the service.
    device: str = os.environ.get("APEX_DEVICE", "cpu")
    default_backend: str = os.environ.get("APEX_BACKEND", "template")
    default_sampling_rate: int = int(os.environ.get("APEX_SAMPLING_RATE", str(DEFAULT_SAMPLING_RATE)))
    warmup_on_startup: bool = os.environ.get("APEX_WARMUP", "1") not in ("0", "false", "False", "")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


SETTINGS = Settings()
