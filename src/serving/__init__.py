"""Phase 8 — structured JSON output layer.

The single source of truth for APEX's response schema and the serializer that composes
every upstream stage (detection, grounding, generation, reliability) into it. The
FastAPI backend (`app/backend/main.py`) and any other caller import from here so there
is one schema, validated in one place.
"""

from src.serving.schema import (
    DISCLAIMER,
    MIN_LEADS,
    MIN_RELIABLE_SECONDS,
    SCHEMA_VERSION,
    APEXReport,
    ConsistencyOut,
    FindingOut,
    Flag,
    FlagType,
    InputValidation,
    InputValidationError,
)
from src.serving.serializer import build_report, validate_signal

__all__ = [
    "APEXReport",
    "FindingOut",
    "Flag",
    "FlagType",
    "ConsistencyOut",
    "InputValidation",
    "InputValidationError",
    "DISCLAIMER",
    "MIN_LEADS",
    "MIN_RELIABLE_SECONDS",
    "SCHEMA_VERSION",
    "build_report",
    "validate_signal",
]
