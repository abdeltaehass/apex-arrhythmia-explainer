"""The APEX structured-output schema (Phase 8).

Pydantic v2 models defining the exact JSON APEX returns for one recording. This is the
"structured schema defined earlier" (the Phase-0 `app/backend/main.py` stub), now
formalized and expanded to carry every stage's output:

    findings[]          label, confidence, implicated leads, per-finding flag status
    impression          the interpretive one/two-liner (Impression section)
    explanation         the full generated report text (Findings + Impression)
    consistency         the src/eval/consistency.py result (asserted vs surfaced)
    review_recommended  a single boolean gate for the caller
    input_validation    lead-count / duration checks (see serializer.validate_signal)

Kept free of torch / model imports so it can be validated, serialized, and tested
without loading anything — the deliverable's schema-validation test suite runs against
these models alone.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from src.config import NUM_LEADS

DISCLAIMER = "Decision support only — verify against the full clinical picture."
SCHEMA_VERSION = "1.0"

# Input-validation thresholds (Phase-8 spec).
MIN_LEADS = NUM_LEADS                 # reject recordings with fewer than 12 leads
MIN_RELIABLE_SECONDS = 5.0            # flag recordings shorter than this as unreliable


class InputValidationError(ValueError):
    """Raised when a recording fails a hard validation rule (e.g. < 12 leads).

    Carries the :class:`InputValidation` so the caller (e.g. the FastAPI layer) can
    surface the specific errors rather than a bare message.
    """

    def __init__(self, validation: InputValidation):
        self.validation = validation
        super().__init__("; ".join(validation.errors) or "invalid input recording")


class FlagType(StrEnum):
    """Why a finding (or the whole report) was flagged for review."""

    LOW_CONFIDENCE = "low_confidence"
    GROUNDING_CONFLICT = "grounding_conflict"
    MUTUAL_EXCLUSIVITY = "mutual_exclusivity"
    UNRELIABLE_INPUT = "unreliable_input"


class Flag(BaseModel):
    type: FlagType
    message: str


class FindingOut(BaseModel):
    """One detected finding, with its evidence and flag status."""

    label: str = Field(..., description="SCP-ECG code, e.g. 'AFIB'")
    description: str = Field("", description="human-readable label, e.g. 'atrial fibrillation'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    leads: list[str] = Field(default_factory=list, description="leads implicated in this finding")
    flags: list[Flag] = Field(default_factory=list, description="per-finding flag status")
    needs_review: bool = Field(..., description="confidence below threshold or any flag present")


class ConsistencyOut(BaseModel):
    """Serialized `src/eval/consistency.py` result."""

    consistent: bool
    asserted: list[str] = Field(default_factory=list, description="findings the explanation names")
    surfaced: list[str] = Field(default_factory=list, description="labels the detector surfaced")
    unsupported: list[str] = Field(default_factory=list,
                                   description="asserted but not surfaced (hallucinations)")


class InputValidation(BaseModel):
    """Result of the lead-count / duration gate."""

    n_leads: int
    duration_s: float
    sampling_rate: int
    ok: bool = Field(..., description="False if a hard reject rule failed (e.g. wrong lead count)")
    reliable: bool = Field(..., description="False if the recording is usable but flagged unreliable")
    errors: list[str] = Field(default_factory=list, description="hard failures that reject the record")
    warnings: list[str] = Field(default_factory=list, description="soft flags (still processed)")


class APEXReport(BaseModel):
    """The complete structured response for one recording."""

    findings: list[FindingOut] = Field(default_factory=list)
    impression: str = ""
    explanation: str = ""
    consistency: ConsistencyOut
    review_recommended: bool
    input_validation: InputValidation | None = None
    disclaimer: str = DISCLAIMER
    schema_version: str = SCHEMA_VERSION
