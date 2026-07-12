"""APEX FastAPI backend.

Exposes the pipeline: signal -> detection -> grounding -> explanation -> review gate.
This is a Phase-0 skeleton; the model is not wired in yet (returns a stub) so the
API contract can be developed against immediately.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import CFG

app = FastAPI(title="APEX", version="0.1.0", description="Arrhythmia Pattern Explainer")


class ECGRequest(BaseModel):
    # 12 leads x T samples. In production this may be a file upload instead.
    signal: list[list[float]] = Field(..., description="12-lead signal, shape [12][T]")
    sampling_rate: int = 100


class Finding(BaseModel):
    label: str
    description: str
    confidence: float
    needs_review: bool


class APEXResponse(BaseModel):
    findings: list[Finding]
    explanation: str
    any_needs_review: bool
    disclaimer: str = "Decision support only — verify against the full clinical picture."


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/analyze", response_model=APEXResponse)
def analyze(req: ECGRequest) -> APEXResponse:
    # TODO(phase1+): run detector -> grounding -> generator on req.signal.
    _ = req  # silence unused until wired
    stub = Finding(
        label="NORM",
        description="Normal ECG",
        confidence=0.0,
        needs_review=True,  # stub is always below threshold
    )
    return APEXResponse(
        findings=[stub],
        explanation="Model not yet wired in (Phase 0 skeleton).",
        any_needs_review=True,
    )
