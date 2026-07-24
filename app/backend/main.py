"""APEX FastAPI backend.

Exposes the pipeline: signal -> detection -> grounding -> explanation -> review gate,
serialized into the Phase-8 structured schema (`src/serving/`). The response contract
(`APEXReport`) lives in `src.serving.schema` so the API, the frontend, and the tests
all share one definition.

The detector is loaded lazily on the first `/analyze` call (and cached), so the app
starts instantly and importing this module never requires torch or a checkpoint.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.schema import APEXReport, InputValidationError
from src.serving.serializer import analyze_signal, validate_signal

app = FastAPI(title="APEX", version="0.8.0", description="Arrhythmia Pattern Explainer")


class ECGRequest(BaseModel):
    # 12 leads x T samples, in the PTB-XL lead order (I, II, III, aVR, aVL, aVF, V1..V6).
    signal: list[list[float]] = Field(..., description="12-lead signal, shape [12][T]")
    sampling_rate: int = 100
    backend: str = Field("template", description="explanation backend: template | claude | local")
    with_grounding: bool = Field(False, description="also compute per-lead saliency (slower)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/validate")
def validate(req: ECGRequest) -> dict:
    """Run only the input gate — cheap, no model load. Returns the InputValidation."""
    try:
        return validate_signal(req.signal, req.sampling_rate).model_dump()
    except InputValidationError as e:
        return e.validation.model_dump()


@app.post("/analyze", response_model=APEXReport)
def analyze(req: ECGRequest) -> APEXReport:
    """Full pipeline -> structured report. 422 if the recording fails a hard input rule."""
    try:
        return analyze_signal(
            req.signal, req.sampling_rate,
            backend=req.backend, with_grounding=req.with_grounding,
        )
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=e.validation.model_dump()) from e
