"""APEX FastAPI service (Phase 9).

Wraps the full pipeline (signal -> detection -> grounding -> generation -> reliability
-> structured report) behind an authenticated, rate-limited HTTP API:

    POST /analyze   signal file, ECG image, or JSON body -> APEXReport (structured JSON)
    POST /validate  JSON body                   -> InputValidation (input gate only)
    GET  /health    model version + status
    GET  /metrics   request count + p50/p95/p99 latency since startup

The response schema and all composition live in `src/serving/`; this module is only the
HTTP surface. The detector is warmed at startup (configurable) and cached, so /analyze
serves warm requests without reloading the checkpoint. Auth (API key) and rate limiting
are enabled by environment (`APEX_API_KEYS`, `APEX_RATE_LIMIT`); with no keys configured
the service runs open for local development.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.serving.loaders import UnsupportedUploadError, parse_signal_upload
from src.serving.metrics import METRICS
from src.serving.schema import SCHEMA_VERSION, APEXReport, InputValidationError
from src.serving.security import rate_limit, require_api_key
from src.serving.serializer import analyze_signal, validate_signal
from src.serving.settings import SETTINGS

API_VERSION = "0.9.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SETTINGS.warmup_on_startup:
        try:
            from src.serving.model_cache import warmup

            warmup(device=SETTINGS.device)
        except Exception as e:  # a missing checkpoint shouldn't stop the app booting
            print(f"[apex] warmup skipped: {e}")
    yield


app = FastAPI(title="APEX", version=API_VERSION,
              description="Arrhythmia Pattern Explainer — clinical decision support (not diagnostic)",
              lifespan=lifespan)


@app.middleware("http")
async def record_latency(request: Request, call_next):
    """Time /analyze requests into the metrics collector (success and failure)."""
    if request.url.path != "/analyze":
        return await call_next(request)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS.record(time.perf_counter() - start, ok=False)
        raise
    METRICS.record(time.perf_counter() - start, ok=response.status_code < 500)
    return response


class ECGRequest(BaseModel):
    signal: list[list[float]] = Field(..., description="12-lead signal, shape [12][T]")
    sampling_rate: int = SETTINGS.default_sampling_rate
    backend: str = Field(SETTINGS.default_backend, description="template | claude | local")
    with_grounding: bool = False


def _model_status() -> dict:
    try:
        from src.serving.model_cache import get_detector

        _, label_space, args = get_detector(device=SETTINGS.device)
        return {"model_loaded": True, "model": args.get("model") or "cnn",
                "num_labels": len(label_space), "device": SETTINGS.device}
    except Exception as e:  # pragma: no cover - only when checkpoint is missing
        return {"model_loaded": False, "error": str(e), "device": SETTINGS.device}


@app.get("/health")
def health() -> dict:
    """Model version + status. Always 200 so an orchestrator can read the body."""
    status = _model_status()
    return {
        "status": "ok" if status.get("model_loaded") else "degraded",
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_version": f"{status.get('model', 'unknown')}-{status.get('num_labels', '?')}labels",
        "auth_enabled": SETTINGS.auth_enabled,
        **status,
    }


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> dict:
    """Request count + p50/p95/p99 latency over the /analyze calls since startup."""
    return METRICS.snapshot().to_dict()


@app.post("/validate", dependencies=[Depends(rate_limit)])
def validate(req: ECGRequest) -> dict:
    """Run only the input gate — cheap, no model load."""
    try:
        return validate_signal(req.signal, req.sampling_rate).model_dump()
    except InputValidationError as e:
        return e.validation.model_dump()


async def _signal_from_request(request: Request):
    """Extract ``(signal, sampling_rate, backend, with_grounding)`` from a request.

    Accepts multipart file upload (``file`` + optional ``sampling_rate``/``backend``/
    ``with_grounding`` form fields) or a JSON body (`ECGRequest`).
    """
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=422, detail="multipart upload requires a 'file' field")
        signal, sr = parse_signal_upload(upload.filename or "upload", await upload.read())
        sampling_rate = int(form.get("sampling_rate") or sr or SETTINGS.default_sampling_rate)
        backend = form.get("backend") or SETTINGS.default_backend
        with_grounding = str(form.get("with_grounding", "")).lower() in ("1", "true", "yes", "on")
        return signal, sampling_rate, backend, with_grounding
    body = await request.json()
    if "signal" not in body:
        raise HTTPException(status_code=422, detail="JSON body must contain a 'signal' field")
    return (body["signal"], int(body.get("sampling_rate", SETTINGS.default_sampling_rate)),
            body.get("backend", SETTINGS.default_backend), bool(body.get("with_grounding", False)))


@app.post("/analyze", response_model=APEXReport,
          dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def analyze(request: Request) -> APEXReport:
    """Full pipeline -> structured report. Accepts a signal file, an ECG image, or JSON.

    An uploaded paper-ECG image is digitized to a signal first (Phase 10). 422 on a hard
    input-validation failure, 415 on an unreadable/unsupported upload.
    """
    try:
        signal, sampling_rate, backend, with_grounding = await _signal_from_request(request)
    except UnsupportedUploadError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e

    try:
        return analyze_signal(signal, sampling_rate, backend=backend,
                              with_grounding=with_grounding, device=SETTINGS.device)
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=e.validation.model_dump()) from e


@app.post("/metrics/reset", dependencies=[Depends(require_api_key)])
def reset_metrics() -> Response:
    """Zero the metrics counters (ops convenience)."""
    METRICS.reset()
    return Response(status_code=204)
