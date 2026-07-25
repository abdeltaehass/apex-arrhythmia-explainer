"""Phase-9 API tests: endpoints, auth, rate limiting, metrics, file upload.

The metrics collector, rate limiter, and settings are process-global, so each test that
touches them saves and restores state. Model-driven `/analyze` assertions are skipped
when torch or the checkpoint is unavailable; the auth/rate-limit/validation tests use
the model-free endpoints so they always run.
"""

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, needed for UploadFile/Form

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import app  # noqa: E402
from src.config import ROOT  # noqa: E402
from src.serving.metrics import METRICS, MetricsCollector, percentile  # noqa: E402
from src.serving.security import LIMITER, RateLimiter  # noqa: E402
from src.serving.settings import SETTINGS  # noqa: E402

HAS_CHECKPOINT = (ROOT / "outputs" / "final_best.pt").exists()
needs_model = pytest.mark.skipif(not HAS_CHECKPOINT, reason="no detector checkpoint")


@pytest.fixture
def client():
    # no context manager -> skip startup warmup; the model loads lazily + cached.
    return TestClient(app)


@pytest.fixture
def restore_auth():
    original = SETTINGS.api_keys
    yield
    SETTINGS.api_keys = original


@pytest.fixture
def restore_limiter():
    limit, window = LIMITER.limit, LIMITER.window_s
    LIMITER.reset()
    yield
    LIMITER.limit, LIMITER.window_s = limit, window
    LIMITER.reset()


def _signal(seed=0):
    return np.random.default_rng(seed).standard_normal((12, 1000)).astype(float)


# --- metrics collector unit tests --------------------------------------------
def test_percentile_interpolates():
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert percentile([0.0, 100.0], 99) == pytest.approx(99.0)


def test_metrics_collector_counts_and_percentiles():
    m = MetricsCollector()
    for latency in [0.01, 0.02, 0.03, 0.04]:
        m.record(latency, ok=True)
    m.record(0.05, ok=False)
    snap = m.snapshot()
    assert snap.request_count == 5
    assert snap.error_count == 1
    assert snap.max_latency_ms == pytest.approx(50.0)
    assert 10.0 <= snap.p50_latency_ms <= 50.0
    assert snap.window_size == 5


def test_metrics_collector_reset():
    m = MetricsCollector()
    m.record(0.1)
    m.reset()
    assert m.snapshot().request_count == 0


# --- rate limiter unit test --------------------------------------------------
def test_rate_limiter_allows_then_blocks():
    rl = RateLimiter(limit=2, window_s=100)
    assert rl.check("c1")[0] is True   # 1
    assert rl.check("c1")[0] is True   # 2
    allowed, remaining, retry = rl.check("c1")  # 3 -> blocked
    assert not allowed and remaining == 0 and retry > 0
    assert rl.check("c2")[0] is True   # a different client is independent


# --- health / metrics endpoints ----------------------------------------------
def test_health_reports_version_and_status(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert "model_version" in body and "schema_version" in body
    assert body["api_version"] == "0.9.0"


def test_metrics_endpoint_shape(client):
    body = client.get("/metrics").json()
    assert "request_count" in body
    assert set(body["latency_ms"]) == {"p50", "p95", "p99", "mean", "max"}


# --- input validation over HTTP ----------------------------------------------
def test_validate_endpoint_rejects_bad_lead_count(client):
    body = client.post("/validate", json={"signal": [[0.0] * 1000 for _ in range(8)],
                                          "sampling_rate": 100}).json()
    assert not body["ok"]


def test_analyze_rejects_fewer_than_twelve_leads(client):
    resp = client.post("/analyze", json={"signal": [[0.0] * 1000 for _ in range(8)],
                                        "sampling_rate": 100})
    assert resp.status_code == 422


def test_analyze_rejects_image_upload(client):
    resp = client.post("/analyze", files={"file": ("scan.png", b"\x89PNG\r\n", "image/png")})
    assert resp.status_code == 415
    assert "image" in resp.json()["detail"].lower()


def test_analyze_missing_signal_field(client):
    assert client.post("/analyze", json={"sampling_rate": 100}).status_code == 422


# --- auth --------------------------------------------------------------------
def test_metrics_requires_key_when_auth_enabled(client, restore_auth):
    SETTINGS.api_keys = frozenset({"secret-key"})
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "secret-key"}).status_code == 200


def test_auth_disabled_by_default_is_open(client):
    assert not SETTINGS.auth_enabled          # no APEX_API_KEYS in the test env
    assert client.get("/metrics").status_code == 200


# --- rate limiting -----------------------------------------------------------
def test_rate_limit_returns_429_after_limit(client, restore_limiter):
    LIMITER.limit = 3
    payload = {"signal": [[0.0] * 1000 for _ in range(12)], "sampling_rate": 100}
    codes = [client.post("/validate", json=payload).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


# --- model-driven end-to-end (needs the checkpoint) --------------------------
@needs_model
def test_analyze_json_returns_report(client):
    from src.serving.schema import APEXReport

    resp = client.post("/analyze", json={"signal": _signal().tolist(), "sampling_rate": 100,
                                        "backend": "template"})
    assert resp.status_code == 200
    report = APEXReport.model_validate(resp.json())
    assert report.input_validation.ok
    assert isinstance(report.review_recommended, bool)


@needs_model
def test_analyze_npy_file_upload_matches_schema(client):
    from src.serving.schema import APEXReport

    buf = io.BytesIO()
    np.save(buf, _signal().astype(np.float32))
    resp = client.post("/analyze",
                       files={"file": ("ecg.npy", buf.getvalue(), "application/octet-stream")},
                       data={"sampling_rate": "100", "backend": "template"})
    assert resp.status_code == 200
    APEXReport.model_validate(resp.json())


@needs_model
def test_analyze_records_into_metrics(client):
    METRICS.reset()
    client.post("/analyze", json={"signal": _signal(1).tolist(), "sampling_rate": 100})
    snap = client.get("/metrics").json()
    assert snap["request_count"] >= 1
    assert snap["latency_ms"]["p50"] > 0
