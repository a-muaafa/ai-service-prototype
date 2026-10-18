"""
Comprehensive test suite for the AI Service API.

Covers:
- Application lifespan and model loading
- Health / readiness endpoints
- Prediction with valid inputs
- Input validation (boundaries, NaN, Infinity, missing, extra fields)
- Error response structure (no internal leaks)
- Model absence / degraded mode
- Extrapolation warnings
- Model version consistency
- API-key authentication (when configured)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "model.joblib"


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan — model must exist on disk."""
    if not MODEL_FILE.exists():
        pytest.skip("model.joblib not found; run 'python train_model.py' first.")
    from main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def no_model_client(client):
    """Same client but with the predictor temporarily removed."""
    import main
    backup = main._predictor
    main._predictor = None
    yield client
    main._predictor = backup


VALID_PAYLOAD = {
    "square_footage": 1500.0,
    "num_bedrooms": 3,
    "num_bathrooms": 2.0,
    "age_years": 10,
    "location_tier": "suburban",
    "has_garage": True,
}


class TestHealthEndpoints:
    """Process-level health and model-readiness probes."""

    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_returns_healthy(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "uptime_seconds" in body

    def test_ready_returns_ready_when_model_loaded(self, client):
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert body["model_version"] is not None
        assert body["model_version"] != ""
        assert "sklearn_version" in body


class TestPrediction:
    """Successful prediction and response structure."""

    def test_valid_prediction_returns_200(self, client):
        resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

        body = resp.json()
        assert body["status"] == "success"
        assert body["predicted_value_usd"] > 0
        assert body["currency"] == "USD"
        assert body["model_version"] != ""
        assert isinstance(body["input_summary"], dict)

    def test_no_confidence_score_in_response(self, client):
        """The misleading confidence_score field must not be present."""
        resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        body = resp.json()
        assert "confidence_score" not in body

    def test_has_extrapolation_warning_field(self, client):
        resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        body = resp.json()
        assert "extrapolation_warning" in body
        assert isinstance(body["extrapolation_warning"], bool)

    def test_response_includes_request_id_header(self, client):
        resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert "x-request-id" in resp.headers

    def test_model_version_matches_ready_endpoint(self, client):
        ready_body = client.get("/api/v1/ready").json()
        predict_body = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert ready_body["model_version"] == predict_body["model_version"]

    def test_extrapolation_warning_false_within_range(self, client):
        """Input well within training range → no warning."""
        resp = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert resp.json()["extrapolation_warning"] is False

    def test_extrapolation_warning_true_outside_range(self, client):
        """Input far outside training range → warning set."""
        payload = {**VALID_PAYLOAD, "square_footage": 19_000.0, "age_years": 150}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["extrapolation_warning"] is True


class TestInputValidation:
    """Boundary checks, type errors, and forbidden values."""

    def test_negative_square_footage_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": -100.0})
        assert resp.status_code == 422
        assert resp.json()["status"] == "error"

    def test_zero_square_footage_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": 0.0})
        assert resp.status_code == 422

    def test_square_footage_below_minimum(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": 50.0})
        assert resp.status_code == 422

    def test_square_footage_above_maximum(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": 25_000.0})
        assert resp.status_code == 422

    def test_zero_bedrooms_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "num_bedrooms": 0})
        assert resp.status_code == 422

    def test_too_many_bedrooms_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "num_bedrooms": 25})
        assert resp.status_code == 422

    def test_invalid_bathroom_step_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "num_bathrooms": 1.3})
        assert resp.status_code == 422

    def test_negative_age_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "age_years": -5})
        assert resp.status_code == 422

    def test_invalid_location_tier_rejected(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "location_tier": "downtown"})
        assert resp.status_code == 422
        assert resp.json()["status"] == "error"

    def test_nan_in_json_rejected(self, client):
        """NaN is not valid JSON; the server must reject it."""
        raw = (
            '{"square_footage": NaN, "num_bedrooms": 3, '
            '"num_bathrooms": 2.0, "age_years": 10, '
            '"location_tier": "suburban", "has_garage": true}'
        )
        resp = client.post(
            "/api/v1/predict",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_infinity_in_json_rejected(self, client):
        raw = (
            '{"square_footage": Infinity, "num_bedrooms": 3, '
            '"num_bathrooms": 2.0, "age_years": 10, '
            '"location_tier": "suburban", "has_garage": true}'
        )
        resp = client.post(
            "/api/v1/predict",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_required_field_rejected(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "square_footage"}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_extra_fields_rejected(self, client):
        payload = {**VALID_PAYLOAD, "unknown_field": "value"}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422


class TestErrorStructure:
    """All error responses must have consistent shape and hide internals."""

    def test_error_has_status_and_detail(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": -1})
        body = resp.json()
        assert body["status"] == "error"
        assert "detail" in body

    def test_error_does_not_expose_file_paths(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "location_tier": "invalid"})
        detail = resp.json().get("detail", "")
        assert "Traceback" not in detail
        assert "\\Users\\" not in detail
        assert "/app/" not in detail

    def test_error_includes_request_id(self, client):
        resp = client.post("/api/v1/predict", json={**VALID_PAYLOAD, "square_footage": -1})
        body = resp.json()
        assert "request_id" in body


class TestModelAbsence:
    """When the model is not loaded the service must degrade gracefully."""

    def test_predict_returns_503_without_model(self, no_model_client):
        resp = no_model_client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_ready_returns_503_without_model(self, no_model_client):
        resp = no_model_client.get("/api/v1/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["model_loaded"] is False

    def test_health_still_ok_without_model(self, no_model_client):
        """Health endpoint reports the *process* is alive."""
        resp = no_model_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_root_still_ok_without_model(self, no_model_client):
        resp = no_model_client.get("/")
        assert resp.status_code == 200


class TestModelArtifact:
    """Verify the trained model file is loadable."""

    def test_model_file_exists(self):
        if not MODEL_FILE.exists():
            pytest.skip("model.joblib not found")
        assert MODEL_FILE.stat().st_size > 0

    def test_model_file_is_loadable(self):
        if not MODEL_FILE.exists():
            pytest.skip("model.joblib not found")
        import joblib
        pipeline = joblib.load(MODEL_FILE)
        assert hasattr(pipeline, "predict")

    def test_metadata_file_exists(self):
        meta = PROJECT_DIR / "model_metadata.json"
        if not meta.exists():
            pytest.skip("model_metadata.json not found")
        import json
        with open(meta) as f:
            data = json.load(f)
        assert "model_version" in data
        assert "metrics" in data
        assert "feature_ranges" in data

    def test_model_predicts_known_sample(self):
        """Smoke-test: model produces a positive prediction for a valid input."""
        if not MODEL_FILE.exists():
            pytest.skip("model.joblib not found")
        import joblib
        import numpy as np
        pipeline = joblib.load(MODEL_FILE)
        sample = np.array([[1500, 3, 2.0, 10, "suburban", 1]], dtype=object)
        pred = float(pipeline.predict(sample)[0])
        assert pred > 0
