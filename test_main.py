import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "square_footage": 1500.0,
    "num_bedrooms": 3,
    "num_bathrooms": 2.0,
    "age_years": 10,
    "location_tier": "suburban",
    "has_garage": True,
}

def test_predict_valid_input_returns_200():
    response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "success"
    assert body["predicted_value_usd"] > 0
    assert 0.0 <= body["confidence_score"] <= 100.0
    assert body["model_version"] != ""

def test_predict_invalid_location_returns_422():
    bad_payload = {**VALID_PAYLOAD, "location_tier": "downtown"}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    assert response.json()["status"] == "error"

def test_predict_square_footage_out_of_range_returns_422():
    bad_payload = {**VALID_PAYLOAD, "square_footage": 50.0}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422

def test_predict_invalid_bathroom_step_returns_422():
    bad_payload = {**VALID_PAYLOAD, "num_bathrooms": 1.3}
    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422

def test_health_check_returns_healthy():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
