import pytest
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
CASES = [
    (
        "/api/v1/predict/tensile-strength",
        "carbon_steel",
    ),
    (
        "/api/v1/predict/yield-strength",
        "low_alloy_normalized",
    ),
]
COMPLETE_PAYLOAD = {
    "C": 0.20,
    "Mn": 1.0,
    "Cr": 1.0,
    "Mo": 0.2,
    "Ni": 1.0,
    "Si": 0.5,
    "V": 0.0,
    "Cu": 0.2,
    "Al": 0.0,
    "austenitize_T": 870.0,
    "temper_T": 620.0,
    "quench_medium": "air",
}
@pytest.mark.parametrize("endpoint,family", CASES)
def test_complete_prediction_includes_applicability(endpoint, family):
    payload = {**COMPLETE_PAYLOAD, "steel_family": family}
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "input_evidence" in body
    assert "applicability" in body
    applicability = body["applicability"]
    assert applicability["status"] in {
        "IN_DOMAIN",
        "BORDERLINE",
        "OUT_OF_DOMAIN",
    }
    assert applicability["distance"] is not None
    assert applicability["provided_feature_count"] == 13
    assert applicability["missing_features"] == []
    assert applicability["confidence_probability"] is None
@pytest.mark.parametrize("endpoint,family", CASES)
def test_partial_prediction_remains_available(endpoint, family):
    response = client.post(
        endpoint,
        json={"C": 0.20, "Mn": 1.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["input_evidence"]["imputed_feature_count"] == 11
    applicability = body["applicability"]
    assert applicability["status"] == "INSUFFICIENT_EVIDENCE"
    assert applicability["distance"] is None
    assert applicability["provided_feature_count"] == 2
    assert len(applicability["missing_features"]) == 11
@pytest.mark.parametrize("endpoint,family", CASES)
def test_extreme_input_is_flagged_without_blocking_prediction(endpoint, family):
    payload = {
        **COMPLETE_PAYLOAD,
        "C": 1_000_000.0,
        "steel_family": family,
    }
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["applicability"]["status"] == "OUT_OF_DOMAIN"
    assert body["applicability"]["distance"] is not None
