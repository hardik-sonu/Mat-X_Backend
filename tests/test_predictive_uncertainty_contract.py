import pytest
from app.main import app
from fastapi.testclient import TestClient
client = TestClient(app)
CASES = [
    ("/api/v1/predict/tensile-strength", "carbon_steel"),
    ("/api/v1/predict/yield-strength", "low_alloy_normalized"),
]
PAYLOAD = {
    "C": 0.20,
    "Mn": 1.0,
    "Cr": 0.5,
    "Mo": 0.2,
    "Ni": 0.1,
    "Si": 0.25,
    "V": 0.02,
    "Cu": 0.1,
    "Al": 0.03,
    "austenitize_T": 850,
    "temper_T": 550,
    "quench_medium": "oil",
}
@pytest.mark.parametrize("endpoint,family", CASES)
def test_prediction_explicitly_reports_unavailable_calibrated_uncertainty(
    endpoint, family
):
    response = client.post(
        endpoint,
        json={**PAYLOAD, "steel_family": family},
    )
    assert response.status_code == 200
    uncertainty = response.json()["predictive_uncertainty"]
    assert uncertainty["status"] == "NOT_AVAILABLE"
    assert uncertainty["calibration_status"] == "NOT_PROMOTED"
    assert uncertainty["interval_lower"] is None
    assert uncertainty["interval_upper"] is None
    assert uncertainty["confidence_level"] is None
    assert uncertainty["method"] is None
    assert "not used" in uncertainty["reason"].lower()
def test_predictive_uncertainty_is_documented_in_openapi():
    schema = app.openapi()["components"]["schemas"]
    assert "PredictiveUncertaintyResponse" in schema
    for response_name in (
        "TensilePredictionResponse",
        "YieldPredictionResponse",
    ):
        properties = schema[response_name]["properties"]
        assert "predictive_uncertainty" in properties
