import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.model_registry import MODELS
from app.services.prediction_service import TENSILE_FEATURES
client = TestClient(app)
CASES = [
    (
        "/api/v1/predict/tensile-strength",
        "tensile_strength",
        "carbon_steel",
    ),
    (
        "/api/v1/predict/yield-strength",
        "yield_strength",
        "low_alloy_normalized",
    ),
]
def fitted_imputation_value(model, feature):
    preprocessing = model.named_steps["preprocessing"]
    for _, transformer, columns in preprocessing.transformers_:
        if isinstance(transformer, str):
            continue
        imputer = getattr(transformer, "named_steps", {}).get("imputer")
        if imputer is not None and feature in columns:
            return imputer.statistics_[list(columns).index(feature)].item() \
                if hasattr(imputer.statistics_[list(columns).index(feature)], "item") \
                else imputer.statistics_[list(columns).index(feature)]
    raise AssertionError(f"No fitted imputation value for {feature}")
@pytest.mark.parametrize("endpoint,model_key,family", CASES)
def test_partial_prediction_reports_fitted_imputation(
    endpoint, model_key, family
):
    response = client.post(endpoint, json={"C": 0.0, "Mn": 0.0})
    assert response.status_code == 200
    evidence = response.json()["input_evidence"]
    assert evidence["total_features"] == 13
    assert evidence["provided_feature_count"] == 2
    assert evidence["imputed_feature_count"] == 11
    assert evidence["provided_feature_fraction"] == pytest.approx(2 / 13)
    assert evidence["feature_details"]["C"]["status"] == "PROVIDED"
    assert evidence["feature_details"]["C"]["supplied_value"] == 0.0
    assert evidence["feature_details"]["Mn"]["status"] == "PROVIDED"
    model = MODELS[model_key]
    for feature in evidence["imputed_features"]:
        detail = evidence["feature_details"][feature]
        assert detail["status"] == "IMPUTED_BY_MODEL"
        assert detail["imputation_strategy"] in {"median", "most_frequent"}
        assert detail["imputation_value"] == fitted_imputation_value(
            model, feature
        )
@pytest.mark.parametrize("endpoint,model_key,family", CASES)
def test_complete_prediction_reports_no_imputation(
    endpoint, model_key, family
):
    payload = {
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
        "steel_family": family,
    }
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    evidence = response.json()["input_evidence"]
    assert evidence["provided_feature_count"] == 13
    assert evidence["imputed_feature_count"] == 0
    assert evidence["imputed_features"] == []
    assert evidence["provided_feature_fraction"] == 1.0
    for feature in TENSILE_FEATURES:
        assert evidence["feature_details"][feature]["status"] == "PROVIDED"
