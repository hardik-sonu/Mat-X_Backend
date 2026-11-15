from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
for endpoint, expected_values in [
    (
        "/api/v1/predict/tensile-strength",
        {
            "Cr": 0.89,
            "quench_medium": "air",
            "steel_family": "carbon_steel",
        },
    ),
    (
        "/api/v1/predict/yield-strength",
        {
            "Cr": 9.0,
            "quench_medium": "air",
            "steel_family": "low_alloy_normalized",
        },
    ),
]:
    response = client.post(endpoint, json={"C": 0.0, "Mn": 0.0})
    print("\nENDPOINT:", endpoint)
    print("HTTP:", response.status_code)
    assert response.status_code == 200, response.text
    result = response.json()
    evidence = result["input_evidence"]
    assert evidence["total_features"] == 13
    assert evidence["provided_feature_count"] == 2
    assert evidence["imputed_feature_count"] == 11
    assert evidence["feature_details"]["C"]["status"] == "PROVIDED"
    assert evidence["feature_details"]["C"]["supplied_value"] == 0.0
    assert evidence["feature_details"]["Mn"]["status"] == "PROVIDED"
    for feature, expected in expected_values.items():
        actual = evidence["feature_details"][feature]["imputation_value"]
        assert actual == expected, (feature, actual, expected)
    print("Prediction:", result["prediction"])
    print("Provided:", evidence["provided_feature_count"])
    print("Imputed:", evidence["imputed_feature_count"])
    print("Provided fraction:", evidence["provided_feature_fraction"])
    print("Imputed fields:")
    for feature in evidence["imputed_features"]:
        detail = evidence["feature_details"][feature]
        print(
            f"  {feature}: "
            f"{detail['imputation_strategy']} -> "
            f"{detail['imputation_value']}"
        )
print("\nPREDICTION INPUT-EVIDENCE VERIFICATION PASSED")
