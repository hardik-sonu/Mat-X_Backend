from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_engineering_report_returns_structured_evidence_report():
    response = client.post(
        "/api/v1/engineering/report",
        json={
            "material_id": "MATX-REPORT-001",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    report = response.json()
    assert report["report"]["title"] == (
        "MAT-X Engineering Material-State Report"
    )
    assert report["report"]["report_schema_version"] == "1.0.0"
    assert report["report"]["report_status"] == (
        "EVIDENCE_AGGREGATED_NOT_CERTIFIED"
    )
    assert report["material"]["material_id"] == "MATX-REPORT-001"
    assert report["executive_summary"]["engineering_decision"] == (
        "HUMAN_REVIEW_REQUIRED"
    )
    assert report["prediction_results"]["status"] == "NO_PERSISTED_PREDICTIONS"
    assert report["prediction_results"]["results"] == []
    assert report["prediction_results"]["record_count"] == 0
    experiment = report["active_investigation"]["experiment_g_evidence"]
    assert experiment["experiment"] == "EXPERIMENT_G"
    assert experiment["status"] == (
        "LIMITED_EXPERIMENTAL_PREDICTION_ERROR_ABLATION"
    )
    assert experiment["scope"]["measures_information_gain"] is False
    assert experiment["scope"]["validates_physical_test_acquisition"] is False

    assert report["validated_evidence_registry"]["record_count"] >= 0
    assert report["limitations"]
    assert "certified" in report["engineering_disclaimer"].lower()
def test_engineering_report_preserves_user_supplied_provenance():
    response = client.post(
        "/api/v1/engineering/report",
        json={
            "material_id": "MATX-REPORT-002",
            "material_family": "low_alloy_steel",
            "composition": {"C": 0.25},
            "measured_properties": {"hardness": 220},
        },
    )
    assert response.status_code == 200
    report = response.json()
    composition = report["source_evidence"]["categories"][
        "chemical_composition"
    ]
    properties = report["source_evidence"]["categories"][
        "mechanical_properties"
    ]
    assert composition["provenance"] == "SOURCE_VALUE"
    assert composition["source"] == "USER_SUPPLIED"
    assert properties["provenance"] == "SOURCE_VALUE"
    assert properties["source"] == "USER_SUPPLIED"
    assert report["prediction_results"]["status"] == "NO_PERSISTED_PREDICTIONS"
def test_engineering_report_rejects_invalid_request():
    response = client.post(
        "/api/v1/engineering/report",
        json={
            "material_id": "",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 422
def test_engineering_report_preserves_unverified_user_damage_observation():
    response = client.post(
        "/api/v1/engineering/report",
        json={
            "material_id": "MATX-REPORT-DAMAGE-001",
            "material_family": "low_alloy_steel",
            "damage_condition": "Operator reports visible corrosion",
        },
    )
    assert response.status_code == 200
    report = response.json()
    observation = report["user_reported_damage_observation"]
    assert observation["status"] == "USER_REPORTED_UNVERIFIED"
    assert observation["value"] == "Operator reports visible corrosion"
    damage_evidence = report["source_evidence"]["categories"][
        "damage_condition"
    ]
    assert damage_evidence["status"] == "UNAVAILABLE"
    assert damage_evidence["provenance"] == "UNAVAILABLE"
    assert report["damage_condition_assessment"]["observation_count"] == 0

def test_engineering_report_includes_persisted_predictions_for_material(
    monkeypatch,
):
    from app.services import engineering_report_service
    records = [
        {
            "prediction_id": "PRED-REPORT-001",
            "material_id": "MATX-REPORT-PRED-001",
            "target_property": "tensile_strength",
            "prediction_type": "NUMERIC",
            "predicted_value": 650.0,
            "predicted_unit": None,
            "model_name": "tensile_model",
            "model_version": "a" * 64,
            "input_json": {"C": 0.25, "Mn": 1.0},
            "applicability_json": {
                "applicability": {"status": "IN_DOMAIN"},
                "predictive_uncertainty": {
                    "status": "NOT_AVAILABLE",
                    "calibration_status": "NOT_PROMOTED",
                },
                "limitations": ["Research model; engineer review required."],
            },
            "prediction_json": {"prediction": 650.0},
            "provenance_classification": "PREDICTED_VALUE",
            "created_at": "2026-11-15T00:00:00+00:00",
        },
        {
            "prediction_id": "PRED-REPORT-002",
            "material_id": "MATX-REPORT-PRED-001",
            "target_property": "inclusion_classification",
            "prediction_type": "CLASSIFICATION",
            "predicted_value": None,
            "predicted_unit": None,
            "model_name": "inclusion_classifier",
            "model_version": "b" * 64,
            "input_json": {"features": {"Ti": 0.2, "O": 0.1}},
            "applicability_json": {
                "scope": {"dataset": "Ti-ULC EDS"},
            },
            "prediction_json": {
                "predicted_class": 1,
                "classification": "Inclusion detected",
                "probabilities": {"0": 0.03, "1": 0.97},
            },
            "provenance_classification": "PREDICTED_VALUE",
            "created_at": "2026-11-15T00:01:00+00:00",
        },
    ]
    calls = []
    def fake_list_records(table, filters=None, limit=100, offset=0, db_path=None):
        calls.append({
            "table": table,
            "filters": filters,
            "limit": limit,
            "offset": offset,
        })
        assert table == "prediction_records"
        return records
    monkeypatch.setattr(
        engineering_report_service,
        "list_scientific_records",
        fake_list_records,
    )
    response = client.post(
        "/api/v1/engineering/report",
        json={
            "material_id": "MATX-REPORT-PRED-001",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    prediction_results = response.json()["prediction_results"]
    assert prediction_results["status"] == "PERSISTED_PREDICTIONS_INCLUDED"
    assert prediction_results["record_count"] == 2
    assert len(prediction_results["results"]) == 2
    tensile = prediction_results["results"][0]
    assert tensile["prediction_id"] == "PRED-REPORT-001"
    assert tensile["prediction_type"] == "NUMERIC"
    assert tensile["predicted_value"] == 650.0
    assert tensile["model"]["artifact_sha256"] == "a" * 64
    assert tensile["applicability"]["status"] == "IN_DOMAIN"
    assert tensile["predictive_uncertainty"]["status"] == "NOT_AVAILABLE"
    assert tensile["provenance_classification"] == "PREDICTED_VALUE"
    inclusion = prediction_results["results"][1]
    assert inclusion["prediction_type"] == "CLASSIFICATION"
    assert inclusion["prediction"]["predicted_class"] == 1
    assert inclusion["scope"]["dataset"] == "Ti-ULC EDS"
    # Predicted outputs must not be inserted into measured source evidence.
    assert "PREDICTED_VALUE" not in str(
        response.json()["source_evidence"]
    )
    assert calls == [{
        "table": "prediction_records",
        "filters": {"material_id": "MATX-REPORT-PRED-001"},
        "limit": 1000,
        "offset": 0,
    }]
def test_engineering_report_prediction_results_are_material_specific(
    monkeypatch,
):
    from app.services import engineering_report_service
    requested_materials = []
    def fake_list_records(table, filters=None, limit=100, offset=0, db_path=None):
        requested_materials.append(filters["material_id"])
        return []
    monkeypatch.setattr(
        engineering_report_service,
        "list_scientific_records",
        fake_list_records,
    )
    for material_id in ("MATX-REPORT-A", "MATX-REPORT-B"):
        response = client.post(
            "/api/v1/engineering/report",
            json={
                "material_id": material_id,
                "material_family": "low_alloy_steel",
            },
        )
        assert response.status_code == 200
        assert response.json()["prediction_results"]["results"] == []
    assert requested_materials == ["MATX-REPORT-A", "MATX-REPORT-B"]
