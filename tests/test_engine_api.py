from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["service"] == "MAT-X API"
def test_fatigue_response_endpoint():
    response = client.get("/api/v1/fatigue/response")
    assert response.status_code == 200
    data = response.json()
    assert data["cycle_feature_records"] == 562
    assert data["independent_specimens"] == 7
    assert len(data["specimen_results"]) == 7
    for specimen in data["specimen_results"]:
        assert 0 < specimen["cycle_coverage_fraction"] <= 1
        assert specimen["independent_specimen"] is True
        assert specimen["loop_area_status"] == "PROVISIONAL_DERIVED_VALUE"
    assert data["status"] == (
        "RESPONSE_ANALYSIS_ONLY_NOT_FATIGUE_LIFE_VALIDATED"
    )
def test_microstructure_map_listing():
    response = client.get("/api/v1/microstructure/maps")
    assert response.status_code == 200
    data = response.json()
    assert data["map_count"] == 5
    assert len(data["maps"]) == 5
def test_eos_microstructure_features():
    response = client.get(
        "/api/v1/microstructure/maps/"
        "e73a32f7c191e735/features"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_points"] == 2160000
    assert data["indexed_points"] == 2092827
    assert "phase_fractions_all_points" in data
    assert "phase_fractions_indexed_points" in data
    assert "indexing_error_counts" in data
    indexed_fraction_sum = sum(
        item["fraction"]
        for item in data["phase_fractions_indexed_points"].values()
    )
    assert abs(indexed_fraction_sum - 1.0) < 1e-6
    error_count_sum = sum(
        item["point_count"]
        for item in data["indexing_error_counts"].values()
    )
    assert error_count_sum == data["total_points"]
def test_registered_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()["registered_models"]
    expected = [
        "tensile_strength",
        "yield_strength",
        "inclusion_classifier"
    ]
    for name in expected:
        assert models[name]["status"] == "available"
def test_tensile_prediction():
    payload = {
        "C": 0.30,
        "Mn": 0.80,
        "Cr": 1.0,
        "Mo": 0.20,
        "Ni": 0.10,
        "Si": 0.25,
        "V": 0.0,
        "Cu": 0.0,
        "Al": 0.02,
        "austenitize_T": 850,
        "temper_T": 550,
        "quench_medium": "Oil",
        "steel_family": "low_alloy"
    }
    response = client.post(
        "/api/v1/predict/tensile-strength",
        json=payload
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))
def test_yield_prediction():
    payload = {
        "C": 0.30,
        "Mn": 0.80,
        "Cr": 1.0,
        "Mo": 0.20,
        "Ni": 0.10,
        "Si": 0.25,
        "V": 0.0,
        "Cu": 0.0,
        "Al": 0.02,
        "austenitize_T": 850,
        "temper_T": 550,
        "quench_medium": "Oil",
        "steel_family": "low_alloy"
    }
    response = client.post(
        "/api/v1/predict/yield-strength",
        json=payload
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))
def test_inclusion_prediction():
    features = [
        "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "S",
        "K", "Ca", "Ti", "Cr", "Mn", "Fe", "La", "Ce"
    ]
    payload = {
        "features": {name: 0.0 for name in features}
    }
    response = client.post(
        "/api/v1/predict/inclusion",
        json=payload
    )
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] in [0, 1]
    assert "probabilities" in result
    assert "validation" in result

def test_active_investigation_partial_evidence():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-TEST-002",
            "material_family": "low_alloy_steel",
            "evidence": {
                "chemical_composition": {"C": 0.25, "Mn": 0.8}
            },
            "uncertainty": {"microstructure": 0.9},
            "top_k": 5
        }
    )
    assert response.status_code == 200
    result = response.json()
    summary = result["evidence_summary"]
    assert summary["supported_categories"] == 7
    assert summary["provided_categories"] == 1
    assert summary["available_categories"] == 1
    assert len(summary["missing_categories"]) == 6
    assert result["recommendation_count"] == 5
def test_active_investigation_rejects_unknown_evidence():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-TEST-003",
            "material_family": "low_alloy_steel",
            "evidence": {"unknown_category": True},
            "uncertainty": {},
            "top_k": 5
        }
    )
    assert response.status_code == 422
def test_material_identity_candidate_endpoint():
    response = client.post(
        "/api/v1/material/identity/candidates",
        json={
            "composition": {
                "C": 0.02,
                "Mn": 1.0,
                "Cr": 20.0,
                "Mo": 6.5,
                "Ni": 25.0,
                "Si": 0.5,
                "V": 0.0,
                "Cu": 1.0
            },
            "top_k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "MAT-X Material Identity Engine"
    assert data["status"] == "CANDIDATE_MATCHING_ONLY"
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["grade_id"] == "EMK0005"
    assert data["methodology"]["definitive_identification"] is False
def test_material_identity_rejects_insufficient_composition():
    response = client.post(
        "/api/v1/material/identity/candidates",
        json={"composition": {"C": 0.2, "Mn": 1.0}}
    )
    assert response.status_code == 422
def test_material_identity_rejects_unknown_element():
    response = client.post(
        "/api/v1/material/identity/candidates",
        json={
            "composition": {
                "C": 0.2,
                "Mn": 1.0,
                "Xx": 0.5
            }
        }
    )
    assert response.status_code == 422
def test_process_history_endpoint():
    response = client.get("/api/v1/process/history")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "MAT-X Process History Reconstruction Engine"
    assert data["status"] == "DOCUMENTED_PROCESS_EVIDENCE_ONLY"
    assert data["total_records"] == 1360
    assert data["unique_heats"] == 1360
    assert data["methodology"]["missing_values_imputed"] is False
    assert data["methodology"]["process_conditions_predicted"] is False
def test_process_history_field_availability():
    response = client.get("/api/v1/process/history")
    assert response.status_code == 200
    fields = response.json()["field_availability"]
    assert fields["austenitize_T"]["available"] == 688
    assert fields["quench_medium"]["available"] == 638
    assert fields["temper_T"]["available"] == 388
def test_process_history_analysis_endpoint():
    response = client.get("/api/v1/process/history/analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "MAT-X Process History Pattern Analysis"
    assert data["status"] == "OBSERVED_PROCESS_COMBINATIONS_ONLY"
    assert data["total_heat_records"] == 1360
    assert data["steel_families"] == 17
    assert data["unique_grades"] == 562
    assert data["complete_route_heat_records"] == 388
    assert len(data["top_observed_process_combinations"]) > 0
    assert len(data["steel_family_summary"]) == 17
    assert data["methodology"]["missing_values_imputed"] is False
def test_process_history_analysis_preserves_missing_evidence():
    response = client.get("/api/v1/process/history/analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["methodology"]["unknown_conditions_predicted"] is False
    assert data["methodology"]["chronological_sequence_assumed"] is False
def test_process_history_heat_lookup():
    response = client.get(
        "/api/v1/process/history/heat/emk_EMK0005_0"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["lookup_type"] == "HEAT_ID"
    assert data["heat_id"] == "emk_EMK0005_0"
    assert data["record_count"] == 1
    assert data["records"][0]["grade_id"] == "EMK0005"
    assert data["records"][0]["austenitize_T"] == 1050.0
    assert data["records"][0]["quench_medium"] == "water"
    assert data["records"][0]["temper_T"] is None
def test_process_history_grade_lookup():
    response = client.get(
        "/api/v1/process/history/grade/EMK0005"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["lookup_type"] == "GRADE_ID"
    assert data["grade_id"] == "EMK0005"
    assert data["record_count"] >= 1
def test_process_history_unknown_heat_returns_404():
    response = client.get(
        "/api/v1/process/history/heat/UNKNOWN_HEAT_ID"
    )
    assert response.status_code == 404
def test_process_history_unknown_grade_returns_404():
    response = client.get(
        "/api/v1/process/history/grade/UNKNOWN_GRADE_ID"
    )
    assert response.status_code == 404
def test_microstructure_provenance_endpoint():
    response = client.get("/api/v1/microstructure/provenance")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "MAT-X EBSD Characterization Provenance Engine"
    assert data["files_discovered"] == 5
    assert data["files_extracted"] == 5
    assert data["failed_files"] == 0
    assert len(data["records"]) == 5
    record = data["records"][0]
    metadata = record["characterization_metadata"]
    assert metadata["manufacturer"]["value"] == "Oxford Instruments"
    assert isinstance(metadata["beam_voltage"]["value"], (int, float))
    assert metadata["beam_voltage"]["unit"] == "kV"
    assert record["manufacturing_history_established"] is False
    assert record["missing_values_imputed"] is False
def test_microstructure_provenance_preserves_source_paths():
    response = client.get("/api/v1/microstructure/provenance")
    assert response.status_code == 200
    data = response.json()
    for record in data["records"]:
        metadata = record["characterization_metadata"]
        assert metadata["acquisition_date"]["source_path"]
        assert metadata["beam_voltage"]["source_path"]
        assert record["source"]["sha256"]


def test_unified_microstructure_record_endpoint():
    response = client.get(
        "/api/v1/microstructure/maps/"
        "e73a32f7c191e735/unified"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "MAT-X Unified Microstructure Record Engine"
    assert data["status"] == "UNIFIED_RECORD_GENERATED"
    assert data["map_id"] == "e73a32f7c191e735"
    features = data["microstructure_features"]
    provenance = data["characterization_provenance"]
    assert features["map_id"] == data["map_id"]
    assert features["total_points"] == 2160000
    assert provenance["status"] == "CHARACTERIZATION_METADATA_EXTRACTED"
    assert provenance["source"]["sha256"]
    assert data["scientific_scope"]["manufacturing_history_established"] is False
    assert data["scientific_scope"]["missing_values_imputed"] is False
def test_unified_microstructure_invalid_map_id():
    response = client.get(
        "/api/v1/microstructure/maps/invalid/unified"
    )
    assert response.status_code == 422
def test_unified_microstructure_unknown_map_id():
    response = client.get(
        "/api/v1/microstructure/maps/"
        "0000000000000000/unified"
    )
    assert response.status_code == 404
def test_fatigue_response_get_does_not_modify_analysis_files():
    from pathlib import Path
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "training_results"
        / "fatigue"
    )
    output_files = [
        output_dir / "fatigue_response_analysis.csv",
        output_dir / "fatigue_response_analysis.json",
    ]
    before = {
        file: file.read_bytes() if file.exists() else None
        for file in output_files
    }
    response = client.get("/api/v1/fatigue/response")
    assert response.status_code == 200
    after = {
        file: file.read_bytes() if file.exists() else None
        for file in output_files
    }
    assert after == before, (
        "GET /fatigue/response must not create or modify analysis files."
    )


def test_material_identity_schema_rejects_invalid_composition_values():
    import pytest
    from pydantic import ValidationError
    from app.schemas.material_identity import MaterialIdentityInput
    invalid = (
        {"C": -0.01, "Mn": 1.0, "Cr": 1.0},
        {"C": 100.01, "Mn": 1.0, "Cr": 1.0},
        {"C": True, "Mn": 1.0, "Cr": 1.0},
        {"C": float("inf"), "Mn": 1.0, "Cr": 1.0},
        {"C": 10 ** 1000, "Mn": 1.0, "Cr": 1.0},
    )
    for composition in invalid:
        with pytest.raises((ValidationError, ValueError)):
            MaterialIdentityInput.model_validate({
                "composition": composition
            })
def test_material_identity_schema_allows_zero_and_unmeasured_elements():
    from app.schemas.material_identity import MaterialIdentityInput
    result = MaterialIdentityInput.model_validate({
        "composition": {"C": 0.0, "Mn": None, "Cr": 1.2},
    })
    assert result.composition == {"C": 0.0, "Mn": None, "Cr": 1.2}

def test_active_investigation_api_rejects_unknown_uncertainty_category():
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-API-INVALID-UNCERTAINTY",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "uncertainty": {"unknown_category": 0.5},
        },
    )
    assert response.status_code == 422
    assert "Unsupported uncertainty categories" in response.text

def test_prediction_openapi_documents_service_unavailable():
    from app.main import app
    schema = app.openapi()
    endpoints = (
        "/api/v1/predict/tensile-strength",
        "/api/v1/predict/yield-strength",
        "/api/v1/predict/inclusion",
    )
    for endpoint in endpoints:
        operation = schema["paths"][endpoint]["post"]
        assert "200" in operation["responses"]
        assert "422" in operation["responses"]
        assert "503" in operation["responses"]
        assert (
            "Prediction model or persistence service unavailable"
            in operation["responses"]["503"]["description"]
        )

def test_inclusion_prediction_persists_and_can_be_retrieved(
    tmp_path, monkeypatch
):
    import sqlite3
    from app import api
    from app.api import routes
    from app.services import prediction_persistence_service
    from app.services.evidence_registry_service import initialize_evidence_registry
    from app.services.matx_database_service import initialize_matx_database
    from app.services.matx_repository_service import (
        create_scientific_record,
        get_scientific_record,
    )
    db = tmp_path / "api_prediction.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    create_scientific_record(
        "materials",
        {
            "material_id": "MATX-API-PERSIST-001",
            "material_family": "Engineering steel",
        },
        db_path=db,
    )
    original_persist = prediction_persistence_service.persist_prediction
    def isolated_persist(**kwargs):
        kwargs["db_path"] = db
        return original_persist(**kwargs)
    monkeypatch.setattr(routes, "persist_prediction", isolated_persist)
    features = {
        "C": 0.12,
        "N": 0.01,
        "O": 0.02,
        "F": 0.01,
        "Na": 0.01,
        "Mg": 0.01,
        "Al": 0.10,
        "Si": 0.15,
        "S": 0.01,
        "K": 0.01,
        "Ca": 0.02,
        "Ti": 0.10,
        "Cr": 0.10,
        "Mn": 0.10,
        "Fe": 99.0,
        "La": 0.01,
        "Ce": 0.01,
    }
    response = client.post(
        "/api/v1/predict/inclusion",
        json={
            "material_id": "MATX-API-PERSIST-001",
            "features": features,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    persistence = body["persistence"]
    assert persistence["status"] == "PERSISTED"
    assert persistence["prediction_id"]
    assert persistence["material_id"] == "MATX-API-PERSIST-001"
    record = get_scientific_record(
        "prediction_records",
        persistence["prediction_id"],
        db_path=db,
    )
    assert record is not None
    assert record["material_id"] == "MATX-API-PERSIST-001"
    assert record["target_property"] == "inclusion_classification"
    assert record["prediction_type"] == "CLASSIFICATION"
    assert record["model_version"]
    assert record["prediction_json"]["predicted_class"] == body["predicted_class"]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM prediction_records"
        ).fetchone()[0] == 1
def test_api_information_gain_accepts_valid_probability_model():
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-API-IG-001",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "top_k": 7,
            "information_gain_inputs": {
                "microstructure": {
                    "prior_probabilities": [0.5, 0.5],
                    "outcome_probabilities": [0.5, 0.5],
                    "posterior_probabilities": [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                }
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    recommendation = next(
        item for item in body["recommendations"]
        if item["category"] == "microstructure"
    )
    assert recommendation["score_type"] == (
        "NORMALIZED_EXPECTED_SHANNON_INFORMATION_GAIN"
    )
    assert recommendation["score"] == 1.0
    assert recommendation["information_gain"]["expected_information_gain_bits"] == 1.0
def test_api_information_gain_rejects_invalid_probability_model():
    response = client.post(
        "/api/v1/investigation/recommend",
        json={
            "material_id": "MATX-API-IG-INVALID",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "information_gain_inputs": {
                "microstructure": {
                    "prior_probabilities": [0.5, 0.5],
                    "outcome_probabilities": [0.5, 0.5],
                    "posterior_probabilities": [
                        [0.8, 0.2],
                        [0.8, 0.2],
                    ],
                }
            },
        },
    )
    assert response.status_code == 422
    assert "law of total probability" in response.json()["detail"].lower()
def test_api_persist_route_retains_information_gain_inputs(monkeypatch):
    from app.api import routes
    supplied_inputs = {
        "microstructure": {
            "prior_probabilities": [0.5, 0.5],
            "outcome_probabilities": [0.5, 0.5],
            "posterior_probabilities": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        }
    }
    captured = {}
    def fake_persist(**kwargs):
        captured.update(kwargs)
        return {
            "status": "PERSISTED",
            "investigation_id": "MATX-API-IG-PERSIST-001",
            "experiment_ids": [],
        }
    monkeypatch.setattr(
        routes,
        "repository_persist_investigation",
        fake_persist,
    )
    response = client.post(
        "/api/v1/investigation/recommend/persist",
        json={
            "material_id": "MATX-API-IG-PERSIST-001",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "top_k": 7,
            "information_gain_inputs": supplied_inputs,
        },
    )
    assert response.status_code == 200, response.text
    assert captured["initial_state"]["information_gain_inputs"] == supplied_inputs
    assert response.json()["persistence"]["status"] == "PERSISTED"

