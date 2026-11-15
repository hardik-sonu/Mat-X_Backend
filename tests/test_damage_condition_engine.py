import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.engines.damage_condition_engine import assess_damage_condition
from app.api import routes
client = TestClient(app)
def record(**overrides):
    item = {
        "evidence_id": "DAMAGE-001",
        "material_id": "MATX-DAMAGE-TEST",
        "evidence_category": "damage_assessment",
        "measured_quantity": "corrosion_observation",
        "value": "Surface corrosion observed",
        "unit": None,
        "uncertainty_value": None,
        "measurement_method": "Visual inspection",
        "test_standard": None,
        "source_type": "LABORATORY",
        "source_name": "Test Lab",
        "source_record_id": None,
        "doi_or_url": None,
        "specimen_id": "SPEC-01",
        "observed_at": None,
        "provenance_classification": "SOURCE_VALUE",
        "validation_status": "VALIDATED",
    }
    item.update(overrides)
    return item
def test_no_records_returns_insufficient_evidence():
    result = assess_damage_condition("MATX-DAMAGE-TEST", [])
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["observation_count"] == 0
    assert result["diagnosis_performed"] is False
def test_validated_damage_observation_is_preserved():
    source = record()
    result = assess_damage_condition("MATX-DAMAGE-TEST", [source])
    assert result["status"] == (
        "OBSERVATIONS_AVAILABLE_REQUIRES_ENGINEERING_REVIEW"
    )
    assert result["observation_count"] == 1
    observation = result["observations"][0]
    assert observation["evidence_id"] == "DAMAGE-001"
    assert observation["value"] == "Surface corrosion observed"
    assert observation["measurement_method"] == "Visual inspection"
    assert observation["provenance_classification"] == "SOURCE_VALUE"
@pytest.mark.parametrize(
    "overrides",
    [
        {"validation_status": "UNVERIFIED"},
        {"validation_status": "REJECTED"},
        {"provenance_classification": "AI_INTERPRETATION"},
        {"provenance_classification": "UNAVAILABLE"},
        {"evidence_category": "mechanical_testing"},
        {"measured_quantity": "hardness"},
        {"value": None},
        {"value": ""},
        {"value": True},
        {"material_id": "ANOTHER-MATERIAL"},
    ],
)
def test_unusable_or_unrelated_records_are_excluded(overrides):
    result = assess_damage_condition(
        "MATX-DAMAGE-TEST",
        [record(**overrides)],
    )
    assert result["observation_count"] == 0
    assert result["excluded_record_count"] == 1
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
def test_mixed_records_only_include_eligible_damage_evidence():
    records = [
        record(),
        record(
            evidence_id="DAMAGE-002",
            validation_status="UNVERIFIED",
        ),
        record(
            evidence_id="DAMAGE-003",
            evidence_category="mechanical_testing",
        ),
    ]
    result = assess_damage_condition("MATX-DAMAGE-TEST", records)
    assert result["observation_count"] == 1
    assert result["excluded_record_count"] == 2
    assert result["observations"][0]["evidence_id"] == "DAMAGE-001"
def test_invalid_engine_arguments_are_rejected():
    with pytest.raises(ValueError):
        assess_damage_condition("", [])
    with pytest.raises(ValueError):
        assess_damage_condition("MATX-DAMAGE-TEST", None)
def test_damage_assessment_api_uses_material_scoped_registry(monkeypatch):
    calls = []
    def fake_list_records(**kwargs):
        calls.append(kwargs)
        return [record()]
    monkeypatch.setattr(routes, "list_evidence_records", fake_list_records)
    response = client.post(
        "/api/v1/damage/assess",
        json={"material_id": "MATX-DAMAGE-TEST"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["observation_count"] == 1
    assert calls == [{
        "material_id": "MATX-DAMAGE-TEST",
        "limit": 1000,
    }]
def test_damage_assessment_api_rejects_unknown_fields():
    response = client.post(
        "/api/v1/damage/assess",
        json={
            "material_id": "MATX-DAMAGE-TEST",
            "damage_severity": "high",
        },
    )
    assert response.status_code == 422

@pytest.mark.parametrize(
    "value",
    [
        10**10000,
        float("inf"),
        float("-inf"),
        float("nan"),
        {"observation": float("nan")},
        {"": "invalid key"},
        ["valid", float("inf")],
        [],
        {},
    ],
    ids=[
        "huge_integer",
        "positive_infinity",
        "negative_infinity",
        "nan",
        "nested_nan",
        "empty_dictionary_key",
        "nested_infinity",
        "empty_list",
        "empty_dictionary",
    ],
)
def test_nonfinite_or_invalid_nested_values_are_excluded(value):
    result = assess_damage_condition(
        "MATX-DAMAGE-TEST",
        [record(value=value)],
    )
    assert result["observation_count"] == 0
    assert result["excluded_record_count"] == 1
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
def test_finite_numeric_damage_value_is_accepted():
    result = assess_damage_condition(
        "MATX-DAMAGE-TEST",
        [record(value=12.5)],
    )
    assert result["observation_count"] == 1
    assert result["observations"][0]["value"] == 12.5
