from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_unsupported_reference_family_does_not_abort_reconstruction():
    payload = {
        "material_id": "MATX-TEST-STATE-REGRESSION",
        "material_family": "low_alloy_steel",
        "composition": {
            "C": 0.30,
            "Mn": 0.80,
            "Cr": 1.0,
            "Mo": 0.20,
        },
        "measured_properties": {
            "tensile_strength": 720,
            "yield_strength": 560,
        },
    }
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    identity = data["material_identity_assessment"]
    assert data["status"] == "EVIDENCE_AGGREGATED_NOT_CERTIFIED"
    assert identity["status"] == "REFERENCE_FAMILY_UNSUPPORTED"
    assert identity["material_family_requested"] == "low_alloy_steel"
    assert identity["recommendations"] == []
    assert identity["methodology"]["reference_filter_applied"] is True
    assert identity["methodology"]["candidate_matching_performed"] is False
    assert "chemical_composition" in (
        data["evidence_summary"]["available_category_names"]
    )
    assert "mechanical_properties" in (
        data["evidence_summary"]["available_category_names"]
    )
    assert (
        data["active_investigation"]["status"]
        == "HEURISTIC_RECOMMENDATION_NOT_EXPERIMENTALLY_VALIDATED"
    )
from app.services import material_state_service
def test_process_history_distinguishes_no_match_from_matched_record(
    monkeypatch,
):
    monkeypatch.setattr(
        material_state_service,
        "get_heat_process_history",
        lambda heat_id: None,
    )
    monkeypatch.setattr(
        material_state_service,
        "get_grade_process_history",
        lambda grade_id: {
            "record_count": 1,
            "records": [{"grade_id": grade_id}],
        },
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-PROCESS-LOOKUP-TEST",
            "material_family": "low_alloy_steel",
            "heat_id": "MISSING-HEAT",
            "grade_id": "GRADE-001",
        },
    )
    assert response.status_code == 200
    process = response.json()["evidence"]["process_history"]["value"]
    assert process["heat_lookup_status"] == "NO_MATCH"
    assert process["grade_lookup_status"] == "MATCHED"
    assert process["data_source_status"] == "AVAILABLE"
def test_process_history_reports_unavailable_dataset(monkeypatch):
    def unavailable(_):
        raise FileNotFoundError("Process dataset unavailable.")
    monkeypatch.setattr(
        material_state_service,
        "get_heat_process_history",
        unavailable,
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-PROCESS-UNAVAILABLE-TEST",
            "material_family": "low_alloy_steel",
            "heat_id": "HEAT-001",
        },
    )
    assert response.status_code == 200
    process = response.json()["evidence"]["process_history"]["value"]
    assert process["heat_lookup_status"] == "DATA_SOURCE_UNAVAILABLE"
    assert (
        process["data_source_status"]
        == "PROCESS_DATA_SOURCE_UNAVAILABLE"
    )

def test_unassessed_damage_condition_is_not_returned_as_evidence():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-DAMAGE-STATUS-TEST",
            "material_family": "low_alloy_steel",
            "damage_condition": "Not assessed",
        },
    )
    assert response.status_code == 200
    result = response.json()
    damage = result["evidence"]["damage_condition"]
    assert damage["status"] == "UNAVAILABLE"
    assert damage["provenance"] == "UNAVAILABLE"
    assert damage["value"] is None
    investigation = result["active_investigation"]
    assert "damage_condition" in investigation["evidence_summary"]["missing_categories"]

def _make_registry_record(**overrides):
    record = {
        "evidence_id": "EVIDENCE-STATE-001",
        "material_id": "MATX-REGISTRY-STATE-TEST",
        "investigation_id": "INV-001",
        "evidence_category": "mechanical_testing",
        "measured_quantity": "hardness",
        "value": 220,
        "unit": "HV",
        "uncertainty_value": 5,
        "measurement_method": "Vickers hardness test",
        "test_standard": None,
        "source_type": "LABORATORY",
        "source_name": "Test Laboratory",
        "source_record_id": None,
        "doi_or_url": None,
        "specimen_id": "SPEC-001",
        "observed_at": None,
        "provenance_classification": "SOURCE_VALUE",
        "validation_status": "VALIDATED",
    }
    record.update(overrides)
    return record
def test_validated_registry_evidence_updates_investigation_availability(
    monkeypatch,
):
    record = _make_registry_record()
    calls = []
    def list_records(**kwargs):
        calls.append(kwargs)
        return [record]
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        list_records,
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-REGISTRY-STATE-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert calls == [{
        "material_id": "MATX-REGISTRY-STATE-TEST",
        "limit": 1000,
    }]
    assert data["evidence_registry"]["record_count"] == 1
    assert data["evidence_registry"]["validated_usable_record_count"] == 1
    assert data["evidence_registry"]["records"][0]["value"] == 220
    investigation_summary = data["active_investigation"]["evidence_summary"]
    assert investigation_summary["available_categories"] == 1
    assert "mechanical_properties" not in (
        investigation_summary["missing_categories"]
    )
def test_unverified_registry_evidence_does_not_update_investigation(
    monkeypatch,
):
    record = _make_registry_record(validation_status="UNVERIFIED")
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        lambda **kwargs: [record],
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-REGISTRY-STATE-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_registry"]["record_count"] == 1
    assert data["evidence_registry"]["validated_usable_record_count"] == 0
    assert "mechanical_properties" in (
        data["active_investigation"]["evidence_summary"]["missing_categories"]
    )
def test_registry_records_for_other_material_are_not_requested(
    monkeypatch,
):
    requested_ids = []
    def list_records(material_id, limit):
        requested_ids.append(material_id)
        return []
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        list_records,
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-ISOLATION-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    assert requested_ids == ["MATX-ISOLATION-TEST"]
    assert response.json()["evidence_registry"]["record_count"] == 0

def test_non_source_or_rejected_registry_records_do_not_update_availability(
    monkeypatch,
):
    invalid_records = [
        _make_registry_record(
            provenance_classification="AI_INTERPRETATION",
        ),
        _make_registry_record(
            validation_status="REJECTED",
        ),
        _make_registry_record(value=None),
    ]
    for record in invalid_records:
        monkeypatch.setattr(
            material_state_service,
            "list_evidence_records",
            lambda **kwargs: [record],
        )
        response = client.post(
            "/api/v1/material/state/reconstruct",
            json={
                "material_id": "MATX-REGISTRY-STATE-TEST",
                "material_family": "low_alloy_steel",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["evidence_registry"]["record_count"] == 1
        assert data["evidence_registry"]["validated_usable_record_count"] == 0
        assert "mechanical_properties" in (
            data["active_investigation"]["evidence_summary"]["missing_categories"]
        )
        assert "mechanical_properties" in data["evidence_summary"]["missing_categories"]

def test_user_reported_damage_text_is_not_validated_evidence():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-USER-DAMAGE-TEST",
            "material_family": "low_alloy_steel",
            "damage_condition": "Visible corrosion reported by operator",
        },
    )
    assert response.status_code == 200
    data = response.json()
    damage = data["evidence"]["damage_condition"]
    assert damage["status"] == "UNAVAILABLE"
    assert damage["provenance"] == "UNAVAILABLE"
    assert damage["user_reported_observation"]["status"] == (
        "USER_REPORTED_UNVERIFIED"
    )
    assert "damage_condition" in (
        data["active_investigation"]["evidence_summary"]["missing_categories"]
    )
def test_validated_damage_registry_record_updates_unified_state(monkeypatch):
    record = _make_registry_record(
        evidence_category="damage_assessment",
        measured_quantity="corrosion_observation",
        value="Surface corrosion observed",
    )
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        lambda **kwargs: [record],
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-REGISTRY-STATE-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["damage_condition_assessment"]["observation_count"] == 1
    assert data["evidence"]["damage_condition"]["status"] == "AVAILABLE"
    assert "damage_condition" not in (
        data["active_investigation"]["evidence_summary"]["missing_categories"]
    )

def test_registry_value_validation_rejects_nonfinite_and_invalid_nested_values():
    invalid_values = [
        float("nan"),
        float("inf"),
        float("-inf"),
        10 ** 1000,
        {},
        [],
        {"hardness": None},
        {"hardness": float("nan")},
        {"hardness": {"value": float("inf")}},
        {"": 220},
        ["unknown"],
        [220, None],
        True,
        "unknown",
        "not assessed",
    ]
    for value in invalid_values:
        record = _make_registry_record(value=value)
        assert not material_state_service._registry_record_has_usable_value(
            record
        ), f"Unexpectedly accepted invalid registry value: {type(value).__name__}"
def test_registry_value_validation_accepts_finite_zero_and_valid_nested_values():
    valid_values = [
        0,
        0.0,
        220,
        "Surface corrosion observed",
        {"hardness": 0, "method": "Vickers"},
        [{"phase": "Ferrite", "fraction": 0.0}],
    ]
    for value in valid_values:
        record = _make_registry_record(value=value)
        assert material_state_service._registry_record_has_usable_value(
            record
        ), f"Unexpectedly rejected valid registry value: {value!r}"
def test_unavailable_user_supplied_categories_have_unavailable_provenance():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-PROVENANCE-EMPTY",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    for category in ("chemical_composition", "mechanical_properties"):
        assert evidence[category]["status"] == "UNAVAILABLE"
        assert evidence[category]["provenance"] == "UNAVAILABLE"
        assert evidence[category]["value"] is None
def test_populated_user_supplied_categories_keep_source_provenance():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-PROVENANCE-POPULATED",
            "material_family": "low_alloy_steel",
            "composition": {"C": 0.25},
            "measured_properties": {"hardness": 220},
        },
    )
    assert response.status_code == 200
    evidence = response.json()["evidence"]
    for category in ("chemical_composition", "mechanical_properties"):
        assert evidence[category]["status"] == "AVAILABLE"
        assert evidence[category]["provenance"] == "SOURCE_VALUE"
        assert evidence[category]["source"] == "USER_SUPPLIED"
def test_validated_mechanical_property_alias_is_exposed_in_material_state(
    monkeypatch,
):
    record = _make_registry_record(
        evidence_category="mechanical_property",
        evidence_id="EVIDENCE-MECHANICAL-ALIAS-001",
        value=250,
        unit="HV",
        uncertainty_value=3,
    )
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        lambda **kwargs: [record],
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-MECHANICAL-ALIAS-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    registry = data["evidence_registry"]
    mechanical = data["evidence"]["mechanical_properties"]
    assert registry["mapped_category_count"] == 1
    assert registry["validated_usable_record_count"] == 1
    assert registry["records"][0]["evidence_category"] == "mechanical_property"
    assert mechanical["status"] == "AVAILABLE"
    assert mechanical["source"] == "VALIDATED_EVIDENCE_REGISTRY"
    assert mechanical["provenance"] == "SOURCE_VALUE"
    assert mechanical["unit_status"] == "PER_RECORD_UNITS_AVAILABLE"
    assert mechanical["validated_registry_record_count"] == 1
    assert mechanical["value"][0]["value"] == 250
    assert mechanical["value"][0]["unit"] == "HV"
    assert mechanical["value"][0]["uncertainty_value"] == 3
    assert mechanical["value"][0]["evidence_id"] == (
        "EVIDENCE-MECHANICAL-ALIAS-001"
    )
    assert "mechanical_properties" in (
        data["evidence_summary"]["available_category_names"]
    )
def test_unverified_mechanical_property_alias_remains_unavailable(
    monkeypatch,
):
    record = _make_registry_record(
        evidence_category="mechanical_property",
        validation_status="UNVERIFIED",
    )
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        lambda **kwargs: [record],
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-MECHANICAL-UNVERIFIED-TEST",
            "material_family": "low_alloy_steel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    mechanical = data["evidence"]["mechanical_properties"]
    assert data["evidence_registry"]["mapped_category_count"] == 1
    assert data["evidence_registry"]["validated_usable_record_count"] == 0
    assert mechanical["status"] == "UNAVAILABLE"
    assert mechanical["value"] is None
    assert mechanical["validated_registry_record_count"] == 0
    assert "mechanical_properties" in (
        data["evidence_summary"]["missing_categories"]
    )


def test_material_state_rejects_out_of_range_composition():
    for composition in ({"C": -0.01}, {"C": 100.01}):
        response = client.post(
            "/api/v1/material/state/reconstruct",
            json={
                "material_id": "MATX-INVALID-COMPOSITION",
                "material_family": "low_alloy_steel",
                "composition": composition,
            },
        )
        assert response.status_code == 422
def test_material_state_null_only_composition_is_unavailable():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-NULL-COMPOSITION",
            "material_family": "low_alloy_steel",
            "composition": {"C": None, "Mn": None, "Cr": None},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence"]["chemical_composition"]["status"] == "UNAVAILABLE"
    assert data["evidence"]["chemical_composition"]["value"] is None
    assert data["material_identity_assessment"]["status"] == "INSUFFICIENT_EVIDENCE"
def test_material_state_zero_composition_is_valid_evidence():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-ZERO-COMPOSITION",
            "material_family": "low_alloy_steel",
            "composition": {"C": 0.0},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence"]["chemical_composition"]["status"] == "AVAILABLE"
    assert data["evidence"]["chemical_composition"]["value"] == {"C": 0.0}
def test_material_state_schema_rejects_oversized_numeric_values():
    import pytest
    from pydantic import ValidationError
    from app.schemas.material_state import MaterialStateReconstructionInput
    with pytest.raises((ValidationError, ValueError)):
        MaterialStateReconstructionInput.model_validate({
            "material_id": "MATX-HUGE-COMPOSITION",
            "material_family": "steel",
            "composition": {"C": 10 ** 1000},
        })
    with pytest.raises((ValidationError, ValueError)):
        MaterialStateReconstructionInput.model_validate({
            "material_id": "MATX-HUGE-PROPERTY",
            "material_family": "steel",
            "measured_properties": {"hardness": 10 ** 1000},
        })
def test_material_state_schema_rejects_boolean_composition():
    import pytest
    from pydantic import ValidationError
    from app.schemas.material_state import MaterialStateReconstructionInput
    with pytest.raises((ValidationError, ValueError)):
        MaterialStateReconstructionInput.model_validate({
            "material_id": "MATX-BOOLEAN-COMPOSITION",
            "material_family": "steel",
            "composition": {"C": True},
        })

def test_validated_registry_categories_are_exposed_in_unified_state(
    monkeypatch,
):
    import pytest
    cases = [
        ("chemical_composition", "carbon_content", "chemical_composition"),
        ("microstructure", "microstructure_observation", "microstructure"),
        ("phase_identification", "phase_fraction", "phase_information"),
        ("grain_characterization", "mean_grain_size", "grain_characterization"),
        ("process_history", "heat_treatment", "process_history"),
    ]
    for registry_category, quantity, state_category in cases:
        record = _make_registry_record(
            evidence_category=registry_category,
            measured_quantity=quantity,
            value={"observed_value": 12.5},
        )
        monkeypatch.setattr(
            material_state_service,
            "list_evidence_records",
            lambda **kwargs: [record],
        )
        response = client.post(
            "/api/v1/material/state/reconstruct",
            json={
                "material_id": "MATX-REGISTRY-CATEGORY-TEST",
                "material_family": "low_alloy_steel",
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        evidence = data["evidence"][state_category]
        assert evidence["status"] == "AVAILABLE"
        assert evidence["source"] == "VALIDATED_EVIDENCE_REGISTRY"
        assert evidence["validated_registry_record_count"] == 1
        if state_category == "process_history":
            assert len(
                evidence["value"]["validated_registry_records"]
            ) == 1
            assert evidence["value"]["heat_lookup_status"] == "NOT_REQUESTED"
            assert evidence["value"]["grade_lookup_status"] == "NOT_REQUESTED"
        else:
            assert evidence["value"][0]["value"] == {
                "observed_value": 12.5
            }
        summary_category = (
            "grain_size"
            if state_category == "grain_characterization"
            else state_category
        )
        assert summary_category in (
            data["evidence_summary"]["available_category_names"]
        )
def test_user_composition_takes_precedence_over_registry_composition(
    monkeypatch,
):
    record = _make_registry_record(
        evidence_category="chemical_composition",
        measured_quantity="carbon_content",
        value=0.45,
    )
    monkeypatch.setattr(
        material_state_service,
        "list_evidence_records",
        lambda **kwargs: [record],
    )
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-COMPOSITION-PRECEDENCE-TEST",
            "material_family": "low_alloy_steel",
            "composition": {"C": 0.30},
        },
    )
    assert response.status_code == 200, response.text
    evidence = response.json()["evidence"]["chemical_composition"]
    assert evidence["status"] == "AVAILABLE"
    assert evidence["source"] == "USER_SUPPLIED"
    assert evidence["value"] == {"C": 0.30}
    assert evidence["validated_registry_record_count"] == 1

def test_material_state_api_rejects_unknown_uncertainty_category():
    response = client.post(
        "/api/v1/material/state/reconstruct",
        json={
            "material_id": "MATX-STATE-API-INVALID-UNCERTAINTY",
            "material_family": "low_alloy_steel",
            "uncertainty": {"unknown_category": 0.5},
        },
    )
    assert response.status_code == 422
    assert "Unsupported uncertainty categories" in response.text
