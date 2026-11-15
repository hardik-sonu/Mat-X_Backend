import math
import sqlite3
import pytest
from app.services.evidence_registry_service import (
    initialize_evidence_registry,
    create_evidence_record,
    get_evidence_record,
    list_evidence_records,
)
@pytest.fixture
def evidence_db(tmp_path):
    db_path = tmp_path / "evidence_test.sqlite3"
    initialize_evidence_registry(db_path)
    return db_path
@pytest.fixture
def valid_payload():
    return {
        "material_id": "MATX-TEST-001",
        "investigation_id": "INV-TEST-001",
        "evidence_category": "mechanical_testing",
        "measured_quantity": "hardness",
        "value": 220,
        "unit": "HV",
        "uncertainty_value": 5,
        "measurement_method": "Vickers hardness test",
        "source_type": "LABORATORY",
        "source_name": "Test Laboratory",
        "provenance_classification": "SOURCE_VALUE",
    }
def test_registry_initializes_with_valid_schema(evidence_db):
    with sqlite3.connect(evidence_db) as connection:
        result = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        columns = connection.execute(
            "PRAGMA table_info(evidence_records)"
        ).fetchall()
    assert result == "ok"
    assert len(columns) == 23
def test_create_and_retrieve_evidence(evidence_db, valid_payload):
    record = create_evidence_record(valid_payload, evidence_db)
    retrieved = get_evidence_record(record["evidence_id"], evidence_db)
    assert retrieved is not None
    assert retrieved["material_id"] == "MATX-TEST-001"
    assert retrieved["value"] == 220
    assert retrieved["unit"] == "HV"
    assert retrieved["uncertainty_value"] == 5
    assert retrieved["provenance_classification"] == "SOURCE_VALUE"
def test_list_evidence_filters_by_material(evidence_db, valid_payload):
    create_evidence_record(valid_payload, evidence_db)
    other = dict(valid_payload)
    other["material_id"] = "MATX-TEST-002"
    create_evidence_record(other, evidence_db)
    records = list_evidence_records(
        material_id="MATX-TEST-001",
        db_path=evidence_db,
    )
    assert len(records) == 1
    assert records[0]["material_id"] == "MATX-TEST-001"
def test_list_evidence_filters_by_investigation(evidence_db, valid_payload):
    create_evidence_record(valid_payload, evidence_db)
    other = dict(valid_payload)
    other["investigation_id"] = "INV-OTHER"
    create_evidence_record(other, evidence_db)
    records = list_evidence_records(
        investigation_id="INV-OTHER",
        db_path=evidence_db,
    )
    assert len(records) == 1
    assert records[0]["investigation_id"] == "INV-OTHER"
def test_database_isolation(tmp_path, valid_payload):
    first_db = tmp_path / "first.sqlite3"
    second_db = tmp_path / "second.sqlite3"
    initialize_evidence_registry(first_db)
    initialize_evidence_registry(second_db)
    record = create_evidence_record(valid_payload, first_db)
    assert get_evidence_record(record["evidence_id"], first_db) is not None
    assert get_evidence_record(record["evidence_id"], second_db) is None
def test_unknown_fields_are_rejected(evidence_db, valid_payload):
    valid_payload["unexpected_field"] = "not allowed"
    with pytest.raises(ValueError, match="Unsupported evidence fields"):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize(
    "field",
    ["material_id", "evidence_category", "measured_quantity"],
)
def test_required_text_fields_cannot_be_blank(
    evidence_db, valid_payload, field
):
    valid_payload[field] = "   "
    with pytest.raises(ValueError):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize(
    "source_type",
    ["UNKNOWN", "", "INFERRED"],
)
def test_invalid_source_type_is_rejected(
    evidence_db, valid_payload, source_type
):
    valid_payload["source_type"] = source_type
    with pytest.raises(ValueError, match="source_type"):
        create_evidence_record(valid_payload, evidence_db)
def test_invalid_provenance_is_rejected(evidence_db, valid_payload):
    valid_payload["provenance_classification"] = "UNCONFIRMED"
    with pytest.raises(ValueError, match="provenance"):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize("uncertainty", [True, -1, math.nan, math.inf, -math.inf])
def test_invalid_uncertainty_is_rejected(
    evidence_db, valid_payload, uncertainty
):
    valid_payload["uncertainty_value"] = uncertainty
    with pytest.raises(ValueError, match="uncertainty_value"):
        create_evidence_record(valid_payload, evidence_db)
def test_non_finite_json_value_is_rejected(evidence_db, valid_payload):
    valid_payload["value"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        create_evidence_record(valid_payload, evidence_db)
def test_unavailable_evidence_cannot_contain_a_value(
    evidence_db, valid_payload
):
    valid_payload["provenance_classification"] = "UNAVAILABLE"
    valid_payload["value"] = 220
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        create_evidence_record(valid_payload, evidence_db)
def test_unavailable_evidence_can_be_recorded_without_value(
    evidence_db, valid_payload
):
    valid_payload["provenance_classification"] = "UNAVAILABLE"
    valid_payload["value"] = None
    valid_payload["unit"] = None
    record = create_evidence_record(valid_payload, evidence_db)
    assert record["value"] is None
    assert record["provenance_classification"] == "UNAVAILABLE"
def test_evidence_records_cannot_be_updated_or_deleted(
    evidence_db, valid_payload
):
    record = create_evidence_record(valid_payload, evidence_db)
    with sqlite3.connect(evidence_db) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE evidence_records SET unit = ? WHERE evidence_id = ?",
                ("MPa", record["evidence_id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
            connection.execute(
                "DELETE FROM evidence_records WHERE evidence_id = ?",
                (record["evidence_id"],),
            )
    assert get_evidence_record(record["evidence_id"], evidence_db) is not None
def test_missing_evidence_returns_none(evidence_db):
    assert get_evidence_record("DOES-NOT-EXIST", evidence_db) is None
def test_invalid_list_limit_is_rejected(evidence_db):
    with pytest.raises(ValueError, match="limit"):
        list_evidence_records(limit=0, db_path=evidence_db)
    with pytest.raises(ValueError, match="limit"):
        list_evidence_records(limit=1001, db_path=evidence_db)
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("unit", 220),
        ("source_name", ["Lab"]),
        ("doi_or_url", True),
        ("specimen_id", {}),
        ("media_type", 123),
    ],
)
def test_optional_metadata_requires_text_or_none(
    evidence_db, valid_payload, field, bad_value
):
    valid_payload[field] = bad_value
    with pytest.raises(ValueError, match=field):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize("status", [True, 1, None, []])
def test_validation_status_requires_string(
    evidence_db, valid_payload, status
):
    valid_payload["validation_status"] = status
    with pytest.raises(ValueError, match="validation_status"):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize("uncertainty", ["5", True, [], {}])
def test_uncertainty_rejects_non_numeric_types(
    evidence_db, valid_payload, uncertainty
):
    valid_payload["uncertainty_value"] = uncertainty
    with pytest.raises(ValueError, match="uncertainty_value"):
        create_evidence_record(valid_payload, evidence_db)
@pytest.mark.parametrize("limit", [True, False, 1.5, "10", None])
def test_list_limit_requires_integer(evidence_db, limit):
    with pytest.raises(ValueError, match="limit"):
        list_evidence_records(limit=limit, db_path=evidence_db)
@pytest.mark.parametrize(
    "field,value",
    [
        ("material_id", 123),
        ("material_id", True),
        ("investigation_id", []),
        ("investigation_id", {}),
    ],
)
def test_list_filters_require_text(
    evidence_db, field, value
):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        list_evidence_records(db_path=evidence_db, **kwargs)
@pytest.mark.parametrize("evidence_id", [None, 123, "", "   "])
def test_get_evidence_requires_nonempty_string(
    evidence_db, evidence_id
):
    with pytest.raises(ValueError, match="evidence_id"):
        get_evidence_record(evidence_id, evidence_db)
def test_nested_json_rejects_non_string_object_keys(
    evidence_db, valid_payload
):
    valid_payload["value"] = {1: "invalid key"}
    with pytest.raises(ValueError, match="keys must be strings"):
        create_evidence_record(valid_payload, evidence_db)
def test_nested_json_rejects_non_finite_numbers(
    evidence_db, valid_payload
):
    valid_payload["value"] = {"hardness": [220, float("inf")]}
    with pytest.raises(ValueError, match="finite JSON"):
        create_evidence_record(valid_payload, evidence_db)
def test_nested_json_round_trip(evidence_db, valid_payload):
    valid_payload["value"] = {
        "hardness": 220,
        "phases": ["ferrite", "pearlite"],
        "metadata": {"method": "Vickers"},
    }
    record = create_evidence_record(valid_payload, evidence_db)
    retrieved = get_evidence_record(record["evidence_id"], evidence_db)
    assert retrieved["value"] == valid_payload["value"]
