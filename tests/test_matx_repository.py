import sqlite3
import pytest
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import initialize_matx_database
from app.services.matx_repository_service import (
    MatxRepositoryError,
    create_scientific_record,
    get_scientific_record,
    list_scientific_records,
)
@pytest.fixture
def database(tmp_path):
    path = tmp_path / "matx_repository.sqlite3"
    initialize_evidence_registry(path)
    initialize_matx_database(path)
    return path
def test_material_create_retrieve_and_json_roundtrip(database):
    created = create_scientific_record(
        "materials",
        {
            "material_id": "MATX-REPO-001",
            "material_family": "Engineering steel",
            "grade": "AISI 1045",
            "metadata_json": {"origin": "laboratory", "tags": ["steel"]},
        },
        database,
    )
    assert created["material_id"] == "MATX-REPO-001"
    assert created["metadata_json"]["tags"] == ["steel"]
    retrieved = get_scientific_record(
        "materials", "MATX-REPO-001", database
    )
    assert retrieved == created
def test_source_and_composition_foreign_key_relationship(database):
    create_scientific_record(
        "materials",
        {"material_id": "MATX-REPO-002"},
        database,
    )
    create_scientific_record(
        "sources",
        {
            "source_id": "SRC-REPO-001",
            "source_type": "LABORATORY",
            "source_name": "Composition test",
            "provenance_classification": "SOURCE_VALUE",
        },
        database,
    )
    record = create_scientific_record(
        "material_compositions",
        {
            "composition_id": "COMP-REPO-001",
            "material_id": "MATX-REPO-002",
            "element": "C",
            "value": 0.45,
            "unit": "wt%",
            "provenance_classification": "SOURCE_VALUE",
            "source_id": "SRC-REPO-001",
        },
        database,
    )
    assert record["value"] == 0.45
    assert record["source_id"] == "SRC-REPO-001"
def test_foreign_key_rejects_unknown_material(database):
    with pytest.raises(MatxRepositoryError, match="Database rejected"):
        create_scientific_record(
            "investigations",
            {
                "investigation_id": "INV-REPO-001",
                "material_id": "UNKNOWN",
                "investigation_goal": "State reconstruction",
            },
            database,
        )
def test_list_filters_and_pagination(database):
    for material_id, family in [
        ("MATX-REPO-003", "Steel"),
        ("MATX-REPO-004", "Steel"),
        ("MATX-REPO-005", "Aluminium"),
    ]:
        create_scientific_record(
            "materials",
            {"material_id": material_id, "material_family": family},
            database,
        )
    results = list_scientific_records(
        "materials",
        filters={"material_family": "Steel"},
        limit=1,
        offset=0,
        db_path=database,
    )
    assert len(results) == 1
    assert results[0]["material_family"] == "Steel"
def test_rejects_unknown_table_and_fields(database):
    with pytest.raises(MatxRepositoryError, match="Unsupported"):
        list_scientific_records("schema_migrations", db_path=database)
    with pytest.raises(MatxRepositoryError, match="Unknown fields"):
        create_scientific_record(
            "materials",
            {"material_id": "MATX-REPO-006", "arbitrary_sql": "x"},
            database,
        )
def test_rejects_nonfinite_values_and_invalid_json(database):
    with pytest.raises(MatxRepositoryError, match="Non-finite"):
        create_scientific_record(
            "materials",
            {
                "material_id": "MATX-REPO-007",
                "metadata_json": {"bad": float("nan")},
            },
            database,
        )
def test_evidence_registry_remains_immutable(database):
    from app.services.evidence_registry_service import (
        create_evidence_record,
    )
    record = create_evidence_record(
        {
            "material_id": "MATX-REPO-008",
            "evidence_category": "mechanical_property",
            "measured_quantity": "hardness",
            "value": 210,
            "unit": "HV",
            "source_type": "LABORATORY",
            "provenance_classification": "SOURCE_VALUE",
        },
        database,
    )
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE evidence_records SET unit = ? WHERE evidence_id = ?",
                ("HB", record["evidence_id"]),
            )
