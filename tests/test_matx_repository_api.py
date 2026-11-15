import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import initialize_matx_database
import app.services.matx_repository_service as repository
@pytest.fixture
def client(tmp_path, monkeypatch):
    database = tmp_path / "scientific_api.sqlite3"
    initialize_evidence_registry(database)
    initialize_matx_database(database)
    monkeypatch.setattr(repository, "EVIDENCE_DB_PATH", database)
    with TestClient(app) as test_client:
        yield test_client
def test_create_and_retrieve_material(client):
    response = client.post(
        "/api/v1/scientific-records/materials",
        json={
            "material_id": "API-MAT-001",
            "material_family": "Engineering steel",
            "grade": "AISI 1045",
            "metadata_json": {"origin": "laboratory"},
        },
    )
    assert response.status_code == 201
    assert response.json()["material_id"] == "API-MAT-001"
    assert response.json()["metadata_json"]["origin"] == "laboratory"
    retrieved = client.get(
        "/api/v1/scientific-records/materials/API-MAT-001"
    )
    assert retrieved.status_code == 200
    assert retrieved.json()["grade"] == "AISI 1045"
def test_list_records_with_filters_and_pagination(client):
    for identifier, family in [
        ("API-MAT-002", "Steel"),
        ("API-MAT-003", "Steel"),
        ("API-MAT-004", "Aluminium"),
    ]:
        result = client.post(
            "/api/v1/scientific-records/materials",
            json={
                "material_id": identifier,
                "material_family": family,
            },
        )
        assert result.status_code == 201
    response = client.get(
        "/api/v1/scientific-records/materials",
        params={
            "filters_json": '{"material_family":"Steel"}',
            "limit": 1,
            "offset": 0,
        },
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["material_family"] == "Steel"
def test_unknown_table_and_missing_record(client):
    unknown = client.get("/api/v1/scientific-records/unknown_table")
    assert unknown.status_code == 422
    missing = client.get(
        "/api/v1/scientific-records/materials/DOES-NOT-EXIST"
    )
    assert missing.status_code == 404
def test_foreign_key_rejection_and_invalid_filters(client):
    invalid = client.post(
        "/api/v1/scientific-records/investigations",
        json={
            "investigation_id": "API-INV-001",
            "material_id": "MISSING-MATERIAL",
            "investigation_goal": "Reconstruct material state",
        },
    )
    assert invalid.status_code == 422
    filters = client.get(
        "/api/v1/scientific-records/materials",
        params={"filters_json": "[]"},
    )
    assert filters.status_code == 422
