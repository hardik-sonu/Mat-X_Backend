import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import initialize_matx_database
from app.services.matx_repository_service import (
    MatxRepositoryError,
    create_scientific_record,
    get_scientific_record,
    update_lifecycle_record,
)
import app.services.matx_repository_service as repository
@pytest.fixture
def database(tmp_path):
    path = tmp_path / "lifecycle.sqlite3"
    initialize_evidence_registry(path)
    initialize_matx_database(path)
    create_scientific_record(
        "materials",
        {"material_id": "LIFE-MAT-001", "material_family": "Steel"},
        path,
    )
    create_scientific_record(
        "investigations",
        {
            "investigation_id": "LIFE-INV-001",
            "material_id": "LIFE-MAT-001",
            "investigation_goal": "Reconstruct engineering steel state",
        },
        path,
    )
    create_scientific_record(
        "experiments",
        {
            "experiment_id": "LIFE-EXP-001",
            "material_id": "LIFE-MAT-001",
            "investigation_id": "LIFE-INV-001",
            "experiment_type": "HARDNESS_TEST",
            "objective": "Measure hardness",
        },
        path,
    )
    return path
def test_investigation_lifecycle_and_completion_timestamp(database):
    updated = update_lifecycle_record(
        "investigations",
        "LIFE-INV-001",
        {
            "status": "IN_PROGRESS",
            "limitations_json": ["No phase identification yet"],
        },
        database,
    )
    assert updated["status"] == "IN_PROGRESS"
    assert updated["limitations_json"] == ["No phase identification yet"]
    assert updated["completed_at"] is None
    completed = update_lifecycle_record(
        "investigations",
        "LIFE-INV-001",
        {
            "status": "COMPLETED",
            "final_state_json": {
                "grade": "AISI 1045",
                "confidence": "LIMITED",
            },
        },
        database,
    )
    assert completed["status"] == "COMPLETED"
    assert completed["final_state_json"]["grade"] == "AISI 1045"
    assert completed["completed_at"] is not None
def test_experiment_lifecycle_and_result_persistence(database):
    started = update_lifecycle_record(
        "experiments",
        "LIFE-EXP-001",
        {"status": "IN_PROGRESS"},
        database,
    )
    assert started["status"] == "IN_PROGRESS"
    assert started["completed_at"] is None
    completed = update_lifecycle_record(
        "experiments",
        "LIFE-EXP-001",
        {
            "status": "COMPLETED",
            "result_json": {
                "hardness": 215,
                "unit": "HV",
                "method": "Vickers",
            },
            "provenance_classification": "SOURCE_VALUE",
        },
        database,
    )
    assert completed["status"] == "COMPLETED"
    assert completed["result_json"]["hardness"] == 215
    assert completed["completed_at"] is not None
def test_completion_requires_final_scientific_result(database):
    with pytest.raises(MatxRepositoryError, match="final_state_json"):
        update_lifecycle_record(
            "investigations",
            "LIFE-INV-001",
            {"status": "COMPLETED"},
            database,
        )
    with pytest.raises(MatxRepositoryError, match="result_json"):
        update_lifecycle_record(
            "experiments",
            "LIFE-EXP-001",
            {"status": "COMPLETED"},
            database,
        )
def test_invalid_transitions_and_terminal_records_are_rejected(database):
    with pytest.raises(MatxRepositoryError, match="Invalid"):
        update_lifecycle_record(
            "investigations",
            "LIFE-INV-001",
            {"status": "FAILED"},
            database,
        )
    update_lifecycle_record(
        "investigations",
        "LIFE-INV-001",
        {
            "status": "COMPLETED",
            "final_state_json": {"state": "recorded"},
        },
        database,
    )
    with pytest.raises(MatxRepositoryError, match="Invalid"):
        update_lifecycle_record(
            "investigations",
            "LIFE-INV-001",
            {"status": "IN_PROGRESS"},
            database,
        )
def test_only_lifecycle_tables_and_fields_can_be_updated(database):
    with pytest.raises(MatxRepositoryError, match="only for investigations"):
        update_lifecycle_record(
            "materials",
            "LIFE-MAT-001",
            {"status": "IN_PROGRESS"},
            database,
        )
    with pytest.raises(MatxRepositoryError, match="cannot be changed"):
        update_lifecycle_record(
            "investigations",
            "LIFE-INV-001",
            {
                "status": "IN_PROGRESS",
                "material_id": "OTHER-MATERIAL",
            },
            database,
        )
    with pytest.raises(MatxRepositoryError, match="completed_at"):
        update_lifecycle_record(
            "investigations",
            "LIFE-INV-001",
            {"completed_at": "2026-11-15T00:00:00Z"},
            database,
        )
    material = get_scientific_record(
        "materials", "LIFE-MAT-001", database
    )
    assert material["material_family"] == "Steel"
@pytest.fixture
def api_client(database, monkeypatch):
    monkeypatch.setattr(repository, "EVIDENCE_DB_PATH", database)
    with TestClient(app) as client:
        yield client
def test_lifecycle_patch_api_success_and_not_found(api_client):
    response = api_client.patch(
        "/api/v1/scientific-records/investigations/"
        "LIFE-INV-001/lifecycle",
        json={
            "status": "IN_PROGRESS",
            "limitations_json": ["Awaiting microscopy"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "IN_PROGRESS"
    missing = api_client.patch(
        "/api/v1/scientific-records/investigations/"
        "UNKNOWN/lifecycle",
        json={"status": "IN_PROGRESS"},
    )
    assert missing.status_code == 404
def test_lifecycle_patch_api_rejects_invalid_table_and_transition(api_client):
    invalid_table = api_client.patch(
        "/api/v1/scientific-records/materials/"
        "LIFE-MAT-001/lifecycle",
        json={"status": "IN_PROGRESS"},
    )
    assert invalid_table.status_code == 422
    invalid_transition = api_client.patch(
        "/api/v1/scientific-records/experiments/"
        "LIFE-EXP-001/lifecycle",
        json={"status": "COMPLETED"},
    )
    assert invalid_transition.status_code == 422

def test_persist_investigation_recommendations_creates_planned_experiments(database):
    from app.services.matx_repository_service import (
        persist_investigation_recommendations,
    )
    recommendations = [
        {
            "category": "chemical_composition",
            "method": "Optical emission spectroscopy",
            "reason": "Composition evidence is missing.",
            "score": 0.65,
            "score_type": "HEURISTIC_INVESTIGATION_PRIORITY",
            "suggested_techniques": ["OES"],
        },
        {
            "category": "microstructure",
            "method": "Optical microscopy",
            "reason": "Microstructure evidence is missing.",
            "score": 0.65,
            "score_type": "HEURISTIC_INVESTIGATION_PRIORITY",
            "suggested_techniques": ["Optical microscopy"],
        },
    ]
    saved = persist_investigation_recommendations(
        material_id="LIFE-MAT-001",
        investigation_goal="material_state_reconstruction",
        initial_state={
            "material_family": "Steel",
            "evidence": {},
            "uncertainty": {},
        },
        limitations=["Heuristic recommendations only."],
        recommendations=recommendations,
        db_path=database,
    )
    assert saved["investigation_status"] == "OPEN"
    assert saved["experiment_count"] == 2
    investigation = get_scientific_record(
        "investigations", saved["investigation_id"], database
    )
    assert investigation["status"] == "OPEN"
    assert investigation["initial_state_json"]["material_family"] == "Steel"
    for item in saved["experiments"]:
        experiment = get_scientific_record(
            "experiments", item["experiment_id"], database
        )
        assert experiment["investigation_id"] == saved["investigation_id"]
        assert experiment["status"] == "PLANNED"
        assert experiment["result_json"] is None
        assert experiment["provenance_classification"] is None
def test_persist_investigation_rejects_unknown_material(database):
    from app.services.matx_repository_service import (
        persist_investigation_recommendations,
    )
    with pytest.raises(MatxRepositoryError, match="Material not found"):
        persist_investigation_recommendations(
            material_id="UNKNOWN-MATERIAL",
            investigation_goal="material_state_reconstruction",
            initial_state={},
            limitations=[],
            recommendations=[{
                "category": "microstructure",
                "method": "Optical microscopy",
            }],
            db_path=database,
        )
    records = __import__(
        "app.services.matx_repository_service",
        fromlist=["list_scientific_records"],
    ).list_scientific_records("investigations", db_path=database)
    assert len(records) == 1
def test_persist_investigation_rolls_back_all_rows_on_experiment_failure(database):
    from app.services.matx_repository_service import (
        persist_investigation_recommendations,
        list_scientific_records,
    )
    import sqlite3
    connection = sqlite3.connect(database)
    try:
        connection.execute("""
            CREATE TRIGGER reject_proposed_experiment
            BEFORE INSERT ON experiments
            BEGIN
                SELECT RAISE(ABORT, 'forced experiment insert failure');
            END
        """)
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(MatxRepositoryError, match="Database rejected"):
        persist_investigation_recommendations(
            material_id="LIFE-MAT-001",
            investigation_goal="material_state_reconstruction",
            initial_state={},
            limitations=[],
            recommendations=[{
                "category": "microstructure",
                "method": "Optical microscopy",
            }],
            db_path=database,
        )
    assert len(list_scientific_records("investigations", db_path=database)) == 1
    assert len(list_scientific_records("experiments", db_path=database)) == 1
def test_recommendation_persist_api_and_existing_endpoint(api_client):
    payload = {
        "material_id": "LIFE-MAT-001",
        "material_family": "Steel",
        "evidence": {},
        "uncertainty": {},
        "top_k": 2,
    }
    before = len(repository.list_scientific_records(
        "investigations", db_path=repository.EVIDENCE_DB_PATH
    ))
    ordinary = api_client.post("/api/v1/investigation/recommend", json=payload)
    assert ordinary.status_code == 200
    assert len(repository.list_scientific_records(
        "investigations", db_path=repository.EVIDENCE_DB_PATH
    )) == before
    persisted = api_client.post(
        "/api/v1/investigation/recommend/persist", json=payload
    )
    assert persisted.status_code == 200, persisted.text
    body = persisted.json()
    assert body["persistence"]["investigation_status"] == "OPEN"
    assert body["persistence"]["experiment_count"] == 2
    assert all(
        item["status"] == "PLANNED"
        and item["result_json"] is None
        for item in body["persistence"]["experiments"]
    )
    missing = dict(payload, material_id="UNKNOWN-MATERIAL")
    not_found = api_client.post(
        "/api/v1/investigation/recommend/persist", json=missing
    )
    assert not_found.status_code == 404
