import sqlite3
import pytest
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import initialize_matx_database, SCHEMA_VERSION
from app.services.matx_repository_service import create_scientific_record
from app.services import prediction_persistence_service as persistence
def make_database(tmp_path):
    db = tmp_path / "prediction_persistence.sqlite3"
    initialize_evidence_registry(db)
    initialize_matx_database(db)
    create_scientific_record(
        "materials",
        {"material_id": "MATX-PRED-001", "material_family": "Engineering steel"},
        db_path=db,
    )
    return db
def test_schema_v3_preserves_existing_prediction_and_adds_typed_payload(tmp_path):
    db = tmp_path / "v3_migration.sqlite3"
    initialize_evidence_registry(db)
    from app.services.matx_database_service import SCHEMA
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            "BEGIN IMMEDIATE;\n" + SCHEMA + """
            INSERT INTO schema_migrations (version, description)
            VALUES (1, 'Initial MAT-X scientific persistence schema');
            COMMIT;
            """
        )
        connection.execute(
            "INSERT INTO materials (material_id) VALUES (?)",
            ("MATX-V3-001",),
        )
        connection.execute(
            """
            INSERT INTO prediction_records (
                prediction_id, material_id, target_property,
                predicted_value, predicted_unit, model_name, model_version,
                input_json, applicability_json, provenance_classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "PRED-LEGACY-001", "MATX-V3-001", "tensile_strength",
                650.0, None, "legacy_model", "legacy_hash",
                "{}", "{}", "PREDICTED_VALUE",
            ),
        )
    result = initialize_matx_database(db)
    assert result["schema_version"] == SCHEMA_VERSION == 4
    assert result["integrity"] == "ok"
    assert result["foreign_key_errors"] == 0
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            """
            SELECT predicted_value, prediction_type, prediction_json
            FROM prediction_records WHERE prediction_id = ?
            """,
            ("PRED-LEGACY-001",),
        ).fetchone()
        assert row == (650.0, "NUMERIC", "{}")
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,), (3,), (4,)]
def test_prediction_without_material_is_explicitly_skipped(tmp_path):
    db = make_database(tmp_path)
    result = persistence.persist_prediction(
        material_id=None,
        target_property="tensile_strength",
        prediction_type="NUMERIC",
        request_features={"C": 0.2},
        result={"prediction": 650.0, "model": "test_model"},
        model_key="tensile_strength",
        db_path=db,
    )
    assert result == {
        "status": "SKIPPED_NO_MATERIAL_ID",
        "prediction_id": None,
        "material_id": None,
    }
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM prediction_records"
        ).fetchone()[0] == 0
def test_unknown_material_is_rejected(tmp_path):
    db = make_database(tmp_path)
    with pytest.raises(ValueError, match="Unknown material_id"):
        persistence.persist_prediction(
            material_id="MATX-DOES-NOT-EXIST",
            target_property="tensile_strength",
            prediction_type="NUMERIC",
            request_features={"C": 0.2},
            result={"prediction": 650.0, "model": "test_model"},
            model_key="tensile_strength",
            db_path=db,
        )
def test_numeric_prediction_persists_value_and_artifact_hash(
    tmp_path, monkeypatch
):
    db = make_database(tmp_path)
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"test model artifact")
    monkeypatch.setitem(
        persistence.MODEL_FILES, "tensile_strength", artifact
    )
    result = persistence.persist_prediction(
        material_id="MATX-PRED-001",
        target_property="tensile_strength",
        prediction_type="NUMERIC",
        request_features={"C": 0.2, "Mn": 1.0},
        result={
            "prediction": 650.0,
            "model": "test_tensile_model",
            "target_unit": "source_dataset_unit_unverified",
            "applicability": {"status": "IN_DOMAIN"},
        },
        model_key="tensile_strength",
        db_path=db,
    )
    assert result["status"] == "PERSISTED"
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            """
            SELECT material_id, target_property, predicted_value,
                   predicted_unit, model_name, model_version,
                   input_json, prediction_type, prediction_json
            FROM prediction_records WHERE prediction_id = ?
            """,
            (result["prediction_id"],),
        ).fetchone()
    assert row[0:6] == (
        "MATX-PRED-001", "tensile_strength", 650.0,
        None, "test_tensile_model",
        __import__("hashlib").sha256(b"test model artifact").hexdigest(),
    )
    assert '"C":0.2' in row[6]
    assert row[7] == "NUMERIC"
    assert '"prediction":650.0' in row[8]
def test_classification_persists_class_and_probabilities_as_json(
    tmp_path, monkeypatch
):
    db = make_database(tmp_path)
    artifact = tmp_path / "classifier.joblib"
    artifact.write_bytes(b"test classifier artifact")
    monkeypatch.setitem(
        persistence.MODEL_FILES, "inclusion_classifier", artifact
    )
    result = persistence.persist_prediction(
        material_id="MATX-PRED-001",
        target_property="inclusion_classification",
        prediction_type="CLASSIFICATION",
        request_features={"features": {"Ti": 0.2, "O": 0.1}},
        result={
            "predicted_class": 1,
            "classification": "Inclusion detected",
            "probabilities": {"0": 0.03, "1": 0.97},
            "model": "test_inclusion_model",
        },
        model_key="inclusion_classifier",
        db_path=db,
    )
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            """
            SELECT predicted_value, prediction_type, prediction_json
            FROM prediction_records WHERE prediction_id = ?
            """,
            (result["prediction_id"],),
        ).fetchone()
    assert row[0] is None
    assert row[1] == "CLASSIFICATION"
    assert '"predicted_class":1' in row[2]
    assert '"1":0.97' in row[2]

@pytest.mark.parametrize(
    "prediction_type,target_property,result_payload",
    [
        (
            "NUMERIC",
            "tensile_strength",
            {"prediction": 650.0, "model": "test_tensile_model"},
        ),
        (
            "CLASSIFICATION",
            "inclusion_classification",
            {
                "predicted_class": 1,
                "classification": "Inclusion detected",
                "probabilities": {"0": 0.03, "1": 0.97},
                "model": "test_inclusion_model",
            },
        ),
    ],
)
def test_prediction_registers_model_version_idempotently(
    tmp_path, monkeypatch, prediction_type, target_property, result_payload
):
    db = make_database(tmp_path)
    artifact = tmp_path / "registered_model.joblib"
    artifact.write_bytes(b"stable model artifact for registration")
    model_key = (
        "tensile_strength"
        if prediction_type == "NUMERIC"
        else "inclusion_classifier"
    )
    monkeypatch.setitem(persistence.MODEL_FILES, model_key, artifact)
    kwargs = {
        "material_id": "MATX-PRED-001",
        "target_property": target_property,
        "prediction_type": prediction_type,
        "request_features": {"C": 0.2},
        "result": result_payload,
        "model_key": model_key,
        "db_path": db,
    }
    first = persistence.persist_prediction(**kwargs)
    second = persistence.persist_prediction(**kwargs)
    expected_hash = __import__("hashlib").sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert first["status"] == "PERSISTED"
    assert second["status"] == "PERSISTED"
    assert first["model_version_id"] == second["model_version_id"]
    assert first["artifact_sha256"] == expected_hash
    assert first["version_label"] == f"sha256-{expected_hash[:12]}"
    with sqlite3.connect(db) as connection:
        versions = connection.execute(
            """
            SELECT model_version_id, model_key, model_name,
                   version_label, artifact_sha256, artifact_path,
                   framework, target_property, metadata_json
            FROM model_versions
            WHERE model_key = ? AND artifact_sha256 = ?
            """,
            (model_key, expected_hash),
        ).fetchall()
        prediction_count = connection.execute(
            """
            SELECT COUNT(*) FROM prediction_records
            WHERE model_version = ?
            """,
            (expected_hash,),
        ).fetchone()[0]
    assert len(versions) == 1
    registered = versions[0]
    assert registered[0] == first["model_version_id"]
    assert registered[1] == model_key
    assert registered[2] == result_payload["model"]
    assert registered[3] == f"sha256-{expected_hash[:12]}"
    assert registered[4] == expected_hash
    assert registered[5] == str(artifact.resolve())
    assert registered[6] is None
    assert registered[7] == target_property
    assert '"hash_algorithm": "SHA-256"' in registered[8]
    assert prediction_count == 2
