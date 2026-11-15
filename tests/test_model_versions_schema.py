import sqlite3
import pytest
from app.services.evidence_registry_service import initialize_evidence_registry
from app.services.matx_database_service import (
    SCHEMA_VERSION,
    initialize_matx_database,
)
def make_database(tmp_path):
    db = tmp_path / "model_versions.sqlite3"
    initialize_evidence_registry(db)
    result = initialize_matx_database(db)
    return db, result
def test_model_version_registry_is_created_and_verified(tmp_path):
    db, result = make_database(tmp_path)
    assert SCHEMA_VERSION == 4
    assert result["schema_version"] == 4
    assert "model_versions" in result["tables"]
    with sqlite3.connect(db) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(model_versions)"
            )
        }
    assert {
        "model_version_id",
        "model_key",
        "model_name",
        "version_label",
        "artifact_sha256",
        "metadata_json",
    }.issubset(columns)
def test_model_version_registry_accepts_artifact_identity(tmp_path):
    db, _ = make_database(tmp_path)
    artifact_hash = "a" * 64
    with sqlite3.connect(db) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_version_id, model_key, model_name,
                version_label, artifact_sha256, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "MV-TEST-001",
                "tensile_strength",
                "test_tensile_model",
                "1.0",
                artifact_hash,
                '{"validation":"test"}',
            ),
        )
        record = connection.execute(
            """
            SELECT model_key, model_name, version_label, artifact_sha256
            FROM model_versions WHERE model_version_id = ?
            """,
            ("MV-TEST-001",),
        ).fetchone()
    assert record == (
        "tensile_strength",
        "test_tensile_model",
        "1.0",
        artifact_hash,
    )
def test_model_version_artifact_identity_must_be_unique(tmp_path):
    db, _ = make_database(tmp_path)
    artifact_hash = "b" * 64
    with sqlite3.connect(db) as connection:
        values = (
            "MV-TEST-002",
            "yield_strength",
            "test_yield_model",
            "1.0",
            artifact_hash,
        )
        connection.execute(
            """
            INSERT INTO model_versions (
                model_version_id, model_key, model_name,
                version_label, artifact_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            values,
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO model_versions (
                    model_version_id, model_key, model_name,
                    version_label, artifact_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "MV-TEST-003",
                    "yield_strength",
                    "another_model_name",
                    "2.0",
                    artifact_hash,
                ),
            )
def test_model_version_rejects_invalid_artifact_hash(tmp_path):
    db, _ = make_database(tmp_path)
    with sqlite3.connect(db) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO model_versions (
                    model_version_id, model_key, model_name,
                    version_label, artifact_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "MV-INVALID",
                    "tensile_strength",
                    "test_model",
                    "1.0",
                    "invalid-hash",
                ),
            )
def test_model_version_migration_is_idempotent(tmp_path):
    db, first = make_database(tmp_path)
    second = initialize_matx_database(db)
    assert first["schema_version"] == second["schema_version"] == 4
    assert first["tables"] == second["tables"]
    with sqlite3.connect(db) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert versions == [(1,), (2,), (3,), (4,)]
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []
