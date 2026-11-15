import json
import math
import sqlite3
from pathlib import Path
from app.core.paths import EVIDENCE_DB_PATH
from app.core.database import database_backend, database_connection, get_table_columns
SCHEMA_VERSION = 4
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS materials (
    material_id TEXT PRIMARY KEY,
    material_family TEXT,
    grade TEXT,
    designation TEXT,
    description TEXT,
    identity_status TEXT NOT NULL DEFAULT 'UNCONFIRMED',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    dataset_name TEXT,
    doi_or_url TEXT,
    publication TEXT,
    original_record_id TEXT,
    method TEXT,
    provenance_classification TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS material_compositions (
    composition_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    element TEXT NOT NULL,
    value REAL NOT NULL CHECK(value >= 0 AND value <= 100),
    unit TEXT NOT NULL DEFAULT 'wt%',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    UNIQUE(material_id, element, composition_id)
);
CREATE TABLE IF NOT EXISTS process_records (
    process_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    process_type TEXT NOT NULL,
    process_parameters_json TEXT NOT NULL DEFAULT '{}',
    sequence_number INTEGER,
    started_at TEXT,
    completed_at TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS heat_treatments (
    treatment_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    treatment_type TEXT NOT NULL,
    temperature_value REAL,
    temperature_unit TEXT,
    holding_time REAL,
    holding_time_unit TEXT,
    cooling_method TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS microstructure_records (
    microstructure_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    characterization_method TEXT NOT NULL,
    image_path TEXT,
    map_path TEXT,
    features_json TEXT NOT NULL DEFAULT '{}',
    phases_json TEXT NOT NULL DEFAULT '{}',
    grain_size_value REAL,
    grain_size_unit TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS property_records (
    property_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    property_name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    unit TEXT,
    uncertainty_value REAL CHECK(
        uncertainty_value IS NULL OR uncertainty_value >= 0
    ),
    measurement_method TEXT,
    test_standard TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS service_records (
    service_record_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    environment_json TEXT NOT NULL DEFAULT '{}',
    temperature_value REAL,
    temperature_unit TEXT,
    duration_value REAL,
    duration_unit TEXT,
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS damage_records (
    damage_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    specimen_id TEXT,
    damage_type TEXT NOT NULL,
    observation_json TEXT NOT NULL,
    severity_value REAL,
    severity_unit TEXT,
    diagnosis_status TEXT NOT NULL DEFAULT 'NOT_DIAGNOSED',
    provenance_classification TEXT NOT NULL,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    initial_state_json TEXT NOT NULL DEFAULT '{}',
    final_state_json TEXT,
    limitations_json TEXT NOT NULL DEFAULT '[]',
    started_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS prediction_records (
    prediction_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_id TEXT REFERENCES investigations(investigation_id),
    target_property TEXT NOT NULL,
    predicted_value REAL,
    predicted_unit TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_json TEXT NOT NULL,
    applicability_json TEXT NOT NULL DEFAULT '{}',
    provenance_classification TEXT NOT NULL DEFAULT 'PREDICTED_VALUE',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS uncertainty_records (
    uncertainty_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_id TEXT REFERENCES investigations(investigation_id),
    prediction_id TEXT REFERENCES prediction_records(prediction_id),
    uncertainty_category TEXT NOT NULL,
    uncertainty_value REAL CHECK(
        uncertainty_value IS NULL OR
        (uncertainty_value >= 0 AND uncertainty_value <= 1)
    ),
    lower_bound REAL,
    upper_bound REAL,
    methodology TEXT,
    calibration_status TEXT NOT NULL DEFAULT 'NOT_CALIBRATED',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES materials(material_id),
    investigation_id TEXT REFERENCES investigations(investigation_id),
    experiment_type TEXT NOT NULL,
    objective TEXT,
    recommended_measurement TEXT,
    experimental_method TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    status TEXT NOT NULL DEFAULT 'PLANNED',
    provenance_classification TEXT,
    source_id TEXT REFERENCES sources(source_id),
    evidence_id TEXT REFERENCES evidence_records(evidence_id),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_composition_material
ON material_compositions(material_id);
CREATE INDEX IF NOT EXISTS idx_process_material
ON process_records(material_id);
CREATE INDEX IF NOT EXISTS idx_heat_treatment_material
ON heat_treatments(material_id);
CREATE INDEX IF NOT EXISTS idx_microstructure_material
ON microstructure_records(material_id);
CREATE INDEX IF NOT EXISTS idx_property_material
ON property_records(material_id);
CREATE INDEX IF NOT EXISTS idx_service_material
ON service_records(material_id);
CREATE INDEX IF NOT EXISTS idx_damage_material
ON damage_records(material_id);
CREATE INDEX IF NOT EXISTS idx_investigation_material
ON investigations(material_id);
CREATE INDEX IF NOT EXISTS idx_prediction_material
ON prediction_records(material_id);
CREATE INDEX IF NOT EXISTS idx_uncertainty_material
ON uncertainty_records(material_id);
CREATE INDEX IF NOT EXISTS idx_experiment_material
ON experiments(material_id);
"""

def _migrate_v1_to_v2(connection):
    """Upgrade uncertainty records without losing existing scientific data."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "ALTER TABLE uncertainty_records "
            "RENAME TO uncertainty_records_v1"
        )
        connection.execute("""
            CREATE TABLE uncertainty_records (
                uncertainty_id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL REFERENCES materials(material_id),
                investigation_id TEXT REFERENCES investigations(investigation_id),
                prediction_id TEXT REFERENCES prediction_records(prediction_id),
                uncertainty_category TEXT NOT NULL,
                uncertainty_value REAL CHECK(
                    uncertainty_value IS NULL OR uncertainty_value >= 0
                ),
                uncertainty_unit TEXT,
                uncertainty_kind TEXT NOT NULL DEFAULT 'UNSPECIFIED',
                lower_bound REAL,
                upper_bound REAL,
                methodology TEXT,
                calibration_status TEXT NOT NULL DEFAULT 'NOT_CALIBRATED',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
        """)
        connection.execute("""
            INSERT INTO uncertainty_records (
                uncertainty_id, material_id, investigation_id,
                prediction_id, uncertainty_category, uncertainty_value,
                lower_bound, upper_bound, methodology, calibration_status,
                metadata_json, created_at
            )
            SELECT
                uncertainty_id, material_id, investigation_id,
                prediction_id, uncertainty_category, uncertainty_value,
                lower_bound, upper_bound, methodology, calibration_status,
                metadata_json, created_at
            FROM uncertainty_records_v1
        """)
        connection.execute("DROP TABLE uncertainty_records_v1")
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_uncertainty_material
            ON uncertainty_records(material_id)
        """)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, description)
            VALUES (?, ?)
            """,
            (
                2,
                "Uncertainty units and kinds; remove normalized 0-1 restriction",
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
def _migrate_v2_to_v3(connection):
    """Add typed prediction payload storage without losing v2 records."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(prediction_records)"
            ).fetchall()
        }
        if "prediction_type" not in columns:
            connection.execute(
                "ALTER TABLE prediction_records "
                "ADD COLUMN prediction_type TEXT NOT NULL DEFAULT 'NUMERIC'"
            )
        if "prediction_json" not in columns:
            connection.execute(
                "ALTER TABLE prediction_records "
                "ADD COLUMN prediction_json TEXT NOT NULL DEFAULT '{}'"
            )
        connection.execute(
            """
            INSERT INTO schema_migrations (version, description)
            VALUES (?, ?)
            """,
            (
                3,
                "Typed prediction payloads for numeric and classification results",
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise

def _migrate_v3_to_v4(connection):
    """Add persistent model-version registry without changing predictions."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_version_id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                version_label TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL CHECK(
                    length(artifact_sha256) = 64
                ),
                artifact_path TEXT,
                framework TEXT,
                target_property TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                registered_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                UNIQUE(model_key, artifact_sha256)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_versions_key
            ON model_versions(model_key)
        """)
        connection.execute("""
            INSERT INTO schema_migrations (version, description)
            VALUES (?, ?)
        """, (
            4,
            "Persistent model-version registry with artifact SHA256 identity",
        ))
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def initialize_matx_database(db_path=None):

    required_columns = {
        "damage_records": {
            "damage_id", "material_id", "specimen_id", "damage_type",
            "observation_json", "severity_value", "severity_unit",
            "diagnosis_status", "provenance_classification", "source_id",
            "evidence_id", "created_at",
        },
        "experiments": {
            "experiment_id", "material_id", "investigation_id",
            "experiment_type", "objective", "recommended_measurement",
            "experimental_method", "parameters_json", "result_json",
            "status", "provenance_classification", "source_id",
            "evidence_id", "created_at", "completed_at",
        },
        "heat_treatments": {
            "treatment_id", "material_id", "specimen_id", "treatment_type",
            "temperature_value", "temperature_unit", "holding_time",
            "holding_time_unit", "cooling_method", "parameters_json",
            "provenance_classification", "source_id", "evidence_id",
            "created_at",
        },
        "investigations": {
            "investigation_id", "material_id", "investigation_goal", "status",
            "initial_state_json", "final_state_json", "limitations_json",
            "started_at", "completed_at",
        },
        "material_compositions": {
            "composition_id", "material_id", "element", "value", "unit",
            "provenance_classification", "source_id", "evidence_id",
            "created_at",
        },
        "materials": {
            "material_id", "material_family", "grade", "designation",
            "description", "identity_status", "metadata_json", "created_at",
        },
        "microstructure_records": {
            "microstructure_id", "material_id", "specimen_id",
            "characterization_method", "image_path", "map_path",
            "features_json", "phases_json", "grain_size_value",
            "grain_size_unit", "provenance_classification", "source_id",
            "evidence_id", "created_at",
        },
        "prediction_records": {
            "prediction_id", "material_id", "investigation_id",
            "target_property", "predicted_value", "predicted_unit",
            "model_name", "model_version", "input_json",
            "applicability_json", "provenance_classification", "created_at",
            "prediction_type", "prediction_json",
        },
        "process_records": {
            "process_id", "material_id", "specimen_id", "process_type",
            "process_parameters_json", "sequence_number", "started_at",
            "completed_at", "provenance_classification", "source_id",
            "evidence_id", "created_at",
        },
        "property_records": {
            "property_id", "material_id", "specimen_id", "property_name",
            "value_json", "unit", "uncertainty_value", "measurement_method",
            "test_standard", "provenance_classification", "source_id",
            "evidence_id", "created_at",
        },
        "service_records": {
            "service_record_id", "material_id", "specimen_id",
            "environment_json", "temperature_value", "temperature_unit",
            "duration_value", "duration_unit", "provenance_classification",
            "source_id", "evidence_id", "created_at",
        },
        "sources": {
            "source_id", "source_type", "source_name", "dataset_name",
            "doi_or_url", "publication", "original_record_id", "method",
            "provenance_classification", "metadata_json", "created_at",
        },
        "uncertainty_records": {
            "uncertainty_id", "material_id", "investigation_id",
            "prediction_id", "uncertainty_category", "uncertainty_value",
            "uncertainty_unit", "uncertainty_kind", "lower_bound",
            "upper_bound", "methodology", "calibration_status",
            "metadata_json", "created_at",
        },
    }
    required_tables = set(required_columns) | {
        "schema_migrations", "model_versions", "evidence_records"
    }
    if database_backend(db_path) == "postgres":
        with database_connection() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            migration_rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            versions = [row["version"] for row in migration_rows]
            if versions != list(range(1, SCHEMA_VERSION + 1)):
                raise RuntimeError(
                    f"PostgreSQL migration history mismatch: {versions}"
                )
            table_rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                """
            ).fetchall()
            tables = sorted(row["table_name"] for row in table_rows)
            missing_tables = required_tables - set(tables)
            if missing_tables:
                raise RuntimeError(
                    f"PostgreSQL required tables missing: {sorted(missing_tables)}"
                )
            for table_name, expected_columns in required_columns.items():
                actual_columns = get_table_columns(connection, table_name)
                missing_columns = expected_columns - set(actual_columns)
                if missing_columns:
                    raise RuntimeError(
                        f"PostgreSQL {table_name} missing columns: "
                        f"{sorted(missing_columns)}"
                    )
            model_columns = {
                "model_version_id", "model_key", "model_name",
                "version_label", "artifact_sha256", "metadata_json",
            }
            actual_model_columns = get_table_columns(
                connection, "model_versions"
            )
            missing_model_columns = model_columns - set(actual_model_columns)
            if missing_model_columns:
                raise RuntimeError(
                    "PostgreSQL model_versions missing columns: "
                    f"{sorted(missing_model_columns)}"
                )
            migration_columns = {"version", "description"}
            actual_migration_columns = get_table_columns(
                connection, "schema_migrations"
            )
            missing_migration_columns = (
                migration_columns - set(actual_migration_columns)
            )
            if missing_migration_columns:
                raise RuntimeError(
                    "PostgreSQL schema_migrations missing columns: "
                    f"{sorted(missing_migration_columns)}"
                )
        return {
            "status": "VERIFIED",
            "database_path": "postgresql",
            "schema_version": SCHEMA_VERSION,
            "table_count": len(tables),
            "tables": tables,
            "integrity": "catalog_verified",
            "foreign_key_errors": None,
        }
    else:
        path = Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            # The base schema remains version 1. Every new database follows
            # the same explicit migration path as an existing version-1 DB.
            migration_script = (
                "BEGIN IMMEDIATE;\n"
                + SCHEMA
                + "\nINSERT OR IGNORE INTO schema_migrations "
                "(version, description) VALUES "
                "(1, 'Initial MAT-X scientific persistence schema');\n"
                "COMMIT;"
            )
            connection.executescript(migration_script)
            version = connection.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0]
            if version is None or version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported database schema version: {version}"
                )
            while version < SCHEMA_VERSION:
                if version == 1:
                    _migrate_v1_to_v2(connection)
                elif version == 2:
                    _migrate_v2_to_v3(connection)
                elif version == 3:
                    _migrate_v3_to_v4(connection)
                else:
                    raise RuntimeError(
                        f"No migration path exists from schema version {version}"
                    )
                version = connection.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0]
            if version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unexpected database schema version: {version}"
                )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(uncertainty_records)"
                ).fetchall()
            }
            required_uncertainty_columns = {
                "uncertainty_value",
                "uncertainty_unit",
                "uncertainty_kind",
            }
            if not required_uncertainty_columns.issubset(columns):
                raise RuntimeError(
                    "Uncertainty schema v2 verification failed."
                )
            prediction_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(prediction_records)"
                ).fetchall()
            }
            required_prediction_columns = {
                "prediction_type", "prediction_json"
            }
            if not required_prediction_columns.issubset(prediction_columns):
                raise RuntimeError("Prediction schema v3 verification failed.")
            model_version_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(model_versions)"
                ).fetchall()
            }
            required_model_version_columns = {
                "model_version_id",
                "model_key",
                "model_name",
                "version_label",
                "artifact_sha256",
                "metadata_json",
            }
            if not required_model_version_columns.issubset(
                model_version_columns
            ):
                raise RuntimeError("Model-version schema v4 verification failed.")
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"Database integrity check failed: {integrity}"
                )
            tables = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            foreign_keys = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if foreign_keys:
                raise RuntimeError(
                    f"Foreign key validation failed: {foreign_keys}"
                )
            return {
                "status": "INITIALIZED",
                "database_path": str(path),
                "schema_version": version,
                "table_count": len(tables),
                "tables": [row[0] for row in tables],
                "integrity": integrity,
                "foreign_key_errors": len(foreign_keys),
            }
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
