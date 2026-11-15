"""Append-only SQLite registry for MAT-X scientific evidence."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from app.core.paths import EVIDENCE_DB_PATH, STORAGE_ROOT
from app.core.database import database_backend, database_connection
SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    investigation_id TEXT,
    evidence_category TEXT NOT NULL,
    measured_quantity TEXT NOT NULL,
    value_json TEXT,
    unit TEXT,
    uncertainty_value REAL,
    measurement_method TEXT,
    test_standard TEXT,
    source_type TEXT NOT NULL CHECK (
        source_type IN (
            'USER_SUPPLIED',
            'LABORATORY',
            'DATASET',
            'COMPUTATIONAL'
        )
    ),
    source_name TEXT,
    source_record_id TEXT,
    doi_or_url TEXT,
    specimen_id TEXT,
    observed_at TEXT,
    provenance_classification TEXT NOT NULL CHECK (
        provenance_classification IN (
            'SOURCE_VALUE',
            'DERIVED_VALUE',
            'AI_INTERPRETATION',
            'UNAVAILABLE'
        )
    ),
    validation_status TEXT NOT NULL DEFAULT 'UNVERIFIED' CHECK (
        validation_status IN (
            'UNVERIFIED',
            'VALIDATED',
            'REJECTED'
        )
    ),
    original_filename TEXT,
    stored_filename TEXT,
    sha256 TEXT,
    media_type TEXT,
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    CHECK (
        uncertainty_value IS NULL OR uncertainty_value >= 0
    )
);
CREATE INDEX IF NOT EXISTS idx_evidence_material
ON evidence_records(material_id);
CREATE INDEX IF NOT EXISTS idx_evidence_investigation
ON evidence_records(investigation_id);
CREATE INDEX IF NOT EXISTS idx_evidence_category
ON evidence_records(evidence_category);
CREATE INDEX IF NOT EXISTS idx_evidence_created
ON evidence_records(created_at);
CREATE TRIGGER IF NOT EXISTS evidence_no_update
BEFORE UPDATE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'Evidence records are immutable.');
END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete
BEFORE DELETE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'Evidence records cannot be deleted.');
END;
"""
@contextmanager
def _database_connection(path: Path | None):
    """Manage database transactions and always close the connection."""
    with database_connection(path, timeout=30) as connection:
        yield connection
def initialize_evidence_registry(
    db_path: Path | str | None = None,
) -> dict:
    """Initialize SQLite or verify the deployed PostgreSQL evidence schema."""
    backend = database_backend(db_path)
    if backend == "postgres":
        required_columns = {
            "evidence_id", "material_id", "investigation_id",
            "evidence_category", "measured_quantity", "value_json",
            "unit", "uncertainty_value", "measurement_method",
            "test_standard", "source_type", "source_name",
            "source_record_id", "doi_or_url", "specimen_id",
            "observed_at", "provenance_classification",
            "validation_status", "original_filename",
            "stored_filename", "sha256", "media_type", "created_at",
        }
        required_indexes = {
            "evidence_records_pkey",
            "idx_evidence_category",
            "idx_evidence_created",
            "idx_evidence_investigation",
            "idx_evidence_material",
        }
        required_triggers = {
            "evidence_no_update",
            "evidence_no_delete",
        }
        with _database_connection(None) as connection:
            columns = {
                row["column_name"]
                for row in connection.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'evidence_records'
                    """
                ).fetchall()
            }
            indexes = {
                row["indexname"]
                for row in connection.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'evidence_records'
                    """
                ).fetchall()
            }
            trigger_rows = connection.execute(
                """
                SELECT t.tgname, p.proname AS function_name
                FROM pg_trigger t
                JOIN pg_class c ON c.oid = t.tgrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_proc p ON p.oid = t.tgfoid
                WHERE n.nspname = 'public'
                  AND c.relname = 'evidence_records'
                  AND NOT t.tgisinternal
                """
            ).fetchall()
        triggers = {
            row["tgname"]: row["function_name"]
            for row in trigger_rows
        }
        missing_columns = required_columns - columns
        missing_indexes = required_indexes - indexes
        missing_triggers = required_triggers - set(triggers)
        invalid_triggers = {
            name: triggers[name]
            for name in required_triggers.intersection(triggers)
            if triggers[name] != "matx_prevent_evidence_mutation"
        }
        if missing_columns or missing_indexes or missing_triggers or invalid_triggers:
            raise RuntimeError(
                "PostgreSQL evidence schema verification failed: "
                f"missing_columns={sorted(missing_columns)}, "
                f"missing_indexes={sorted(missing_indexes)}, "
                f"missing_triggers={sorted(missing_triggers)}, "
                f"invalid_triggers={invalid_triggers}"
            )
        return {
            "status": "VERIFIED",
            "database_path": "postgresql",
            "table": "evidence_records",
            "column_count": len(columns),
            "index_count": len(indexes),
            "integrity": "catalog_verified",
        }
    path = Path(db_path) if db_path else EVIDENCE_DB_PATH
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _database_connection(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(
                f"Evidence database integrity check failed: {integrity}"
            )
        table_info = connection.execute(
            "PRAGMA table_info(evidence_records)"
        ).fetchall()
        indexes = connection.execute(
            "PRAGMA index_list(evidence_records)"
        ).fetchall()
    return {
        "status": "INITIALIZED",
        "database_path": str(path),
        "table": "evidence_records",
        "column_count": len(table_info),
        "index_count": len(indexes),
        "integrity": integrity,
    }
import json
import math
import uuid
SOURCE_TYPES = {
    "USER_SUPPLIED",
    "LABORATORY",
    "DATASET",
    "COMPUTATIONAL",
}
PROVENANCE_TYPES = {
    "SOURCE_VALUE",
    "DERIVED_VALUE",
    "AI_INTERPRETATION",
    "UNAVAILABLE",
}
EVIDENCE_FIELDS = {
    "material_id",
    "investigation_id",
    "evidence_category",
    "measured_quantity",
    "value",
    "unit",
    "uncertainty_value",
    "measurement_method",
    "test_standard",
    "source_type",
    "source_name",
    "source_record_id",
    "doi_or_url",
    "specimen_id",
    "observed_at",
    "provenance_classification",
    "validation_status",
    "original_filename",
    "stored_filename",
    "sha256",
    "media_type",
}
def _required_text(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
def _optional_text(payload: dict, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string or None.")
    return value.strip()
def _validate_json_value(value, path="value"):
    """Allow only strict JSON-compatible values and finite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Evidence value must be finite JSON data ({path}).")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings.")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path} contains unsupported JSON data: {type(value).__name__}."
    )
def create_evidence_record(
    payload: dict,
    db_path: Path | str | None = None,
) -> dict:
    """Insert one immutable evidence record."""
    if not isinstance(payload, dict):
        raise ValueError("Evidence payload must be a dictionary.")
    unknown = set(payload) - EVIDENCE_FIELDS
    if unknown:
        raise ValueError(
            f"Unsupported evidence fields: {sorted(unknown)}"
        )
    material_id = _required_text(payload, "material_id")
    category = _required_text(payload, "evidence_category")
    quantity = _required_text(payload, "measured_quantity")
    source_type = _required_text(payload, "source_type").upper()
    if source_type not in SOURCE_TYPES:
        raise ValueError("Unsupported source_type.")
    provenance = _required_text(
        payload, "provenance_classification"
    ).upper()
    if provenance not in PROVENANCE_TYPES:
        raise ValueError("Unsupported provenance_classification.")
    optional_text_fields = (
        "investigation_id",
        "unit",
        "measurement_method",
        "test_standard",
        "source_name",
        "source_record_id",
        "doi_or_url",
        "specimen_id",
        "observed_at",
        "original_filename",
        "stored_filename",
        "sha256",
        "media_type",
    )
    optional_text = {
        field: _optional_text(payload, field)
        for field in optional_text_fields
    }
    validation_status = payload.get("validation_status", "UNVERIFIED")
    if not isinstance(validation_status, str):
        raise ValueError("validation_status must be a string.")
    validation_status = validation_status.strip().upper()
    if validation_status not in {"UNVERIFIED", "VALIDATED", "REJECTED"}:
        raise ValueError("Unsupported validation_status.")
    value = payload.get("value")
    if value is not None:
        _validate_json_value(value)
        value_json = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    else:
        value_json = None
    uncertainty = payload.get("uncertainty_value")
    if uncertainty is not None:
        if isinstance(uncertainty, bool) or not isinstance(
            uncertainty, (int, float)
        ):
            raise ValueError("uncertainty_value must be numeric.")
        uncertainty = float(uncertainty)
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError(
                "uncertainty_value must be finite and non-negative."
            )
    if provenance == "UNAVAILABLE" and value is not None:
        raise ValueError(
            "UNAVAILABLE evidence cannot contain a measured value."
        )
    evidence_id = str(uuid.uuid4())
    backend = database_backend(db_path)
    path = (
        Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        if backend == "sqlite"
        else None
    )
    columns = [
        "evidence_id",
        "material_id",
        "investigation_id",
        "evidence_category",
        "measured_quantity",
        "value_json",
        "unit",
        "uncertainty_value",
        "measurement_method",
        "test_standard",
        "source_type",
        "source_name",
        "source_record_id",
        "doi_or_url",
        "specimen_id",
        "observed_at",
        "provenance_classification",
        "validation_status",
        "original_filename",
        "stored_filename",
        "sha256",
        "media_type",
    ]
    values = [
        evidence_id,
        material_id,
        optional_text["investigation_id"],
        category,
        quantity,
        value_json,
        optional_text["unit"],
        uncertainty,
        optional_text["measurement_method"],
        optional_text["test_standard"],
        source_type,
        optional_text["source_name"],
        optional_text["source_record_id"],
        optional_text["doi_or_url"],
        optional_text["specimen_id"],
        optional_text["observed_at"],
        provenance,
        validation_status,
        optional_text["original_filename"],
        optional_text["stored_filename"],
        optional_text["sha256"],
        optional_text["media_type"],
    ]
    placeholder = "%s" if backend == "postgres" else "?"
    placeholders = ", ".join(placeholder for _ in columns)
    column_sql = ", ".join(columns)
    with _database_connection(path) as connection:
        connection.execute(
            f"INSERT INTO evidence_records ({column_sql}) "
            f"VALUES ({placeholders})",
            values,
        )
    return get_evidence_record(evidence_id, path)
def _decode_evidence_row(row: sqlite3.Row) -> dict:
    result = dict(row)
    raw_value = result.pop("value_json")
    result["value"] = (
        json.loads(raw_value) if raw_value is not None else None
    )
    return result
def get_evidence_record(
    evidence_id: str,
    db_path: Path | str | None = None,
) -> dict | None:
    """Retrieve a single evidence record."""
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("evidence_id must be a non-empty string.")
    backend = database_backend(db_path)
    path = (
        Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        if backend == "sqlite"
        else None
    )
    placeholder = "%s" if backend == "postgres" else "?"
    with _database_connection(path) as connection:
        row = connection.execute(
            f"SELECT * FROM evidence_records WHERE evidence_id = {placeholder}",
            (evidence_id.strip(),),
        ).fetchone()
    return _decode_evidence_row(row) if row else None
def list_evidence_records(
    material_id: str | None = None,
    investigation_id: str | None = None,
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict]:
    """List evidence records with optional material/investigation filters."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer between 1 and 1000.")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000.")
    for field, value in (
        ("material_id", material_id),
        ("investigation_id", investigation_id),
    ):
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"{field} must be a non-empty string or None.")
    backend = database_backend(db_path)
    placeholder = "%s" if backend == "postgres" else "?"
    clauses = []
    parameters = []
    if material_id is not None:
        clauses.append(f"material_id = {placeholder}")
        parameters.append(material_id.strip())
    if investigation_id is not None:
        clauses.append(f"investigation_id = {placeholder}")
        parameters.append(investigation_id.strip())
    where_sql = (
        " WHERE " + " AND ".join(clauses) if clauses else ""
    )
    path = (
        Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        if backend == "sqlite"
        else None
    )
    with _database_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM evidence_records"
            + where_sql
            + f" ORDER BY created_at DESC, evidence_id DESC LIMIT {placeholder}",
            (*parameters, limit),
        ).fetchall()
    return [_decode_evidence_row(row) for row in rows]
