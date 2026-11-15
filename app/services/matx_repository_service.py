import json
import math
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from app.core.paths import EVIDENCE_DB_PATH
# Only these application-owned scientific tables may be accessed here.
# Evidence records retain their dedicated immutable registry service.
REPOSITORY_TABLES = {
    "materials",
    "sources",
    "material_compositions",
    "process_records",
    "heat_treatments",
    "microstructure_records",
    "property_records",
    "service_records",
    "damage_records",
    "investigations",
    "prediction_records",
    "uncertainty_records",
    "experiments",
}
JSON_COLUMNS = {
    "metadata_json",
    "process_parameters_json",
    "parameters_json",
    "features_json",
    "phases_json",
    "environment_json",
    "observation_json",
    "initial_state_json",
    "final_state_json",
    "limitations_json",
    "input_json",
    "applicability_json",
    "prediction_json",
    "metadata_json",
    "parameters_json",
    "result_json",
    "value_json",
}
class MatxRepositoryError(ValueError):
    """Raised when a scientific persistence operation is invalid."""
def _connect(db_path=None):
    """Open the configured backend while preserving explicit SQLite paths."""
    from app.core.database import connect_database, database_backend
    backend = database_backend(db_path)
    if backend == "sqlite":
        path = Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        if not path.exists():
            raise MatxRepositoryError(
                f"MAT-X database does not exist: {path}"
            )
        return connect_database(path, timeout=30)
    return connect_database(None, timeout=30)
def _is_postgres(connection):
    return not isinstance(connection, sqlite3.Connection)
def _execute(connection, sql, parameters=()):
    """Execute qmark SQL against SQLite or psycopg."""
    if _is_postgres(connection):
        sql = sql.replace("?", "%s")
    return connection.execute(sql, parameters)
def _transaction(connection):
    """Commit or roll back a unit of work without closing the connection."""
    from contextlib import contextmanager
    @contextmanager
    def transaction_scope():
        if _is_postgres(connection):
            connection.commit()
            with connection.transaction():
                yield
        else:
            with connection:
                yield
    return transaction_scope()
def _integrity_errors():
    """Return integrity exception types supported by this installation."""
    try:
        import psycopg
        return (sqlite3.IntegrityError, psycopg.IntegrityError)
    except ImportError:
        return (sqlite3.IntegrityError,)
def _validate_table(table):
    if not isinstance(table, str) or table not in REPOSITORY_TABLES:
        raise MatxRepositoryError(
            f"Unsupported scientific table: {table!r}"
        )
def _table_metadata(connection, table):
    """Return validated column metadata and the single primary key."""
    from app.core.database import get_table_columns
    _validate_table(table)
    if isinstance(connection, sqlite3.Connection):
        rows = connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
        if not rows:
            raise MatxRepositoryError(
                f"Scientific table is not initialized: {table}"
            )
        columns = {row["name"]: row for row in rows}
        primary_keys = [
            row["name"] for row in rows if row["pk"]
        ]
    else:
        names = get_table_columns(connection, table)
        if not names:
            raise MatxRepositoryError(
                f"Scientific table is not initialized: {table}"
            )
        columns = {name: None for name in names}
        rows = _execute(
            connection,
            """
            SELECT k.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage k
              ON k.constraint_schema = tc.constraint_schema
             AND k.table_name = tc.table_name
             AND k.constraint_name = tc.constraint_name
            WHERE tc.table_schema = current_schema()
              AND tc.table_name = ?
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY k.ordinal_position
            """,
            (table,),
        ).fetchall()
        primary_keys = [row["column_name"] for row in rows]
    if len(primary_keys) != 1:
        raise MatxRepositoryError(
            f"Expected one primary key for table: {table}"
        )
    return columns, primary_keys[0]
def _validate_json_value(value, location="value"):
    if isinstance(value, float) and not math.isfinite(value):
        raise MatxRepositoryError(
            f"Non-finite numeric value is not allowed at {location}."
        )
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MatxRepositoryError(
                    f"JSON object keys must be strings at {location}."
                )
            _validate_json_value(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
def _encode_payload(payload, columns, primary_key):
    if not isinstance(payload, dict) or not payload:
        raise MatxRepositoryError(
            "Record payload must be a non-empty object."
        )
    unknown = set(payload) - set(columns)
    if unknown:
        raise MatxRepositoryError(
            "Unknown fields: " + ", ".join(sorted(unknown))
        )
    if primary_key not in payload:
        raise MatxRepositoryError(
            f"Missing primary key: {primary_key}"
        )
    encoded = {}
    for key, value in payload.items():
        if key in {"created_at", "started_at", "completed_at",
                   "applied_at"}:
            raise MatxRepositoryError(
                f"Timestamp field is database-managed: {key}"
            )
        if key in JSON_COLUMNS and value is not None:
            _validate_json_value(value, key)
            try:
                value = json.dumps(
                    value,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise MatxRepositoryError(
                    f"Invalid JSON value for {key}: {exc}"
                ) from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise MatxRepositoryError(
                f"Non-finite numeric value is not allowed: {key}"
            )
        encoded[key] = value
    identifier = encoded[primary_key]
    if not isinstance(identifier, str) or not identifier.strip():
        raise MatxRepositoryError(
            f"{primary_key} must be a non-empty string."
        )
    return encoded
def _decode_record(record):
    if record is None:
        return None
    result = dict(record)
    for key in JSON_COLUMNS.intersection(result):
        value = result[key]
        if value is not None:
            try:
                result[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                # Preserve legacy values rather than silently discarding them.
                pass
    return result
def create_scientific_record(table, payload, db_path=None):
    """Insert one append-only scientific record and return the stored row."""
    _validate_table(table)
    connection = _connect(db_path)
    try:
        columns, primary_key = _table_metadata(connection, table)
        encoded = _encode_payload(payload, columns, primary_key)
        names = list(encoded)
        placeholders = ", ".join("?" for _ in names)
        quoted_names = ", ".join(f'"{name}"' for name in names)
        sql = (
            f'INSERT INTO "{table}" ({quoted_names}) '
            f"VALUES ({placeholders})"
        )
        with _transaction(connection):
            _execute(connection, 
                sql,
                [encoded[name] for name in names],
            )
        row = _execute(connection, 
            f'SELECT * FROM "{table}" WHERE "{primary_key}" = ?',
            (encoded[primary_key],),
        ).fetchone()
        return _decode_record(row)
    except _integrity_errors() as exc:
        raise MatxRepositoryError(
            f"Database rejected {table} record: {exc}"
        ) from exc
    finally:
        connection.close()
def get_scientific_record(table, record_id, db_path=None):
    """Retrieve one scientific record by its table-specific primary key."""
    _validate_table(table)
    if not isinstance(record_id, str) or not record_id.strip():
        raise MatxRepositoryError("Record ID must be a non-empty string.")
    connection = _connect(db_path)
    try:
        _, primary_key = _table_metadata(connection, table)
        row = _execute(connection, 
            f'SELECT * FROM "{table}" WHERE "{primary_key}" = ?',
            (record_id,),
        ).fetchone()
        return _decode_record(row)
    finally:
        connection.close()
def list_scientific_records(
    table,
    filters=None,
    limit=100,
    offset=0,
    db_path=None,
):
    """List records with validated equality filters and bounded pagination."""
    _validate_table(table)
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise MatxRepositoryError("Limit must be an integer.")
    if not 1 <= limit <= 1000:
        raise MatxRepositoryError("Limit must be between 1 and 1000.")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise MatxRepositoryError("Offset must be a non-negative integer.")
    filters = filters or {}
    if not isinstance(filters, dict):
        raise MatxRepositoryError("Filters must be an object.")
    connection = _connect(db_path)
    try:
        columns, _ = _table_metadata(connection, table)
        unknown = set(filters) - set(columns)
        if unknown:
            raise MatxRepositoryError(
                "Unknown filter fields: " + ", ".join(sorted(unknown))
            )
        where = ""
        parameters = []
        if filters:
            clauses = []
            for key, value in filters.items():
                operator = (
                    "IS NOT DISTINCT FROM"
                    if _is_postgres(connection)
                    else "IS"
                )
                clauses.append(f'"{key}" {operator} ?')
                parameters.append(value)
            where = " WHERE " + " AND ".join(clauses)
        if _is_postgres(connection):
            order_column = (
                "created_at" if "created_at" in columns else primary_key
            )
            ordering = (
                f'"{order_column}" DESC, "{primary_key}" DESC'
                if order_column != primary_key
                else f'"{primary_key}" DESC'
            )
        else:
            ordering = "rowid DESC"
        sql = (
            f'SELECT * FROM "{table}"{where} '
            f"ORDER BY {ordering} LIMIT ? OFFSET ?"
        )
        parameters.extend([limit, offset])
        rows = _execute(connection, sql, parameters).fetchall()
        return [_decode_record(row) for row in rows]
    finally:
        connection.close()

# ---------------------------------------------------------------------
# Controlled investigation and experiment lifecycle updates
# ---------------------------------------------------------------------
LIFECYCLE_RULES = {
    "investigations": {
        "editable_fields": {
            "status", "final_state_json", "limitations_json"
        },
        "transitions": {
            "OPEN": {"IN_PROGRESS", "COMPLETED", "CANCELLED"},
            "IN_PROGRESS": {"COMPLETED", "CANCELLED"},
        },
        "completion_field": "final_state_json",
    },
    "experiments": {
        "editable_fields": {
            "status", "result_json", "provenance_classification",
            "source_id", "evidence_id",
        },
        "transitions": {
            "PLANNED": {
                "IN_PROGRESS", "COMPLETED", "CANCELLED", "FAILED"
            },
            "IN_PROGRESS": {"COMPLETED", "CANCELLED", "FAILED"},
        },
        "completion_field": "result_json",
    },
}
def update_lifecycle_record(table, record_id, payload, db_path=None):
    """Update only controlled investigation or experiment lifecycle fields."""
    if table not in LIFECYCLE_RULES:
        raise MatxRepositoryError(
            "Lifecycle updates are allowed only for investigations "
            "and experiments."
        )
    if not isinstance(record_id, str) or not record_id.strip():
        raise MatxRepositoryError("Record ID must be a non-empty string.")
    if not isinstance(payload, dict) or not payload:
        raise MatxRepositoryError(
            "Lifecycle update payload must be a non-empty object."
        )
    rules = LIFECYCLE_RULES[table]
    unknown = set(payload) - rules["editable_fields"]
    if unknown:
        raise MatxRepositoryError(
            "Fields cannot be changed through lifecycle updates: "
            + ", ".join(sorted(unknown))
        )
    if "status" not in payload:
        raise MatxRepositoryError(
            "A lifecycle update must explicitly include the target status."
        )
    target_status = payload["status"]
    if not isinstance(target_status, str):
        raise MatxRepositoryError("Status must be a string.")
    target_status = target_status.strip().upper()
    if not target_status:
        raise MatxRepositoryError("Status cannot be empty.")
    connection = _connect(db_path)
    try:
        columns, primary_key = _table_metadata(connection, table)
        for key in payload:
            if key not in columns:
                raise MatxRepositoryError(f"Unknown field: {key}")
        row = _execute(connection, 
            f'SELECT * FROM "{table}" WHERE "{primary_key}" = ?',
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        current = _decode_record(row)
        current_status = current["status"]
        allowed = rules["transitions"].get(current_status, set())
        if target_status not in allowed:
            raise MatxRepositoryError(
                f"Invalid {table} status transition: "
                f"{current_status} -> {target_status}."
            )
        encoded = {}
        for key, value in payload.items():
            if key == "status":
                encoded[key] = target_status
            elif key in JSON_COLUMNS and value is not None:
                _validate_json_value(value, key)
                try:
                    encoded[key] = json.dumps(
                        value,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError) as exc:
                    raise MatxRepositoryError(
                        f"Invalid JSON value for {key}: {exc}"
                    ) from exc
            elif isinstance(value, float) and not math.isfinite(value):
                raise MatxRepositoryError(
                    f"Non-finite numeric value is not allowed: {key}"
                )
            else:
                encoded[key] = value
        if target_status == "COMPLETED":
            completion_field = rules["completion_field"]
            completion_value = encoded.get(
                completion_field, current.get(completion_field)
            )
            if completion_value is None:
                raise MatxRepositoryError(
                    f"Cannot complete {table} without {completion_field}."
                )
        assignments = [f'"{key}" = ?' for key in encoded]
        values = list(encoded.values())
        if target_status == "COMPLETED":
            if _is_postgres(connection):
                assignments.append(
                    '"completed_at" = '
                    """to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', """
                    """'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')"""
                )
            else:
                assignments.append(
                    '"completed_at" = strftime'
                    "('%Y-%m-%dT%H:%M:%fZ', 'now')"
                )
        sql = (
            f'UPDATE "{table}" SET {", ".join(assignments)} '
            f'WHERE "{primary_key}" = ?'
        )
        values.append(record_id)
        with _transaction(connection):
            _execute(connection, sql, values)
        updated = _execute(connection, 
            f'SELECT * FROM "{table}" WHERE "{primary_key}" = ?',
            (record_id,),
        ).fetchone()
        return _decode_record(updated)
    except _integrity_errors() as exc:
        raise MatxRepositoryError(
            f"Database rejected {table} lifecycle update: {exc}"
        ) from exc
    finally:
        connection.close()

# ---------------------------------------------------------------------
# Atomic persistence of active-investigation recommendations
# ---------------------------------------------------------------------
def persist_investigation_recommendations(
    material_id,
    investigation_goal,
    initial_state,
    limitations,
    recommendations,
    db_path=None,
):
    """Persist one investigation and its proposed experiments atomically.
    Recommendations are plans only. No experimental result, measurement
    provenance, or completed-experiment status is created here.
    """
    if not isinstance(material_id, str) or not material_id.strip():
        raise MatxRepositoryError("material_id must be a non-empty string.")
    if not isinstance(investigation_goal, str) or not investigation_goal.strip():
        raise MatxRepositoryError("investigation_goal must be a non-empty string.")
    if not isinstance(initial_state, dict):
        raise MatxRepositoryError("initial_state must be an object.")
    if not isinstance(limitations, list):
        raise MatxRepositoryError("limitations must be a list.")
    if not isinstance(recommendations, list) or not recommendations:
        raise MatxRepositoryError(
            "At least one recommendation is required for persistence."
        )
    investigation_id = uuid.uuid4().hex
    experiment_rows = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise MatxRepositoryError("Each recommendation must be an object.")
        category = recommendation.get("category")
        method = recommendation.get("method")
        if not isinstance(category, str) or not category.strip():
            raise MatxRepositoryError("Recommendation category is required.")
        if not isinstance(method, str) or not method.strip():
            raise MatxRepositoryError("Recommendation method is required.")
        parameters = {
            "recommendation": recommendation,
            "recommendation_category": category,
            "material_family": initial_state.get("material_family"),
            "score_type": recommendation.get("score_type"),
            "persistence_semantics": "PROPOSED_EXPERIMENT_NOT_PERFORMED",
        }
        _validate_json_value(parameters, "parameters_json")
        _validate_json_value(recommendation, "recommendation")
        experiment_rows.append({
            "experiment_id": uuid.uuid4().hex,
            "experiment_type": category.upper(),
            "objective": recommendation.get("reason"),
            "recommended_measurement": method,
            "experimental_method": method,
            "parameters_json": json.dumps(
                parameters, allow_nan=False, separators=(",", ":")
            ),
        })
    _validate_json_value(initial_state, "initial_state_json")
    _validate_json_value(limitations, "limitations_json")
    initial_json = json.dumps(
        initial_state, allow_nan=False, separators=(",", ":")
    )
    limitations_json = json.dumps(
        limitations, allow_nan=False, separators=(",", ":")
    )
    connection = _connect(db_path)
    try:
        material = _execute(connection, 
            "SELECT material_id FROM materials WHERE material_id = ?",
            (material_id,),
        ).fetchone()
        if material is None:
            raise MatxRepositoryError(
                f"Material not found: {material_id}"
            )
        with _transaction(connection):
            _execute(connection, 
                """
                INSERT INTO investigations (
                    investigation_id, material_id, investigation_goal,
                    status, initial_state_json, limitations_json
                ) VALUES (?, ?, ?, 'OPEN', ?, ?)
                """,
                (
                    investigation_id, material_id, investigation_goal,
                    initial_json, limitations_json,
                ),
            )
            for experiment in experiment_rows:
                _execute(connection, 
                    """
                    INSERT INTO experiments (
                        experiment_id, material_id, investigation_id,
                        experiment_type, objective, recommended_measurement,
                        experimental_method, parameters_json, status,
                        result_json, provenance_classification
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PLANNED', NULL, NULL)
                    """,
                    (
                        experiment["experiment_id"],
                        material_id,
                        investigation_id,
                        experiment["experiment_type"],
                        experiment["objective"],
                        experiment["recommended_measurement"],
                        experiment["experimental_method"],
                        experiment["parameters_json"],
                    ),
                )
        return {
            "investigation_id": investigation_id,
            "investigation_status": "OPEN",
            "experiment_count": len(experiment_rows),
            "experiments": [
                {
                    "experiment_id": item["experiment_id"],
                    "experiment_type": item["experiment_type"],
                    "status": "PLANNED",
                    "result_json": None,
                    "provenance_classification": None,
                }
                for item in experiment_rows
            ],
            "persistence_semantics": (
                "Recommendations are saved as proposed experiments only. "
                "No physical experiment or measurement is claimed."
            ),
        }
    except _integrity_errors() as exc:
        raise MatxRepositoryError(
            f"Database rejected investigation persistence: {exc}"
        ) from exc
    finally:
        connection.close()
