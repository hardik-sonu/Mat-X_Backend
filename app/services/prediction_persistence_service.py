import hashlib
import json
import uuid
from app.core.database import database_connection, is_postgres
from pathlib import Path
from app.core.model_registry import MODEL_FILES
from app.services.matx_repository_service import (
    create_scientific_record,
    get_scientific_record,
)
def _register_model_version(
    *,
    model_key,
    model_name,
    target_property,
    artifact_path,
    artifact_hash,
    db_path=None,
):
    """Register an immutable model artifact identity, idempotently."""
    metadata = {
        "artifact_filename": artifact_path.name,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "hash_algorithm": "SHA-256",
        "registration_source": "prediction_persistence_service",
    }
    postgres = is_postgres(db_path)
    if postgres:
        insert_sql = """
            INSERT INTO model_versions (
                model_version_id,
                model_key,
                model_name,
                version_label,
                artifact_sha256,
                artifact_path,
                framework,
                target_property,
                metadata_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_key, artifact_sha256) DO NOTHING
        """
        select_sql = """
            SELECT model_version_id
            FROM model_versions
            WHERE model_key = %s AND artifact_sha256 = %s
        """
    else:
        insert_sql = """
            INSERT OR IGNORE INTO model_versions (
                model_version_id,
                model_key,
                model_name,
                version_label,
                artifact_sha256,
                artifact_path,
                framework,
                target_property,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        select_sql = """
            SELECT model_version_id
            FROM model_versions
            WHERE model_key = ? AND artifact_sha256 = ?
        """
    parameters = (
        str(uuid.uuid4()),
        model_key,
        model_name,
        f"sha256-{artifact_hash[:12]}",
        artifact_hash,
        str(artifact_path),
        None,
        target_property,
        json.dumps(metadata, sort_keys=True),
    )
    with database_connection(db_path, timeout=30) as connection:
        connection.execute(insert_sql, parameters)
        record = connection.execute(
            select_sql,
            (model_key, artifact_hash),
        ).fetchone()
        if record is None:
            raise RuntimeError(
                "Model artifact registration could not be verified."
            )
        return record["model_version_id"]

def persist_prediction(
    *,
    material_id,
    target_property,
    prediction_type,
    request_features,
    result,
    model_key,
    db_path=None,
):
    """Persist a prediction only when the caller supplies a known material."""
    if not material_id:
        return {
            "status": "SKIPPED_NO_MATERIAL_ID",
            "prediction_id": None,
            "material_id": None,
        }
    material = get_scientific_record(
        "materials", material_id, db_path=db_path
    )
    if material is None:
        raise ValueError(
            f"Unknown material_id: {material_id}. Register the material first."
        )
    artifact_path = Path(MODEL_FILES[model_key])
    if not artifact_path.is_file():
        raise RuntimeError(
            f"Model artifact is unavailable: {artifact_path.name}"
        )
    digest = hashlib.sha256()
    with artifact_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    artifact_hash = digest.hexdigest()
    model_name = result.get("model", model_key)
    numeric_value = None
    if prediction_type == "NUMERIC":
        numeric_value = result.get("prediction")
        if numeric_value is None:
            raise ValueError("Numeric prediction result is missing its value.")
    model_version_id = _register_model_version(
        model_key=model_key,
        model_name=model_name,
        target_property=target_property,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        db_path=db_path,
    )
    prediction_id = str(uuid.uuid4())
    applicability = {
        key: result.get(key)
        for key in (
            "applicability",
            "input_evidence",
            "predictive_uncertainty",
            "validation",
            "limitations",
            "scope",
        )
        if key in result
    }
    create_scientific_record(
        "prediction_records",
        {
            "prediction_id": prediction_id,
            "material_id": material_id,
            "target_property": target_property,
            "predicted_value": numeric_value,
            "predicted_unit": None,
            "model_name": model_name,
            "model_version": artifact_hash,
            "input_json": request_features,
            "applicability_json": applicability,
            "provenance_classification": "PREDICTED_VALUE",
            "prediction_type": prediction_type,
            "prediction_json": result,
        },
        db_path=db_path,
    )
    return {
        "status": "PERSISTED",
        "prediction_id": prediction_id,
        "material_id": material_id,
        "model_version_id": model_version_id,
        "artifact_sha256": artifact_hash,
        "version_label": f"sha256-{artifact_hash[:12]}",
    }
