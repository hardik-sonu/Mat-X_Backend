import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
TARGET_ARTIFACTS = {
    "tensile_strength": "applicability_tensile_reference_v1.json",
    "yield_strength": "applicability_yield_reference_v1.json",
}
def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _resolve_promoted_model_path(provenance):
    recorded_path = str(provenance.get("promoted_model", "")).strip()
    filename = Path(recorded_path.replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise ValueError("Applicability provenance has an invalid promoted model path.")
    model_path = MODEL_DIR / filename
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Promoted model required by applicability reference "
            f"is missing from the project models directory: {filename}"
        )
    return model_path
@lru_cache(maxsize=2)
def load_reference(target):
    if target not in TARGET_ARTIFACTS:
        raise ValueError(f"Unsupported applicability target: {target}")
    path = MODEL_DIR / TARGET_ARTIFACTS[target]
    if not path.exists():
        raise FileNotFoundError(
            f"Applicability reference artifact not found: {path}"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "EXPERIMENTAL_RUNTIME_SCREENING_REFERENCE":
        raise RuntimeError("Applicability reference status is invalid.")
    if artifact.get("target") != target:
        raise RuntimeError("Applicability reference target mismatch.")
    provenance = artifact["provenance"]
    model_path = _resolve_promoted_model_path(provenance)
    actual_hash = _sha256(model_path)
    if actual_hash != provenance["promoted_model_sha256"]:
        raise RuntimeError(
            "Promoted model hash does not match applicability reference."
        )
    features = artifact["features"]["all"]
    numeric = artifact["features"]["numeric"]
    categorical = artifact["features"]["categorical"]
    records = artifact["training_reference"]["records_data"]
    if len(features) != 13:
        raise RuntimeError("Expected exactly 13 applicability features.")
    if len(records) != artifact["training_reference"]["records"]:
        raise RuntimeError("Applicability reference record count mismatch.")
    if len(numeric) + len(categorical) != 13:
        raise RuntimeError("Applicability dimension count mismatch.")
    return artifact
def _validate_complete_input(payload, features):
    missing = [
        feature
        for feature in features
        if payload.get(feature) is None
    ]
    return missing
def _calculate_distances(payload, artifact):
    reference = artifact["reference_statistics"]
    numeric_features = artifact["features"]["numeric"]
    categorical_features = artifact["features"]["categorical"]
    records = artifact["training_reference"]["records_data"]
    dimensions = len(numeric_features) + len(categorical_features)
    query_numeric = []
    for feature in numeric_features:
        value = float(payload[feature])
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be finite.")
        item = reference["numeric"][feature]
        query_numeric.append(
            (value - item["median"]) / item["scale"]
        )
    query_numeric = np.asarray(query_numeric, dtype=float)
    reference_numeric = np.empty(
        (len(records), len(numeric_features)),
        dtype=float,
    )
    for column, feature in enumerate(numeric_features):
        item = reference["numeric"][feature]
        values = np.asarray(
            [record[feature] for record in records],
            dtype=float,
        )
        values[~np.isfinite(values)] = item["median"]
        reference_numeric[:, column] = (
            values - item["median"]
        ) / item["scale"]
    if len(numeric_features):
        numeric_squared = np.sum(
            (reference_numeric - query_numeric[None, :]) ** 2,
            axis=1,
        )
    else:
        numeric_squared = np.zeros(len(records), dtype=float)
    categorical_mismatches = np.zeros(len(records), dtype=float)
    unseen_categories = {}
    for feature in categorical_features:
        query_value = str(payload[feature])
        known = reference["categorical"][feature]
        if query_value not in known:
            unseen_categories[feature] = query_value
        reference_values = np.asarray(
            [
                record[feature]
                if record[feature] is not None
                else reference["missing_category_token"]
                for record in records
            ],
            dtype=str,
        )
        categorical_mismatches += (
            reference_values != query_value
        ).astype(float)
    distances = np.sqrt(
        (numeric_squared + categorical_mismatches) / dimensions
    )
    if not np.isfinite(distances).all():
        raise RuntimeError("Non-finite applicability distances.")
    return distances, unseen_categories
def assess_applicability(target, payload):
    artifact = load_reference(target)
    features = artifact["features"]["all"]
    missing = _validate_complete_input(payload, features)
    if missing:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "method": "EXPERIMENTAL_MIXED_FEATURE_DISTANCE_SCREENING",
            "distance": None,
            "thresholds": artifact["thresholds"],
            "eligible_dimensions": 13,
            "provided_feature_count": len(features) - len(missing),
            "missing_features": missing,
            "unseen_categories": {},
            "range_violations": [],
            "interpretation": (
                "Applicability distance was not calculated because "
                "one or more model inputs are missing. The prediction "
                "pipeline may still impute these inputs separately."
            ),
            "confidence_probability": None,
        }
    distances, unseen_categories = _calculate_distances(
        payload,
        artifact,
    )
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])
    numeric_reference = artifact["reference_statistics"]["numeric"]
    range_violations = []
    for feature in artifact["features"]["numeric"]:
        value = float(payload[feature])
        item = numeric_reference[feature]
        if value < item["q01"] or value > item["q99"]:
            range_violations.append({
                "feature": feature,
                "value": value,
                "q01": item["q01"],
                "q99": item["q99"],
                "type": "OUTSIDE_Q01_Q99",
            })
        if value < item["minimum"] or value > item["maximum"]:
            range_violations.append({
                "feature": feature,
                "value": value,
                "minimum": item["minimum"],
                "maximum": item["maximum"],
                "type": "OUTSIDE_TRAINING_MIN_MAX",
            })
    thresholds = artifact["thresholds"]
    q90 = float(thresholds["q90"])
    q95 = float(thresholds["q95"])
    if nearest_distance > q95:
        status = "OUT_OF_DOMAIN"
    elif (
        nearest_distance > q90
        or range_violations
        or unseen_categories
    ):
        status = "BORDERLINE"
    else:
        status = "IN_DOMAIN"
    return {
        "status": status,
        "method": "EXPERIMENTAL_MIXED_FEATURE_DISTANCE_SCREENING",
        "distance": nearest_distance,
        "thresholds": {
            "q90": q90,
            "q95": q95,
            "calibration_unit": "grade_id",
            "calibration_grades": thresholds["calibration_grades"],
        },
        "eligible_dimensions": 13,
        "provided_feature_count": 13,
        "missing_features": [],
        "unseen_categories": unseen_categories,
        "range_violations": range_violations,
        "interpretation": (
            "Experimental screening against the promoted model's "
            "grade-grouped training reference. This distance is not "
            "predictive uncertainty, probability of correctness, "
            "or a calibrated prediction-error estimate."
        ),
        "confidence_probability": None,
    }
