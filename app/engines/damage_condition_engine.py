import math
ENGINE_VERSION = "1.0.0"
DAMAGE_QUANTITIES = {
    "damage_observation",
    "damage_type",
    "condition_observation",
    "corrosion_observation",
    "crack_observation",
    "wear_observation",
    "deformation_observation",
    "fatigue_observation",
    "oxidation_observation",
    "inspection_finding",
    "inspection_result",
}
def _usable_value(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            return False
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str)
            and bool(key.strip())
            and _usable_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_usable_value(item) for item in value)
    return False
def assess_damage_condition(material_id, records):
    if not isinstance(material_id, str) or not material_id.strip():
        raise ValueError("material_id must be a non-empty string.")
    if not isinstance(records, list):
        raise ValueError("records must be a list.")
    observations = []
    excluded_record_count = 0
    for record in records:
        if not isinstance(record, dict):
            excluded_record_count += 1
            continue
        category = record.get("evidence_category")
        quantity = record.get("measured_quantity")
        provenance = record.get("provenance_classification")
        validation = record.get("validation_status")
        value = record.get("value")
        if (
            category != "damage_assessment"
            or not isinstance(quantity, str)
            or quantity.strip().casefold() not in DAMAGE_QUANTITIES
            or validation != "VALIDATED"
            or provenance not in {"SOURCE_VALUE", "DERIVED_VALUE"}
            or not _usable_value(value)
            or record.get("material_id") != material_id
        ):
            excluded_record_count += 1
            continue
        observations.append({
            "evidence_id": record.get("evidence_id"),
            "measured_quantity": quantity,
            "value": value,
            "unit": record.get("unit"),
            "uncertainty_value": record.get("uncertainty_value"),
            "measurement_method": record.get("measurement_method"),
            "test_standard": record.get("test_standard"),
            "source_type": record.get("source_type"),
            "source_name": record.get("source_name"),
            "source_record_id": record.get("source_record_id"),
            "doi_or_url": record.get("doi_or_url"),
            "specimen_id": record.get("specimen_id"),
            "observed_at": record.get("observed_at"),
            "provenance_classification": provenance,
            "validation_status": validation,
        })
    return {
        "engine": "MAT-X Damage and Condition Evidence Assessment Engine",
        "engine_version": ENGINE_VERSION,
        "material_id": material_id,
        "status": (
            "OBSERVATIONS_AVAILABLE_REQUIRES_ENGINEERING_REVIEW"
            if observations else "INSUFFICIENT_EVIDENCE"
        ),
        "evidence_status": (
            "VALIDATED_DAMAGE_RELATED_EVIDENCE_AVAILABLE"
            if observations else "NO_VALIDATED_DAMAGE_RELATED_EVIDENCE"
        ),
        "observation_count": len(observations),
        "excluded_record_count": excluded_record_count,
        "observations": observations,
        "diagnosis_performed": False,
        "severity_assessed": False,
        "remaining_life_predicted": False,
        "limitations": [
            "This engine catalogs validated damage-related observations only.",
            "It does not diagnose damage, determine severity, or establish failure mechanisms.",
            "Absence of recorded observations does not establish absence of damage.",
            "Engineering review and appropriate inspection remain necessary.",
        ],
    }
