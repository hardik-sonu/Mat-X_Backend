import math
from app.services.evidence_registry_service import list_evidence_records
from app.engines.damage_condition_engine import assess_damage_condition
from app.engines.active_investigation_engine import (
    recommend_next_investigation,
)
from app.engines.material_identity_engine import (
    identify_material_candidates,
)
from app.services.microstructure_service import (
    get_unified_microstructure_record,
)
from app.services.process_history_service import (
    get_heat_process_history,
    get_grade_process_history,
)
def _has_value(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        unavailable_values = {
            "", "unknown", "not known", "not assessed",
            "not available", "unavailable", "not evaluated",
            "not inspected", "not provided", "n/a", "na",
            "none", "null",
        }
        return normalized not in unavailable_values
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str)
            and bool(key.strip())
            and _has_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(_has_value(item) for item in value)
    return False
REGISTRY_CATEGORY_MAPPING = {
    "chemical_composition": "chemical_composition",
    "microstructure": "microstructure",
    "phase_identification": "phase_information",
    "grain_characterization": "grain_size",
    "mechanical_testing": "mechanical_properties",
    "mechanical_property": "mechanical_properties",
    "process_history": "process_history",
    "damage_assessment": "damage_condition",
}
def _registry_record_has_usable_value(record):
    if not isinstance(record, dict):
        return False
    if record.get("validation_status") != "VALIDATED":
        return False
    if record.get("provenance_classification") not in (
        "SOURCE_VALUE",
        "DERIVED_VALUE",
    ):
        return False
    return _has_value(record.get("value"))
def _get_microstructure_features(microstructure_record):
    if not isinstance(microstructure_record, dict):
        return {}
    features = microstructure_record.get("microstructure_features")
    return features if isinstance(features, dict) else {}
def _is_finite_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
def _has_valid_phase_measurements(features):
    fractions = features.get(
        "phase_fractions_indexed_points"
    )
    if not isinstance(fractions, dict) or not fractions:
        return False
    for phase_data in fractions.values():
        if not isinstance(phase_data, dict):
            continue
        count = phase_data.get("point_count")
        fraction = phase_data.get("fraction")
        phase_id = phase_data.get("phase_id")
        if (
            isinstance(phase_id, int)
            and not isinstance(phase_id, bool)
            and phase_id > 0
            and isinstance(count, int)
            and count > 0
            and _is_finite_number(fraction)
            and 0.0 <= float(fraction) <= 1.0
        ):
            return True
    return False
def _has_valid_grain_measurements(features):
    grain_count = features.get("grain_count")
    mean_diameter = features.get("grain_eq_diameter_um_mean")
    return (
        isinstance(grain_count, int)
        and not isinstance(grain_count, bool)
        and grain_count > 0
        and _is_finite_number(mean_diameter)
        and float(mean_diameter) > 0
        and features.get("grain_size_method") != "UNAVAILABLE"
    )
def _has_valid_microstructure_measurements(features):
    total_points = features.get("total_points")
    if (
        not isinstance(total_points, int)
        or isinstance(total_points, bool)
        or total_points <= 0
    ):
        return False
    return any((
        _has_valid_phase_measurements(features),
        _has_valid_grain_measurements(features),
        _is_finite_number(features.get("kam_deg_mean")),
        _is_finite_number(features.get("mad_deg_mean")),
    ))
def _validated_registry_evidence_records(records):
    fields = (
        "evidence_id",
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
    )
    return [
        {field: record.get(field) for field in fields}
        for record in records
    ]
def _registry_evidence_provenance(records):
    classifications = {
        record.get("provenance_classification")
        for record in records
    }
    if not classifications:
        return "UNAVAILABLE"
    if len(classifications) == 1:
        return next(iter(classifications))
    return "MIXED_SOURCE_AND_DERIVED"
def reconstruct_material_state(request):
    composition = request.composition or {}
    measured_properties = request.measured_properties or {}
    # Validate explicitly supplied measured property values.
    for name, value in measured_properties.items():
        if not name.strip():
            raise ValueError("Measured property names cannot be empty.")
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Measured property '{name}' must be finite."
            )
    composition_available = any(
        _is_finite_number(value) for value in composition.values()
    )
    measured_properties_available = bool(measured_properties)
    damage_available = False  # Only validated registry observations count.
    # Registry records are retained separately from request-supplied evidence.
    registry_records = list_evidence_records(
        material_id=request.material_id,
        limit=1000,
    )
    registry_records_by_category = {
        category: [] for category in set(REGISTRY_CATEGORY_MAPPING.values())
    }
    for record in registry_records:
        if not isinstance(record, dict):
            continue
        mapped_category = REGISTRY_CATEGORY_MAPPING.get(
            record.get("evidence_category")
        )
        if mapped_category is not None:
            registry_records_by_category[mapped_category].append(record)
    validated_registry_categories = {
        category: any(
            _registry_record_has_usable_value(record)
            for record in records
        )
        for category, records in registry_records_by_category.items()
    }
    validated_mechanical_records = [
        record
        for record in registry_records_by_category["mechanical_properties"]
        if _registry_record_has_usable_value(record)
    ]
    validated_records_by_category = {
        category: [
            record for record in records
            if _registry_record_has_usable_value(record)
        ]
        for category, records in registry_records_by_category.items()
    }
    damage_condition_assessment = assess_damage_condition(
        material_id=request.material_id,
        records=registry_records,
    )
    damage_available = (
        damage_condition_assessment["observation_count"] > 0
    )
    # The generic registry mapping is intentionally overridden here:
    # only the dedicated damage engine's accepted observations count.
    validated_registry_categories["damage_condition"] = damage_available
    # Material identity remains a candidate assessment.
    if sum(value is not None for value in composition.values()) >= 3:
        try:
            identity = identify_material_candidates(
                composition=composition,
                material_family=request.material_family,
                top_k=request.identity_top_k,
            )
        except ValueError as exc:
            message = str(exc)
            if not message.startswith(
                "No reference profiles found for family:"
            ):
                raise
            identity = {
                "status": "REFERENCE_FAMILY_UNSUPPORTED",
                "material_family_requested": request.material_family,
                "recommendations": [],
                "methodology": {
                    "reference_filter_applied": True,
                    "definitive_identification": False,
                    "candidate_matching_performed": False,
                },
                "limitations": [
                    message,
                    "No candidates were returned because the requested "
                    "family has no exact matching reference profiles.",
                    "The family filter was not removed or broadened.",
                    "Other material-state evidence remains available.",
                ],
            }
    else:
        identity = {
            "status": "INSUFFICIENT_EVIDENCE",
            "recommendations": [],
            "methodology": {
                "minimum_measured_elements_required": 3,
                "definitive_identification": False,
            },
            "limitations": [
                "At least three measured chemical elements are required "
                "for reference candidate matching."
            ],
        }
    # Retrieve EBSD characterization only when a map is explicitly supplied.
    microstructure = None
    if request.microstructure_map_id:
        microstructure = get_unified_microstructure_record(
            request.microstructure_map_id
        )
    features = _get_microstructure_features(microstructure)
    microstructure_available = _has_valid_microstructure_measurements(
        features
    )
    phase_available = _has_valid_phase_measurements(features)
    grain_available = _has_valid_grain_measurements(features)
    # Process history is kept separate by heat and grade.
    process_history = {
        "heat_level": None,
        "grade_level": None,
        "heat_lookup_status": "NOT_REQUESTED",
        "grade_lookup_status": "NOT_REQUESTED",
        "data_source_status": "NOT_REQUESTED",
    }
    process_queries_requested = bool(request.heat_id or request.grade_id)
    process_query_successes = 0
    process_query_failures = 0
    if request.heat_id:
        try:
            process_history["heat_level"] = (
                get_heat_process_history(request.heat_id)
            )
            process_query_successes += 1
            heat_record = process_history["heat_level"]
            process_history["heat_lookup_status"] = (
                "MATCHED"
                if isinstance(heat_record, dict)
                and heat_record.get("record_count", 0) > 0
                else "NO_MATCH"
            )
        except FileNotFoundError:
            process_query_failures += 1
            process_history["heat_lookup_status"] = (
                "DATA_SOURCE_UNAVAILABLE"
            )
    if request.grade_id:
        try:
            process_history["grade_level"] = (
                get_grade_process_history(request.grade_id)
            )
            process_query_successes += 1
            grade_record = process_history["grade_level"]
            process_history["grade_lookup_status"] = (
                "MATCHED"
                if isinstance(grade_record, dict)
                and grade_record.get("record_count", 0) > 0
                else "NO_MATCH"
            )
        except FileNotFoundError:
            process_query_failures += 1
            process_history["grade_lookup_status"] = (
                "DATA_SOURCE_UNAVAILABLE"
            )
    if process_queries_requested:
        if process_query_failures == 0:
            process_history["data_source_status"] = "AVAILABLE"
        elif process_query_successes > 0:
            process_history["data_source_status"] = "PARTIALLY_UNAVAILABLE"
        else:
            process_history["data_source_status"] = (
                "PROCESS_DATA_SOURCE_UNAVAILABLE"
            )
    heat_available = bool(
        process_history["heat_level"]
        and process_history["heat_level"].get("record_count", 0) > 0
    )
    grade_available = bool(
        process_history["grade_level"]
        and process_history["grade_level"].get("record_count", 0) > 0
    )
    process_available = heat_available or grade_available
    validated_process_records = validated_records_by_category[
        "process_history"
    ]
    process_history["validated_registry_records"] = (
        _validated_registry_evidence_records(validated_process_records)
    )
    # These keys exactly match the existing seven-category investigation engine.
    investigation_evidence = {
        "chemical_composition": (
            composition_available
            or validated_registry_categories["chemical_composition"]
        ),
        "microstructure": (
            microstructure_available
            or validated_registry_categories["microstructure"]
        ),
        "phase_information": (
            phase_available
            or validated_registry_categories["phase_information"]
        ),
        "grain_size": (
            grain_available
            or validated_registry_categories["grain_size"]
        ),
        "mechanical_properties": (
            measured_properties_available
            or validated_registry_categories["mechanical_properties"]
        ),
        "process_history": (
            process_available
            or validated_registry_categories["process_history"]
        ),
        "damage_condition": (
            damage_available
            or validated_registry_categories["damage_condition"]
        ),
    }
    investigation = recommend_next_investigation(
        material_id=request.material_id,
        material_family=request.material_family,
        evidence=investigation_evidence,
        uncertainty=request.uncertainty,
        investigation_goal="material_state_reconstruction",
        top_k=request.investigation_top_k,
    )
    validated_composition_records = validated_records_by_category[
        "chemical_composition"
    ]
    validated_microstructure_records = validated_records_by_category[
        "microstructure"
    ]
    validated_phase_records = validated_records_by_category[
        "phase_information"
    ]
    validated_grain_records = validated_records_by_category[
        "grain_size"
    ]
    composition_registry_value = _validated_registry_evidence_records(
        validated_composition_records
    )
    microstructure_registry_value = _validated_registry_evidence_records(
        validated_microstructure_records
    )
    phase_registry_value = _validated_registry_evidence_records(
        validated_phase_records
    )
    grain_registry_value = _validated_registry_evidence_records(
        validated_grain_records
    )
    evidence_items = {
        "chemical_composition": {
            "status": (
                "AVAILABLE"
                if composition_available or validated_composition_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if composition_available
                else _registry_evidence_provenance(
                    validated_composition_records
                )
            ),
            "source": (
                "USER_SUPPLIED"
                if composition_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_composition_records
                    else "USER_SUPPLIED"
                )
            ),
            "value": (
                composition
                if composition_available
                else (
                    composition_registry_value
                    if validated_composition_records else None
                )
            ),
            "validated_registry_record_count": len(
                validated_composition_records
            ),
        },
        "mechanical_properties": {
            "status": (
                "AVAILABLE"
                if measured_properties_available or validated_mechanical_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if measured_properties_available
                else _registry_evidence_provenance(
                    validated_mechanical_records
                )
            ),
            "source": (
                "USER_SUPPLIED"
                if measured_properties_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_mechanical_records
                    else "USER_SUPPLIED"
                )
            ),
            "value": (
                measured_properties
                if measured_properties_available
                else (
                    _validated_registry_evidence_records(
                        validated_mechanical_records
                    )
                    if validated_mechanical_records else None
                )
            ),
            "unit_status": (
                "NOT_SPECIFIED"
                if measured_properties_available
                else (
                    "PER_RECORD_UNITS_AVAILABLE"
                    if validated_mechanical_records
                    else "NOT_SPECIFIED"
                )
            ),
            "validated_registry_record_count": len(
                validated_mechanical_records
            ),
        },
        "microstructure": {
            "status": (
                "AVAILABLE"
                if microstructure_available or validated_microstructure_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if microstructure_available
                else _registry_evidence_provenance(
                    validated_microstructure_records
                )
            ),
            "source": (
                "EBSD_CHARACTERIZATION_DATA"
                if microstructure_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_microstructure_records
                    else "EBSD_CHARACTERIZATION_DATA"
                )
            ),
            "value": (
                microstructure
                if microstructure_available
                else (
                    microstructure_registry_value
                    if validated_microstructure_records else None
                )
            ),
            "validated_registry_record_count": len(
                validated_microstructure_records
            ),
        },
        "phase_information": {
            "status": (
                "AVAILABLE"
                if phase_available or validated_phase_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if phase_available
                else _registry_evidence_provenance(
                    validated_phase_records
                )
            ),
            "source": (
                "EBSD_CHARACTERIZATION_DATA"
                if phase_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_phase_records
                    else "EBSD_CHARACTERIZATION_DATA"
                )
            ),
            "value": (
                {
                    "phase_fractions_all_points": features.get(
                        "phase_fractions_all_points"
                    ),
                    "phase_fractions_indexed_points": features.get(
                        "phase_fractions_indexed_points"
                    ),
                }
                if phase_available
                else (
                    phase_registry_value
                    if validated_phase_records else None
                )
            ),
            "validated_registry_record_count": len(
                validated_phase_records
            ),
        },
        "grain_characterization": {
            "status": (
                "AVAILABLE"
                if grain_available or validated_grain_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if grain_available
                else _registry_evidence_provenance(
                    validated_grain_records
                )
            ),
            "source": (
                "EBSD_CHARACTERIZATION_DATA"
                if grain_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_grain_records
                    else "EBSD_CHARACTERIZATION_DATA"
                )
            ),
            "value": (
                {
                    key: value
                    for key, value in features.items()
                    if (
                        key == "grain_count"
                        or key.startswith("grain_eq_diameter_um_")
                        or key in (
                            "grain_area_mean_um2",
                            "grain_area_median_um2",
                            "grain_size_method",
                        )
                    )
                }
                if grain_available
                else (
                    grain_registry_value
                    if validated_grain_records else None
                )
            ),
            "validated_registry_record_count": len(
                validated_grain_records
            ),
        },
        "process_history": {
            "status": (
                "AVAILABLE"
                if process_available or validated_process_records
                else "UNAVAILABLE"
            ),
            "provenance": (
                "SOURCE_VALUE"
                if process_available
                else _registry_evidence_provenance(
                    validated_process_records
                )
            ),
            "source": (
                "DOCUMENTED_PROCESS_RECORDS"
                if process_available
                else (
                    "VALIDATED_EVIDENCE_REGISTRY"
                    if validated_process_records
                    else "DOCUMENTED_PROCESS_RECORDS"
                )
            ),
            "value": process_history,
            "validated_registry_record_count": len(
                validated_process_records
            ),
        },
        "damage_condition": {
            "status": "AVAILABLE" if damage_available else "UNAVAILABLE",
            "provenance": (
                "SOURCE_VALUE" if damage_available else "UNAVAILABLE"
            ),
            "source": "VALIDATED_EVIDENCE_REGISTRY",
            "value": (
                damage_condition_assessment["observations"]
                if damage_available else None
            ),
            "user_reported_observation": (
                {
                    "status": "USER_REPORTED_UNVERIFIED",
                    "value": request.damage_condition,
                }
                if _has_value(request.damage_condition) else None
            ),
        },
    }
    available_categories = [
        key for key, available in investigation_evidence.items()
        if available
    ]
    missing_categories = [
        key for key, available in investigation_evidence.items()
        if not available
    ]
    total_categories = len(investigation_evidence)
    available_count = len(available_categories)
    return {
        "engine": "MAT-X Unified Material-State Reconstruction Engine",
        "engine_version": "1.0.0",
        "status": "EVIDENCE_AGGREGATED_NOT_CERTIFIED",
        "material": {
            "material_id": request.material_id,
            "material_family": request.material_family,
        },
        "material_identity_assessment": identity,
        "evidence": evidence_items,
        "damage_condition_assessment": damage_condition_assessment,
        "evidence_registry": {
            "status": "RETRIEVED",
            "material_id_filter": request.material_id,
            "record_count": len(registry_records),
            "mapped_category_count": sum(
                bool(records)
                for records in registry_records_by_category.values()
            ),
            "validated_usable_record_count": sum(
                _registry_record_has_usable_value(record)
                for record in registry_records
            ),
            "records": registry_records,
            "category_mapping": REGISTRY_CATEGORY_MAPPING,
            "interpretation": (
                "Registry records are preserved independently. Only "
                "VALIDATED records with usable values and SOURCE_VALUE "
                "or DERIVED_VALUE provenance contribute to investigation "
                "evidence availability. No registry value overwrites "
                "request-supplied evidence."
            ),
        },
        "evidence_summary": {
            "supported_categories": total_categories,
            "available_categories": available_count,
            "missing_categories": missing_categories,
            "available_category_names": available_categories,
            "category_coverage_fraction": round(
                available_count / total_categories, 4
            ),
            "interpretation": (
                "Descriptive evidence availability only; "
                "not a confidence or reliability score."
            ),
        },
        "active_investigation": investigation,
        "scientific_scope": {
            "measured_properties_are_not_model_predictions": True,
            "identity_candidates_are_not_confirmed": True,
            "process_history_is_not_assumed_complete": True,
            "grade_history_is_not_heat_history": True,
            "missing_values_imputed": False,
            "uncertainty_generated_automatically": False,
            "engineering_decision_certified": False,
            "human_review_required": True,
        },
        "limitations": [
            "This service aggregates available evidence; it does not "
            "establish a unique or complete latent material state.",
            "Composition matching is limited to the SteelBench reference "
            "profiles and is not calibrated confidence.",
            "EBSD characterization metadata does not establish "
            "manufacturing history.",
            "Documented process records may be incomplete.",
            "Grade-level records are not substitutes for heat-level records.",
            "Measured property units are not inferred when not supplied.",
            "Investigation priorities are heuristic, not measured "
            "Shannon information gain.",
            "No certified engineering decision or failure diagnosis "
            "is produced.",
        ],
    }

