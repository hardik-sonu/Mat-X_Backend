from datetime import datetime, timezone
from app.services.material_state_service import reconstruct_material_state
from app.engines.active_investigation_experiment_evidence import evaluate_experiment_g
from app.services.matx_repository_service import list_scientific_records

def _get_persisted_prediction_results(material_id):
    """Retrieve existing predictions for one material without rerunning models."""
    records = list_scientific_records(
        "prediction_records",
        filters={"material_id": material_id},
        limit=1000,
        offset=0,
    )
    results = []
    for record in records:
        prediction_json = record.get("prediction_json") or {}
        input_json = record.get("input_json") or {}
        applicability_json = record.get("applicability_json") or {}
        results.append({
            "prediction_id": record.get("prediction_id"),
            "material_id": record.get("material_id"),
            "target_property": record.get("target_property"),
            "prediction_type": record.get("prediction_type", "NUMERIC"),
            "predicted_value": record.get("predicted_value"),
            "predicted_unit": record.get("predicted_unit"),
            "prediction": prediction_json,
            "input_features": input_json,
            "model": {
                "name": record.get("model_name"),
                "artifact_sha256": record.get("model_version"),
            },
            "applicability": applicability_json.get("applicability"),
            "input_evidence": applicability_json.get("input_evidence"),
            "predictive_uncertainty": applicability_json.get(
                "predictive_uncertainty"
            ),
            "validation": applicability_json.get("validation"),
            "limitations": applicability_json.get("limitations", []),
            "scope": applicability_json.get("scope"),
            "provenance_classification": record.get(
                "provenance_classification", "PREDICTED_VALUE"
            ),
            "created_at": record.get("created_at"),
        })
    return {
        "status": (
            "PERSISTED_PREDICTIONS_INCLUDED"
            if results else "NO_PERSISTED_PREDICTIONS"
        ),
        "record_count": len(results),
        "results": results,
        "interpretation": (
            "These are previously persisted model predictions for this "
            "material. They are not measured evidence and were not generated "
            "again while creating this report. Model artifact hashes, inputs, "
            "applicability, uncertainty, and provenance are retained. "
            "Predictions require qualified engineering review."
        ),
    }

def generate_engineering_report(request):
    """Build a traceable engineering report from unified state evidence."""
    state = reconstruct_material_state(request)
    evidence = state.get("evidence", {})
    evidence_summary = state.get("evidence_summary", {})
    registry = state.get("evidence_registry", {})
    investigation = state.get("active_investigation", {})
    try:
        experiment_g_evidence = evaluate_experiment_g()
    except (FileNotFoundError, ValueError, OSError) as exc:
        experiment_g_evidence = {
            "experiment": "EXPERIMENT_G",
            "status": "UNAVAILABLE",
            "reason": str(exc),
        }
    damage_evidence = evidence.get("damage_condition", {})
    user_reported_damage = damage_evidence.get(
        "user_reported_observation"
    )
    available = evidence_summary.get("available_categories", 0)
    total = evidence_summary.get("supported_categories", 0)
    return {
        "report": {
            "title": "MAT-X Engineering Material-State Report",
            "report_schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_status": "EVIDENCE_AGGREGATED_NOT_CERTIFIED",
        },
        "material": state.get("material", {}),
        "executive_summary": {
            "evidence_categories_available": available,
            "evidence_categories_supported": total,
            "missing_evidence_categories": evidence_summary.get(
                "missing_categories", []
            ),
            "coverage_fraction": evidence_summary.get(
                "category_coverage_fraction"
            ),
            "coverage_interpretation": (
                "Descriptive evidence availability only; it is not a "
                "confidence, reliability, or probability score."
            ),
            "engineering_decision": "HUMAN_REVIEW_REQUIRED",
        },
        "source_evidence": {
            "categories": evidence,
            "interpretation": (
                "SOURCE_VALUE identifies supplied source data; it does "
                "not independently establish laboratory verification. "
                "Unavailable categories are not treated as measured values."
            ),
        },
        "validated_evidence_registry": {
            "status": registry.get("status", "UNAVAILABLE"),
            "record_count": registry.get("record_count", 0),
            "validated_usable_record_count": registry.get(
                "validated_usable_record_count", 0
            ),
            "records": registry.get("records", []),
            "interpretation": (
                "Registry records are reported separately from request "
                "evidence. Only validated records with usable values and "
                "SOURCE_VALUE or DERIVED_VALUE provenance contribute to "
                "investigation evidence availability."
            ),
        },
        "material_identity_assessment": state.get(
            "material_identity_assessment"
        ),
        "damage_condition_assessment": state.get(
            "damage_condition_assessment"
        ),
        "user_reported_damage_observation": (
            user_reported_damage
            if user_reported_damage is not None
            else {"status": "NOT_REPORTED", "value": None}
        ),
        "active_investigation": {
            "result": investigation,
            "experiment_g_evidence": experiment_g_evidence,
            "interpretation": (
                "The recommendation engine currently provides heuristic "
                "priorities. Experiment G supplies limited grade-held-out "
                "prediction-error ablation evidence for composition and "
                "process feature groups on its evaluated targets only. "
                "It does not measure Shannon information gain or the "
                "uncertainty reduction caused by physically acquiring a test."
            ),
        },
        "prediction_results": _get_persisted_prediction_results(
            request.material_id
        ),
        "scientific_scope": state.get("scientific_scope", {}),
        "limitations": state.get("limitations", []),
        "engineering_disclaimer": (
            "This report is an evidence-aggregation aid. It is not a "
            "certified engineering assessment, failure diagnosis, remaining "
            "useful life estimate, or substitute for laboratory testing "
            "and qualified engineering judgment."
        ),
    }
