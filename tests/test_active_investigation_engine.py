import pytest
from app.engines.active_investigation_engine import (
    CHARACTERIZATION_METHODS,
    _normalize_evidence,
    recommend_next_investigation,
)
from app.core.characterization import CHARACTERIZATION_CATEGORIES

def test_characterization_categories_match_engine_methods():
    assert set(CHARACTERIZATION_CATEGORIES) == set(CHARACTERIZATION_METHODS)


def test_zero_numeric_evidence_is_available():
    assert _normalize_evidence(0) is True
    assert _normalize_evidence(0.0) is True
def test_false_boolean_evidence_is_unavailable():
    assert _normalize_evidence(False) is False
def test_missing_evidence_values_are_unavailable():
    for value in [None, "", "unknown", "unavailable", "missing", "N/A"]:
        assert _normalize_evidence(value) is False
def test_empty_collection_is_unavailable():
    for value in [[], {}, (), set()]:
        assert _normalize_evidence(value) is False
def test_nonempty_collection_is_available():
    for value in [[0], {"C": 0.0}, ("measured",), {"measured"}]:
        assert _normalize_evidence(value) is True
def test_equal_scores_use_fixed_category_priority():
    result = recommend_next_investigation(
        material_id="MATX-TEST-ORDER",
        material_family="low_alloy_steel",
        evidence={},
        top_k=7,
    )
    actual = [
        item["category"]
        for item in result["recommendations"]
    ]
    expected = [
        category
        for category, definition in sorted(
            CHARACTERIZATION_METHODS.items(),
            key=lambda item: item[1]["priority"],
        )
    ]
    assert actual == expected
def test_supplied_uncertainty_changes_ranking():
    result = recommend_next_investigation(
        material_id="MATX-TEST-UNCERTAINTY",
        material_family="low_alloy_steel",
        evidence={
            "chemical_composition": {"C": 0.3},
            "microstructure": {"grain_structure": "observed"},
            "phase_information": {"phase": "ferrite"},
            "grain_size": {"mean_grain_size": 10},
            "mechanical_properties": {"yield_strength": 500},
            "process_history": {"heat_treatment": "normalized"},
            "damage_condition": {"damage": "none"},
        },
        uncertainty={"microstructure": 1.0},
        top_k=7,
    )
    assert result["recommendations"][0]["category"] == "microstructure"
    assert result["recommendations"][0]["score"] == 0.35
    assert result["recommendations"][0]["evidence_status"] == "AVAILABLE"
@pytest.mark.parametrize("value", [-0.01, 1.01, "invalid", None])
def test_invalid_uncertainty_is_rejected(value):
    with pytest.raises(ValueError):
        recommend_next_investigation(
            material_id="MATX-TEST-INVALID",
            material_family="low_alloy_steel",
            evidence={},
            uncertainty={"microstructure": value},
        )

@pytest.mark.parametrize("value", [True, False])
def test_boolean_uncertainty_is_rejected(value):
    with pytest.raises(ValueError):
        recommend_next_investigation(
            material_id="MATX-TEST-BOOLEAN",
            material_family="low_alloy_steel",
            evidence={},
            uncertainty={"microstructure": value},
        )
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_uncertainty_is_rejected(value):
    with pytest.raises(ValueError):
        recommend_next_investigation(
            material_id="MATX-TEST-NONFINITE",
            material_family="low_alloy_steel",
            evidence={},
            uncertainty={"microstructure": value},
        )

@pytest.mark.parametrize(
    "value",
    [
        "not assessed",
        "not inspected",
        "not provided",
        "not evaluated",
        "not known",
        "null",
    ],
)
def test_unassessed_evidence_is_unavailable(value):
    assert _normalize_evidence(value) is False
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_numeric_evidence_is_unavailable(value):
    assert _normalize_evidence(value) is False

def test_recommendation_rank_is_sequential_and_priority_is_preserved():
    result = recommend_next_investigation(
        material_id="MATX-TEST-SEQUENTIAL-RANK",
        material_family="low_alloy_steel",
        evidence={
            "chemical_composition": {"C": 0.3},
            "mechanical_properties": {"yield_strength": 500},
        },
        top_k=7,
    )
    recommendations = result["recommendations"]
    assert [
        item["recommendation_rank"] for item in recommendations
    ] == list(range(1, len(recommendations) + 1))
    for item in recommendations:
        assert item["priority_rank"] == CHARACTERIZATION_METHODS[
            item["category"]
        ]["priority"]
def test_evidence_summary_distinguishes_provided_and_evaluated():
    result = recommend_next_investigation(
        material_id="MATX-TEST-EVIDENCE-COUNTS",
        material_family="low_alloy_steel",
        evidence={"chemical_composition": {"C": 0.3}},
    )
    summary = result["evidence_summary"]
    assert summary["supported_categories"] == 7
    assert summary["evaluated_categories"] == 7
    assert summary["provided_categories"] == 1
    assert summary["available_categories"] == 1
def test_nested_evidence_requires_at_least_one_usable_value():
    assert _normalize_evidence({"C": None}) is False
    assert _normalize_evidence({"C": "unknown"}) is False
    assert _normalize_evidence({"C": float("nan")}) is False
    assert _normalize_evidence([None, "not assessed"]) is False
    assert _normalize_evidence({"nested": {"value": None}}) is False
    assert _normalize_evidence({"C": None, "Mn": 0.0}) is True
    assert _normalize_evidence([None, 0]) is True
def test_extremely_large_numeric_evidence_is_unavailable():
    assert _normalize_evidence(10 ** 10000) is False
def test_extremely_large_uncertainty_is_rejected():
    with pytest.raises(ValueError, match="finite numeric value"):
        recommend_next_investigation(
            material_id="MATX-TEST-LARGE-UNCERTAINTY",
            material_family="low_alloy_steel",
            evidence={},
            uncertainty={"microstructure": 10 ** 10000},
        )
def test_explicit_no_damage_observation_is_available():
    assert _normalize_evidence("none") is True
    assert _normalize_evidence("no damage observed") is True
    assert _normalize_evidence(None) is False
    assert _normalize_evidence("null") is False


def test_investigation_schema_accepts_all_supported_uncertainty_categories():
    from app.core.characterization import CHARACTERIZATION_CATEGORIES
    from app.schemas.investigation import ActiveInvestigationInput
    request = ActiveInvestigationInput.model_validate({
        "material_id": "MATX-SCHEMA-VALID",
        "material_family": "low_alloy_steel",
        "evidence": {},
        "uncertainty": {category: 0.25 for category in CHARACTERIZATION_CATEGORIES},
    })
    assert len(request.uncertainty) == 7
def test_investigation_schema_rejects_unknown_uncertainty_category():
    from pydantic import ValidationError
    from app.schemas.investigation import ActiveInvestigationInput
    with pytest.raises(ValidationError, match="Unsupported uncertainty categories"):
        ActiveInvestigationInput.model_validate({
            "material_id": "MATX-SCHEMA-INVALID",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "uncertainty": {"unknown_category": 0.5},
        })
def test_material_state_schema_accepts_all_supported_uncertainty_categories():
    from app.core.characterization import CHARACTERIZATION_CATEGORIES
    from app.schemas.material_state import MaterialStateReconstructionInput
    request = MaterialStateReconstructionInput.model_validate({
        "material_id": "MATX-STATE-SCHEMA-VALID",
        "material_family": "low_alloy_steel",
        "uncertainty": {category: 0.25 for category in CHARACTERIZATION_CATEGORIES},
    })
    assert len(request.uncertainty) == 7
def test_material_state_schema_rejects_unknown_uncertainty_category():
    from pydantic import ValidationError
    from app.schemas.material_state import MaterialStateReconstructionInput
    with pytest.raises(ValidationError, match="Unsupported uncertainty categories"):
        MaterialStateReconstructionInput.model_validate({
            "material_id": "MATX-STATE-SCHEMA-INVALID",
            "material_family": "low_alloy_steel",
            "uncertainty": {"unknown_category": 0.5},
        })

def test_supported_investigation_goal_is_returned():
    result = recommend_next_investigation(
        material_id="MATX-GOAL-VALID",
        material_family="low_alloy_steel",
        evidence={},
        investigation_goal="material_state_reconstruction",
    )
    assert result["investigation_goal"] == "material_state_reconstruction"
@pytest.mark.parametrize(
    "goal",
    [
        "unknown_goal",
        "failure_diagnosis",
        "predict_remaining_life",
        "",
        None,
        123,
    ],
)
def test_engine_rejects_unsupported_investigation_goal(goal):
    with pytest.raises(ValueError, match="Unsupported investigation_goal"):
        recommend_next_investigation(
            material_id="MATX-GOAL-INVALID",
            material_family="low_alloy_steel",
            evidence={},
            investigation_goal=goal,
        )
def test_investigation_schema_documents_single_supported_goal():
    from app.schemas.investigation import ActiveInvestigationInput
    request = ActiveInvestigationInput.model_validate({
        "material_id": "MATX-GOAL-SCHEMA",
        "material_family": "low_alloy_steel",
        "evidence": {},
    })
    assert request.investigation_goal == "material_state_reconstruction"
@pytest.mark.parametrize(
    "goal",
    [
        "unknown_goal",
        "failure_diagnosis",
        "predict_remaining_life",
        "",
    ],
)
def test_investigation_schema_rejects_unsupported_goals(goal):
    from pydantic import ValidationError
    from app.schemas.investigation import ActiveInvestigationInput
    with pytest.raises(ValidationError):
        ActiveInvestigationInput.model_validate({
            "material_id": "MATX-GOAL-SCHEMA-INVALID",
            "material_family": "low_alloy_steel",
            "evidence": {},
            "investigation_goal": goal,
        })

def test_composition_recommendation_exposes_experiment_g_evidence():
    result = recommend_next_investigation(
        material_id="MATX-EXP-G-COMPOSITION",
        material_family="low_alloy_steel",
        evidence={},
        top_k=7,
    )
    item = next(
        item for item in result["recommendations"]
        if item["category"] == "chemical_composition"
    )
    empirical = item["empirical_evidence"]
    assert empirical["status"] == "EMPIRICALLY_EVALUATED_ON_STEELBENCH"
    assert empirical["unit_status"] == "UNVERIFIED_SOURCE_UNITS"
    assert "not established as" in empirical["unit_interpretation"]
    assert empirical["targets"]["tensile_strength"]["test_records"] == 314
    assert empirical["targets"]["tensile_strength"][
        "mean_absolute_error_improvement_source_value"
    ] == 123.3465
    assert empirical["targets"]["yield_strength"]["records_improved"] == 130
def test_process_history_recommendation_reports_negative_empirical_result():
    result = recommend_next_investigation(
        material_id="MATX-EXP-G-PROCESS",
        material_family="low_alloy_steel",
        evidence={},
        top_k=7,
    )
    item = next(
        item for item in result["recommendations"]
        if item["category"] == "process_history"
    )
    empirical = item["empirical_evidence"]
    assert empirical["status"] == "EMPIRICALLY_EVALUATED_ON_STEELBENCH"
    assert empirical["unit_status"] == "UNVERIFIED_SOURCE_UNITS"
    assert "not established as" in empirical["unit_interpretation"]
    assert empirical["targets"]["tensile_strength"][
        "mean_absolute_error_improvement_source_value"
    ] < 0
    assert empirical["targets"]["yield_strength"][
        "mean_absolute_error_improvement_source_value"
    ] < 0
def test_unsupported_categories_are_not_claimed_as_empirically_validated():
    result = recommend_next_investigation(
        material_id="MATX-EXP-G-UNSUPPORTED",
        material_family="low_alloy_steel",
        evidence={},
        top_k=7,
    )
    unsupported = {
        "microstructure",
        "phase_identification",
        "grain_characterization",
        "mechanical_testing",
        "damage_assessment",
    }
    for item in result["recommendations"]:
        if item["category"] in unsupported:
            assert item["empirical_evidence"]["status"] == (
                "NOT_EMPIRICALLY_EVALUATED"
            )
def test_experiment_g_context_does_not_change_heuristic_scores():
    result = recommend_next_investigation(
        material_id="MATX-EXP-G-SCORE",
        material_family="low_alloy_steel",
        evidence={},
        top_k=7,
    )
    assert all(
        item["score_type"] == "HEURISTIC_INVESTIGATION_PRIORITY"
        for item in result["recommendations"]
    )
    assert all(item["score"] == 0.65 for item in result["recommendations"])
def test_methodology_explicitly_limits_experiment_g_claims():
    result = recommend_next_investigation(
        material_id="MATX-EXP-G-METHOD",
        material_family="low_alloy_steel",
        evidence={},
    )
    context = result["methodology"]["empirical_context"]
    assert "does not" in context
    assert "Other characterization categories remain unevaluated" in context

def test_expected_information_gain_calculates_binary_entropy_reduction():
    from app.engines.active_investigation_engine import calculate_expected_information_gain
    result = calculate_expected_information_gain(
        [0.5, 0.5], [0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert result["expected_information_gain_bits"] == pytest.approx(1.0)
    assert result["normalized_information_gain"] == pytest.approx(1.0)
def test_expected_information_gain_is_zero_for_uninformative_test():
    from app.engines.active_investigation_engine import calculate_expected_information_gain
    result = calculate_expected_information_gain(
        [0.5, 0.5], [0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]]
    )
    assert result["expected_information_gain_bits"] == pytest.approx(0.0)
    assert result["normalized_information_gain"] == pytest.approx(0.0)
@pytest.mark.parametrize("payload", [
    ([0.4, 0.4], [0.5, 0.5], [[1, 0], [0, 1]]),
    ([0.5, 0.5], [0.7, 0.7], [[1, 0], [0, 1]]),
    ([0.5, 0.5], [0.5, 0.5], [[1, 0]]),
    ([0.5, 0.5], [0.5, 0.5], [[0.8, 0.2], [0.8, 0.2]]),
])
def test_expected_information_gain_rejects_invalid_probability_models(payload):
    from app.engines.active_investigation_engine import calculate_expected_information_gain
    with pytest.raises(ValueError):
        calculate_expected_information_gain(*payload)
def test_information_gain_inputs_are_optional_for_existing_requests():
    from app.schemas.investigation import ActiveInvestigationInput
    request = ActiveInvestigationInput.model_validate({
        "material_id": "MATX-IG-BACKWARD-COMPAT",
        "material_family": "low_alloy_steel",
        "evidence": {},
    })
    assert request.information_gain_inputs is None
def test_engine_uses_calculated_information_gain_when_supplied():
    from app.engines.active_investigation_engine import recommend_next_investigation
    result = recommend_next_investigation(
        material_id="MATX-IG-RANKING",
        material_family="low_alloy_steel",
        evidence={},
        information_gain_inputs={
            "microstructure": {
                "prior_probabilities": [0.5, 0.5],
                "outcome_probabilities": [0.5, 0.5],
                "posterior_probabilities": [[1.0, 0.0], [0.0, 1.0]],
            }
        },
        top_k=7,
    )
    item = next(x for x in result["recommendations"] if x["category"] == "microstructure")
    assert item["score"] == pytest.approx(1.0)
    assert item["score_type"] == "NORMALIZED_EXPECTED_SHANNON_INFORMATION_GAIN"
    assert item["information_gain"]["expected_information_gain_bits"] == pytest.approx(1.0)
def test_engine_keeps_heuristic_for_categories_without_probability_inputs():
    from app.engines.active_investigation_engine import recommend_next_investigation
    result = recommend_next_investigation(
        material_id="MATX-IG-HYBRID",
        material_family="low_alloy_steel",
        evidence={},
        information_gain_inputs={
            "microstructure": {
                "prior_probabilities": [0.5, 0.5],
                "outcome_probabilities": [0.5, 0.5],
                "posterior_probabilities": [[1.0, 0.0], [0.0, 1.0]],
            }
        },
        top_k=7,
    )
    by_category = {x["category"]: x for x in result["recommendations"]}
    assert by_category["microstructure"]["score_type"] == (
        "NORMALIZED_EXPECTED_SHANNON_INFORMATION_GAIN"
    )
    assert by_category["chemical_composition"]["score_type"] == (
        "HEURISTIC_INVESTIGATION_PRIORITY"
    )
    assert by_category["chemical_composition"]["score"] == pytest.approx(0.65)
def test_engine_rejects_unknown_information_gain_category():
    from app.engines.active_investigation_engine import recommend_next_investigation
    with pytest.raises(ValueError, match="Unsupported information-gain categories"):
        recommend_next_investigation(
            material_id="MATX-IG-UNKNOWN",
            material_family="low_alloy_steel",
            evidence={},
            information_gain_inputs={"unknown_category": {}},
        )
