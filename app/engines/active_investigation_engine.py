import math
from typing import Any
ENGINE_VERSION = "1.0.0"
SUPPORTED_INVESTIGATION_GOALS = frozenset({
    "material_state_reconstruction",
})
CHARACTERIZATION_METHODS = {
    "chemical_composition": {
        "method": "Chemical composition analysis",
        "techniques": ["OES", "ICP-OES", "XRF"],
        "evidence": ["chemical_composition"],
        "reason": "Establishes elemental composition and supports material identity.",
        "priority": 1,
    },
    "microstructure": {
        "method": "Microstructure characterization",
        "techniques": ["Optical microscopy", "SEM"],
        "evidence": ["microstructure"],
        "reason": "Reveals phases, morphology, inclusions and microstructural condition.",
        "priority": 2,
    },
    "phase_identification": {
        "method": "Phase identification",
        "techniques": ["XRD", "EBSD"],
        "evidence": ["phase_information"],
        "reason": "Helps resolve phase identity and phase distribution.",
        "priority": 3,
    },
    "grain_characterization": {
        "method": "Grain characterization",
        "techniques": ["EBSD", "ASTM E112 microscopy"],
        "evidence": ["grain_size"],
        "reason": "Provides grain-size and grain-boundary evidence.",
        "priority": 4,
    },
    "mechanical_testing": {
        "method": "Mechanical property testing",
        "techniques": ["Tensile test", "Hardness test"],
        "evidence": ["mechanical_properties"],
        "reason": "Provides direct experimental mechanical-property evidence.",
        "priority": 5,
    },
    "process_history": {
        "method": "Process-history investigation",
        "techniques": ["Manufacturing records", "Heat-treatment records"],
        "evidence": ["process_history"],
        "reason": "Constrains the processing and thermal history of the material.",
        "priority": 6,
    },
    "damage_assessment": {
        "method": "Damage and condition assessment",
        "techniques": ["SEM fractography", "NDT", "Metallography"],
        "evidence": ["damage_condition"],
        "reason": "Investigates cracks, defects, degradation and damage mechanisms.",
        "priority": 7,
    },
}
# Limited empirical context from Experiment G (SteelBench v1.0).
# These are grade-held-out prediction-error changes, NOT laboratory
# measurement benefits, calibrated uncertainty reduction, or information gain.
EMPIRICAL_EVIDENCE_CONTEXT = {
    "chemical_composition": {
        "status": "EMPIRICALLY_EVALUATED_ON_STEELBENCH",
        "experiment": "MAT-X Experiment G",
        "dataset": "SteelBench v1.0",
        "validation": "Grade-held-out paired hide-and-reveal evaluation",
        "unit_status": "UNVERIFIED_SOURCE_UNITS",
        "unit_interpretation": (
            "Held-out records span multiple source datasets. Source units "
            "are undocumented; pooled metrics are reported only on the "
            "source-reported numeric scale and are not established as "
            "physically comparable across sources."
        ),
        "targets": {
            "tensile_strength": {
                "test_records": 314,
                "mean_absolute_error_improvement_source_value": 123.3465,
                "ci_95_source_value": [108.0338, 138.7875],
                "records_improved": 249,
                "records_worsened": 65,
            },
            "yield_strength": {
                "test_records": 204,
                "mean_absolute_error_improvement_source_value": 32.3423,
                "ci_95_source_value": [20.1153, 44.8306],
                "records_improved": 130,
                "records_worsened": 74,
            },
        },
        "interpretation": (
            "Composition features reduced prediction error in this "
            "SteelBench evaluation. This does not quantify the benefit "
            "of performing a new laboratory composition measurement."
        ),
    },
    "process_history": {
        "status": "EMPIRICALLY_EVALUATED_ON_STEELBENCH",
        "experiment": "MAT-X Experiment G",
        "dataset": "SteelBench v1.0",
        "validation": "Grade-held-out paired hide-and-reveal evaluation",
        "unit_status": "UNVERIFIED_SOURCE_UNITS",
        "unit_interpretation": (
            "Held-out records span multiple source datasets. Source units "
            "are undocumented; pooled metrics are reported only on the "
            "source-reported numeric scale and are not established as "
            "physically comparable across sources."
        ),
        "targets": {
            "tensile_strength": {
                "test_records": 314,
                "mean_absolute_error_improvement_source_value": -3.4386,
                "ci_95_source_value": [-6.1870, -0.6160],
                "records_improved": 51,
                "records_worsened": 107,
                "records_unchanged": 156,
            },
            "yield_strength": {
                "test_records": 204,
                "mean_absolute_error_improvement_source_value": -3.2562,
                "ci_95_source_value": [-7.1937, 0.5114],
                "records_improved": 45,
                "records_worsened": 37,
                "records_unchanged": 122,
            },
        },
        "interpretation": (
            "Process-history-only features did not demonstrate a "
            "consistent prediction-error improvement in this evaluation. "
            "This does not establish that process history is unimportant "
            "for material-state reconstruction."
        ),
    },
}
def _empirical_context(category: str) -> dict:
    context = EMPIRICAL_EVIDENCE_CONTEXT.get(category)
    if context is not None:
        return context.copy()
    return {
        "status": "NOT_EMPIRICALLY_EVALUATED",
        "experiment": "MAT-X Experiment G",
        "reason": (
            "Experiment G did not evaluate this characterization category."
        ),
    }

def _normalize_evidence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        unavailable_values = {
            "",
            "unknown",
            "not known",
            "not assessed",
            "not available",
            "unavailable",
            "not evaluated",
            "not inspected",
            "not provided",
            "missing",
            "n/a",
            "na",
            "null",
        }
        return normalized not in unavailable_values
    if isinstance(value, dict):
        return any(_normalize_evidence(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_normalize_evidence(item) for item in value)
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False
    return bool(value)

def calculate_expected_information_gain(
    prior_probabilities,
    outcome_probabilities,
    posterior_probabilities,
):
    """Calculate expected Shannon information gain from supplied distributions."""
    def validate(values, label):
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"{label} must contain at least two probabilities.")
        if any(isinstance(v, bool) for v in values):
            raise ValueError(f"{label} cannot contain Boolean probabilities.")
        try:
            values = [float(v) for v in values]
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label} must contain numeric probabilities.")
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError(f"{label} probabilities must be finite and in [0, 1].")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{label} probabilities must sum to 1.")
        return values
    prior = validate(prior_probabilities, "Prior")
    outcomes = validate(outcome_probabilities, "Outcome")
    if not isinstance(posterior_probabilities, (list, tuple)):
        raise ValueError("Posterior probabilities must be a list of distributions.")
    if len(posterior_probabilities) != len(outcomes):
        raise ValueError("Each outcome must have exactly one posterior distribution.")
    posteriors = [
        validate(p, f"Posterior {i}")
        for i, p in enumerate(posterior_probabilities)
    ]
    if any(len(p) != len(prior) for p in posteriors):
        raise ValueError("Every posterior must match the prior state count.")
    reconstructed_prior = [
        sum(outcomes[j] * posteriors[j][i] for j in range(len(outcomes)))
        for i in range(len(prior))
    ]
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)
        for a, b in zip(prior, reconstructed_prior)
    ):
        raise ValueError(
            "Prior and posterior distributions violate the law of total probability."
        )
    def entropy(probabilities):
        return -sum(p * math.log2(p) for p in probabilities if p > 0)
    prior_entropy = entropy(prior)
    expected_posterior_entropy = sum(
        outcomes[i] * entropy(posteriors[i])
        for i in range(len(outcomes))
    )
    information_gain = prior_entropy - expected_posterior_entropy
    if information_gain < -1e-6:
        raise ValueError("Calculated information gain cannot be negative.")
    information_gain = max(0.0, information_gain)
    normalized_gain = (
        information_gain / prior_entropy if prior_entropy > 0 else 0.0
    )
    return {
        "prior_entropy_bits": prior_entropy,
        "expected_posterior_entropy_bits": expected_posterior_entropy,
        "expected_information_gain_bits": information_gain,
        "normalized_information_gain": min(1.0, normalized_gain),
    }
def recommend_next_investigation(
    material_id: str,
    material_family: str,
    evidence: dict,
    uncertainty: dict | None = None,
    information_gain_inputs: dict | None = None,
    investigation_goal: str = "material_state_reconstruction",
    top_k: int = 5,
):
    if not isinstance(evidence, dict):
        raise ValueError("Evidence must be supplied as a dictionary.")
    if not material_id or not material_id.strip():
        raise ValueError("material_id cannot be empty.")
    if not material_family or not material_family.strip():
        raise ValueError("material_family cannot be empty.")
    if (
        not isinstance(investigation_goal, str)
        or investigation_goal not in SUPPORTED_INVESTIGATION_GOALS
    ):
        raise ValueError(
            "Unsupported investigation_goal. Supported goals: "
            + ", ".join(sorted(SUPPORTED_INVESTIGATION_GOALS))
        )
    if top_k < 1 or top_k > len(CHARACTERIZATION_METHODS):
        raise ValueError("top_k must be between 1 and 7.")
    if uncertainty is None:
        uncertainty = {}
    if not isinstance(uncertainty, dict):
        raise ValueError("Uncertainty must be supplied as a dictionary.")
    unknown_uncertainty = set(uncertainty) - set(CHARACTERIZATION_METHODS)
    if unknown_uncertainty:
        raise ValueError(
            f"Unknown uncertainty categories: {sorted(unknown_uncertainty)}"
        )
    supported_evidence_keys = {
        definition["evidence"][0]
        for definition in CHARACTERIZATION_METHODS.values()
    }
    unknown_evidence = set(evidence) - supported_evidence_keys
    if unknown_evidence:
        raise ValueError(
            f"Unknown evidence categories: {sorted(unknown_evidence)}"
        )
    available_evidence = {
        key: _normalize_evidence(evidence.get(key))
        for key in supported_evidence_keys
    }
    recommendations = []
    if information_gain_inputs is None:
        information_gain_inputs = {}
    if not isinstance(information_gain_inputs, dict):
        raise ValueError("information_gain_inputs must be an object.")
    unknown_ig = set(information_gain_inputs) - set(CHARACTERIZATION_METHODS)
    if unknown_ig:
        raise ValueError(
            "Unsupported information-gain categories: "
            + ", ".join(sorted(unknown_ig))
        )
    for category, definition in CHARACTERIZATION_METHODS.items():
        evidence_key = definition["evidence"][0]
        evidence_available = available_evidence.get(evidence_key, False)
        # Missing evidence creates an explicit investigation opportunity.
        missing_score = 1.0 if not evidence_available else 0.0
        # Uncertainty must be supplied explicitly on a normalized 0-1 scale.
        uncertainty_score = uncertainty.get(category, 0.0)
        if isinstance(uncertainty_score, bool):
            raise ValueError(
                f"Uncertainty for {category} must be numeric, not Boolean."
            )
        try:
            uncertainty_score = float(uncertainty_score)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                f"Uncertainty for {category} must be a finite numeric value."
            )
        if not math.isfinite(uncertainty_score):
            raise ValueError(
                f"Uncertainty for {category} must be finite."
            )
        if not 0.0 <= uncertainty_score <= 1.0:
            raise ValueError(
                f"Uncertainty for {category} must be between 0 and 1."
            )
        information_gain = None
        if category in information_gain_inputs:
            model = information_gain_inputs[category]
            if not isinstance(model, dict):
                raise ValueError(
                    f"Information-gain input for {category} must be an object."
                )
            required = {
                "prior_probabilities",
                "outcome_probabilities",
                "posterior_probabilities",
            }
            if set(model) != required:
                raise ValueError(
                    f"Information-gain input for {category} must contain exactly "
                    "prior_probabilities, outcome_probabilities, and "
                    "posterior_probabilities."
                )
            information_gain = calculate_expected_information_gain(
                model["prior_probabilities"],
                model["outcome_probabilities"],
                model["posterior_probabilities"],
            )
            score = information_gain["normalized_information_gain"]
            score_type = "NORMALIZED_EXPECTED_SHANNON_INFORMATION_GAIN"
        else:
            # Existing transparent heuristic retained for backward compatibility.
            score = 0.65 * missing_score + 0.35 * uncertainty_score
            score_type = "HEURISTIC_INVESTIGATION_PRIORITY"
        if (
            evidence_available
            and uncertainty_score == 0
            and (information_gain is None or score == 0)
        ):
            continue
        recommendations.append({
            "category": category,
            "method": definition["method"],
            "suggested_techniques": definition["techniques"],
            "score": round(score, 4),
            "missing_evidence_score": missing_score,
            "uncertainty_score": uncertainty_score,
            "reason": definition["reason"],
            "evidence_status": (
                "MISSING" if not evidence_available else "AVAILABLE"
            ),
            "priority_rank": definition["priority"],
            "score_type": score_type,
            "information_gain": information_gain,
            "empirical_evidence": _empirical_context(category),
        })
    recommendations.sort(
        key=lambda item: (
            -item["score"],
            item["priority_rank"]
        )
    )
    selected = recommendations[:top_k]
    for rank, recommendation in enumerate(selected, start=1):
        recommendation["recommendation_rank"] = rank
    return {
        "engine": "MAT-X Active Investigation Engine",
        "engine_version": ENGINE_VERSION,
        "material_id": material_id,
        "material_family": material_family,
        "investigation_goal": investigation_goal,
        "evidence_summary": {
            "supported_categories": len(supported_evidence_keys),
            "evaluated_categories": len(supported_evidence_keys),
            "provided_categories": len(evidence),
            "provided_categories_definition": (
                "Number of evidence keys explicitly supplied to the engine."
            ),
            "available_categories": sum(available_evidence.values()),
            "missing_categories": [
                key for key, available in available_evidence.items()
                if not available
            ],
        },
        "recommendations": selected,
        "recommendation_count": len(selected),
        "methodology": {
            "scoring": (
                "Normalized expected Shannon information gain when supplied; "
                "otherwise 0.65 x missing-evidence score + "
                "0.35 x supplied uncertainty score"
            ),
            "score_range": [0.0, 1.0],
            "interpretation": (
                "Information gain is calculated from caller-supplied probability "
                "distributions and is not proof of physical test benefit. "
                "Categories without those inputs retain heuristic scoring; "
                "the score types have different evidential meaning."
            ),
            "human_review_required": True,
            "empirical_context": (
                "Experiment G provides limited grade-held-out prediction-error "
                "evidence for chemical composition and process history only. "
                "This context is displayed alongside recommendations but does "
                "not alter heuristic scores or establish laboratory information "
                "gain. Other characterization categories remain unevaluated."
            ),
        },
        "limitations": [
            "Recommendations depend on the evidence supplied by the user.",
            "Uncertainty values must be explicitly supplied and normalized.",
            "The ranking is a heuristic, not a validated probabilistic model.",
            "Material-specific feasibility, cost and expected measurement "
            "outcomes are not yet incorporated.",
            "An engineer must approve the recommended investigation."
        ],
        "status": "HEURISTIC_RECOMMENDATION_NOT_EXPERIMENTALLY_VALIDATED",
    }


