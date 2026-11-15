import math
import pandas as pd
from app.core.model_registry import MODELS
from app.engines.applicability_engine import assess_applicability
TENSILE_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T", "quench_medium", "steel_family"
]
INCLUSION_FEATURES = [
    "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "S",
    "K", "Ca", "Ti", "Cr", "Mn", "Fe", "La", "Ce"
]
def check_numeric_values(values):
    for key, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
def build_input_evidence(model, payload, feature_names):
    """Report supplied inputs and the fitted pipeline's actual imputations."""
    preprocessing = model.named_steps["preprocessing"]
    imputation_metadata = {}
    for transformer_name, transformer, columns in preprocessing.transformers_:
        if isinstance(transformer, str):
            continue
        pipeline_steps = getattr(transformer, "named_steps", {})
        imputer = pipeline_steps.get("imputer")
        if imputer is None or not hasattr(imputer, "statistics_"):
            continue
        for feature, statistic in zip(columns, imputer.statistics_):
            if hasattr(statistic, "item"):
                statistic = statistic.item()
            imputation_metadata[feature] = {
                "strategy": imputer.strategy,
                "value": statistic
            }
    details = {}
    provided_features = []
    missing_features = []
    for feature in feature_names:
        supplied_value = payload.get(feature)
        if supplied_value is not None:
            provided_features.append(feature)
            details[feature] = {
                "status": "PROVIDED",
                "supplied_value": supplied_value,
                "imputation_strategy": None,
                "imputation_value": None
            }
        else:
            missing_features.append(feature)
            fitted = imputation_metadata.get(feature)
            if fitted is None:
                raise RuntimeError(
                    f"No fitted imputation metadata found for {feature}."
                )
            details[feature] = {
                "status": "IMPUTED_BY_MODEL",
                "supplied_value": None,
                "imputation_strategy": fitted["strategy"],
                "imputation_value": fitted["value"]
            }
    total = len(feature_names)
    return {
        "method": "FITTED_PIPELINE_IMPUTATION_DISCLOSURE",
        "total_features": total,
        "provided_feature_count": len(provided_features),
        "imputed_feature_count": len(missing_features),
        "provided_feature_fraction": len(provided_features) / total,
        "provided_features": provided_features,
        "imputed_features": missing_features,
        "feature_details": details,
        "interpretation": (
            "PROVIDED means a non-null value was supplied in the request; "
            "it does not independently establish laboratory measurement. "
            "IMPUTED_BY_MODEL means the fitted preprocessing pipeline "
            "replaces missing input using its learned statistic."
        )
    }


def build_model_explainability(model, feature_names, top_k=10):
    """Return genuine global ExtraTrees feature importance, never local or causal claims."""
    import math
    estimator = model
    names = list(feature_names or [])
    preprocessor = None
    # Inclusion artifacts store the trained estimator and feature names in a dictionary.
    if isinstance(model, dict):
        estimator = model.get("model", model.get("estimator", model))
        stored_names = model.get("features")
        if stored_names:
            names = list(stored_names)
    # Tensile and yield artifacts are sklearn pipelines.
    steps = getattr(model, "steps", None)
    if steps:
        named_steps = dict(steps)
        estimator = steps[-1][1]
        preprocessor = named_steps.get("preprocessing")
    # Use the actual post-preprocessing feature names where supported.
    if preprocessor is not None:
        get_names = getattr(preprocessor, "get_feature_names_out", None)
        if callable(get_names):
            try:
                transformed_names = list(get_names())
                if transformed_names:
                    names = [str(name) for name in transformed_names]
            except (AttributeError, TypeError, ValueError):
                pass
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return {
            "status": "NOT_AVAILABLE",
            "method": None,
            "scope": "GLOBAL_MODEL_LEVEL_NOT_PER_PREDICTION_OR_CAUSAL",
            "features": [],
            "reason": "The estimator does not expose native feature importances.",
        }
    try:
        values = [float(value) for value in importances]
    except (TypeError, ValueError):
        values = []
    if len(names) != len(values):
        raise RuntimeError(
            f"Feature importance count does not match feature-name count: "
            f"{len(values)} importances for {len(names)} feature names."
        )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError(
            "Feature importance values must be finite and nonnegative."
        )
    if not values or sum(values) <= 0:
        return {
            "status": "NOT_AVAILABLE",
            "method": None,
            "scope": "GLOBAL_MODEL_LEVEL_NOT_PER_PREDICTION_OR_CAUSAL",
            "features": [],
            "reason": "Feature importance values are empty or sum to zero.",
        }
    ranked = sorted(
        zip(names, values),
        key=lambda item: item[1],
        reverse=True,
    )[:max(1, int(top_k))]
    return {
        "status": "AVAILABLE",
        "method": "EXTRA_TREES_IMPURITY_BASED_GLOBAL_FEATURE_IMPORTANCE",
        "scope": "GLOBAL_MODEL_LEVEL_NOT_PER_PREDICTION_OR_CAUSAL",
        "features": [
            {
                "feature": str(name),
                "importance": float(value),
                "rank": rank,
            }
            for rank, (name, value) in enumerate(ranked, start=1)
        ],
        "reason": (
            "Global ExtraTrees impurity-based feature importance. "
            "These values are not per-prediction explanations and do not establish causation."
        ),
    }





def predict_tensile(data):
    model = MODELS.get("tensile_strength")
    if model is None:
        raise RuntimeError("Tensile-strength model is unavailable.")
    payload = data.model_dump()
    check_numeric_values(payload)
    frame = pd.DataFrame(
        [{key: payload[key] for key in TENSILE_FEATURES}],
        columns=TENSILE_FEATURES
    )
    prediction = float(model.predict(frame)[0])
    return {
        "prediction": prediction,
        "explainability": build_model_explainability(model, TENSILE_FEATURES),
        "input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),
        "applicability": assess_applicability("tensile_strength", payload),
        "predictive_uncertainty": {
            "status": "NOT_AVAILABLE",
            "interval_lower": None,
            "interval_upper": None,
            "confidence_level": None,
            "method": None,
            "calibration_status": "NOT_PROMOTED",
            "reason": (
                "No calibration artifact compatible with this promoted "
                "production model has been validated and promoted. "
                "Experimental calibration intervals are not used."
            )
        },
        "target": "Tensile strength",
        "target_unit": "source_dataset_unit_unverified",
        "model": "steelbench_tensile_extratrees",
        "validation": {
            "method": "grade-held-out",
            "test_records": 314,
            "MAE": 134.895,
            "RMSE": 183.634,
            "R2": 0.463
        },
        "limitations": [
            "Initial baseline model, not production validated.",
            "Prediction is limited to the training dataset's domain.",
            "Target unit has not yet been independently verified."
        ]
    }
def predict_yield(data):
    model = MODELS.get("yield_strength")
    if model is None:
        raise RuntimeError("Yield-strength model is unavailable.")
    payload = data.model_dump()
    check_numeric_values(payload)
    frame = pd.DataFrame(
        [{key: payload[key] for key in TENSILE_FEATURES}],
        columns=TENSILE_FEATURES
    )
    prediction = float(model.predict(frame)[0])
    return {
        "prediction": prediction,
        "explainability": build_model_explainability(model, TENSILE_FEATURES),
        "input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),
        "applicability": assess_applicability("yield_strength", payload),
        "predictive_uncertainty": {
            "status": "NOT_AVAILABLE",
            "interval_lower": None,
            "interval_upper": None,
            "confidence_level": None,
            "method": None,
            "calibration_status": "NOT_PROMOTED",
            "reason": (
                "No calibration artifact compatible with this promoted "
                "production model has been validated and promoted. "
                "Experimental calibration intervals are not used."
            )
        },
        "target": "Yield strength",
        "target_unit": "source_dataset_unit_unverified",
        "model": "steelbench_yield_extratrees",
        "validation": {
            "method": "grade-held-out",
            "training_records": 783,
            "test_records": 204,
            "MAE": 95.9567,
            "RMSE": 170.3404,
            "R2": 0.3059,
            "median_baseline_MAE": 140.0245
        },
        "limitations": [
            "Initial baseline model, not production validated.",
            "Prediction is limited to the SteelBench dataset domain.",
            "Target unit has not yet been independently verified.",
            "13 strongly anomalous records were excluded from training."
        ]
    }
def predict_inclusion(data):
    artifact = MODELS.get("inclusion_classifier")
    if artifact is None:
        raise RuntimeError("Inclusion classifier is unavailable.")
    features = artifact["features"]
    supplied = data.features
    missing = [key for key in features if key not in supplied]
    extra = [key for key in supplied if key not in features]
    if missing or extra:
        raise ValueError({
            "missing_features": missing,
            "unexpected_features": extra
        })
    ordered_values = [supplied[key] for key in features]
    check_numeric_values(supplied)
    frame = pd.DataFrame([ordered_values], columns=features)
    model = artifact["model"]
    predicted_class = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    class_labels = artifact["classes"]
    probability_results = {
        class_labels[int(class_id)]: float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }
    return {
        "predicted_class": predicted_class,
        "explainability": build_model_explainability(model, features),
        "classification": class_labels[predicted_class],
        "probabilities": probability_results,
        "model": "matx_inclusion_eds_extratrees",
        "scope": "Ti-ULC steel dataset; EDS chemistry features",
        "validation": {
            "method": "official held-out test set",
            "test_records": 618,
            "accuracy": 0.9725,
            "precision": 0.982,
            "recall": 0.984,
            "F1": 0.983,
            "ROC_AUC": 0.9972
        },
        "limitations": [
            "This is an EDS-feature classifier, not an SEM image detector.",
            "Performance is established on the source dataset, not all steel types."
        ]
    }
