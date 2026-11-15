from pathlib import Path
path = Path("app/services/prediction_service.py")
source = path.read_text(encoding="utf-8")
helper = '''
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
'''
marker = '\ndef predict_tensile(data):'
if "def build_input_evidence(" not in source:
    if marker not in source:
        raise RuntimeError("Could not find predict_tensile insertion point.")
    source = source.replace(marker, helper + marker, 1)
old = '"prediction": prediction,\n        "target":'
new = (
    '"prediction": prediction,\n'
    '        "input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),\n'
    '        "target":'
)
if source.count(old) != 2:
    raise RuntimeError(
        f"Expected exactly 2 prediction response locations; found {source.count(old)}."
    )
source = source.replace(old, new)
path.write_text(source, encoding="utf-8")
print("Updated:", path)
