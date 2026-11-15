from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(".")
MODEL_DIR = ROOT / "models"
AD_DIR = ROOT / "training_results" / "applicability_domain_v3_retry"
models = {
    "tensile_strength": MODEL_DIR / "steelbench_tensile_extratrees.joblib",
    "yield_strength": MODEL_DIR / "steelbench_yield_extratrees.joblib",
}
print("=" * 100)
print("PROMOTED MODEL ↔ APPLICABILITY V3 COMPATIBILITY AUDIT")
print("=" * 100)
for target, path in models.items():
    print(f"\n{'-' * 80}")
    print(f"TARGET: {target}")
    print("MODEL:", path)
    print("EXISTS:", path.exists())
    if not path.exists():
        continue
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        print("Artifact type: dict")
        print("Keys:", list(artifact.keys()))
        pipeline = artifact.get("pipeline", artifact.get("model"))
    else:
        pipeline = artifact
    if pipeline is None:
        print("Could not identify pipeline inside artifact.")
        continue
    print("Pipeline type:", type(pipeline).__name__)
    print("Pipeline steps:", list(getattr(pipeline, "named_steps", {}).keys()))
    feature_names = getattr(pipeline, "feature_names_in_", None)
    if feature_names is not None:
        print("Pipeline feature names:", list(feature_names))
    preprocessing = getattr(pipeline, "named_steps", {}).get("preprocessing")
    if preprocessing is not None:
        print("\nPREPROCESSOR TRANSFORMERS:")
        for name, transformer, columns in preprocessing.transformers_:
            if name == "remainder":
                continue
            print(f"\n  Transformer: {name}")
            print("  Columns:", list(columns))
            if hasattr(transformer, "named_steps"):
                for step_name, step in transformer.named_steps.items():
                    print(f"  Step: {step_name} | Type: {type(step).__name__}")
                    if hasattr(step, "statistics_"):
                        print("  Imputation statistics:")
                        for feature, value in zip(columns, step.statistics_):
                            print(f"    {feature}: {value}")
                    if hasattr(step, "categories_"):
                        print("  Encoded categories:")
                        for feature, values in zip(columns, step.categories_):
                            print(f"    {feature}: {list(values)}")
    report_path = (
        ROOT / "training_results" / "regression" /
        f"steelbench_{target.split('_')[0]}_candidate_training_report.json"
    )
    print("\nCandidate report exists:", report_path.exists())
feature_file = AD_DIR / "feature_domain_summary.csv"
fold_file = AD_DIR / "fold_applicability_summary.csv"
print("\n" + "=" * 100)
print("APPLICABILITY V3 FEATURE DEFINITIONS")
print("=" * 100)
if feature_file.exists():
    features = pd.read_csv(feature_file)
    for target in ["tensile_strength", "yield_strength"]:
        subset = features[features["target"] == target]
        print(f"\n{target}:")
        print("Rows:", len(subset))
        print("Variants:", subset["variant"].dropna().unique().tolist())
        print("Held-out sources:", subset["held_out_source"].dropna().unique().tolist())
        columns = [
            c for c in [
                "variant", "feature", "feature_type",
                "training_status", "eligible_for_distance",
                "training_valid_records", "training_missing_fraction",
                "training_unique_values", "training_min",
                "training_max", "training_q01", "training_q99"
            ]
            if c in subset.columns
        ]
        print(subset[columns].to_string(index=False))
if fold_file.exists():
    folds = pd.read_csv(fold_file)
    print("\n" + "=" * 100)
    print("APPLICABILITY V3 FOLD SUMMARY")
    print("=" * 100)
    for target in ["tensile_strength", "yield_strength"]:
        subset = folds[folds["target"] == target]
        print(f"\n{target}: {len(subset)} fold rows")
        columns = [
            c for c in [
                "held_out_source", "variant",
                "features_requested", "eligible_feature_count",
                "train_records", "calibration_records", "test_records",
                "train_grades", "calibration_grades", "test_grades",
                "calibration_q95", "mean_eligible_evidence_fraction"
            ]
            if c in subset.columns
        ]
        print(subset[columns].to_string(index=False))
print("\nREAD-ONLY COMPATIBILITY AUDIT COMPLETE.")
