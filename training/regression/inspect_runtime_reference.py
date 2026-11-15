from pathlib import Path
import pandas as pd
import joblib
ROOT = Path("training_results")
for folder in ["applicability_domain_v3", "applicability_domain_v3_retry"]:
    base = ROOT / folder
    print("\n" + "=" * 90)
    print("REFERENCE DIRECTORY:", folder)
    print("=" * 90)
    summary_path = base / "fold_applicability_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        selected = df[
            df["target"].isin(["tensile_strength", "yield_strength"])
            & df["variant"].isin(["chemistry_2", "chemistry_3", "full_13"])
        ]
        cols = [
            "target", "held_out_source", "variant",
            "eligible_numeric_features", "eligible_categorical_features",
            "eligible_feature_count", "train_records",
            "calibration_records", "calibration_grades",
            "calibration_q90", "calibration_q95",
            "mean_eligible_evidence_fraction", "status_counts"
        ]
        print("\nFOLD SUMMARIES:")
        print(selected[cols].to_string(index=False))
    feature_path = base / "feature_domain_summary.csv"
    if feature_path.exists():
        features = pd.read_csv(feature_path)
        selected = features[
            features["target"].isin(["tensile_strength", "yield_strength"])
            & features["variant"].isin(["chemistry_2", "chemistry_3", "full_13"])
        ]
        cols = [
            "target", "held_out_source", "variant", "feature",
            "feature_type", "training_status", "training_valid_records",
            "training_missing_fraction", "training_unique_values",
            "training_min", "training_max", "training_median",
            "training_scale", "eligible_for_distance"
        ]
        print("\nFEATURE REFERENCE DETAILS:")
        print(selected[cols].to_string(index=False))
print("\n" + "=" * 90)
print("PROMOTED MODEL PIPELINE FEATURE METADATA")
print("=" * 90)
for name in [
    "steelbench_tensile_extratrees.joblib",
    "steelbench_yield_extratrees.joblib",
]:
    path = Path("models") / name
    print("\nMODEL:", name)
    artifact = joblib.load(path)
    print("Artifact type:", type(artifact).__name__)
    if hasattr(artifact, "feature_names_in_"):
        print("feature_names_in_:", list(artifact.feature_names_in_))
    if hasattr(artifact, "named_steps"):
        print("Pipeline steps:", list(artifact.named_steps.keys()))
        for step_name, step in artifact.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                print(step_name, "features:", list(step.feature_names_in_))
    if hasattr(artifact, "n_features_in_"):
        print("n_features_in_:", artifact.n_features_in_)
print("\nRUNTIME REFERENCE INSPECTION COMPLETE")
