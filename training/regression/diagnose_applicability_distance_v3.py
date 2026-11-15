from pathlib import Path
import pandas as pd
root = Path("training_results/applicability_domain_v2")
folds = pd.read_csv(root / "fold_applicability_summary.csv")
print("Fold summary columns:", folds.columns.tolist())
full = folds[folds["variant"] == "full_13"].copy()
print(full.to_string(index=False))
print("\n=== FULL_13 FEATURE TRAINING STATISTICS ===")
features = pd.read_csv(root / "feature_domain_summary.csv")
selected = features[
    (features["variant"] == "full_13") &
    (features["target"] == "tensile_strength") &
    (features["held_out_source"] == "EMK_spec")
]
columns = [
    c for c in [
        "feature", "feature_type", "training_min", "training_max",
        "training_q01", "training_q99", "training_median",
        "training_scale", "training_missing_fraction",
        "test_missing_fraction", "test_outside_training_minmax",
        "test_outside_training_q01q99"
    ] if c in selected.columns
]
print(selected[columns].to_string(index=False))
print("\n=== FULL_13 STATUS AND DISTANCE BY FOLD ===")
records = pd.read_csv(root / "record_applicability.csv")
full_records = records[records["variant"] == "full_13"]
summary = (
    full_records.groupby(["target", "held_out_source"])
    .agg(
        records=("heat_id", "size"),
        median_distance=("nearest_training_distance", "median"),
        q95_threshold=("training_distance_q95", "first"),
        q99_threshold=("training_distance_q99", "first"),
        median_missing_fraction=("missing_feature_fraction", "median"),
        median_outside_minmax=("outside_training_minmax_count", "median"),
        median_outside_q01q99=("outside_training_q01q99_count", "median"),
        median_unseen_categories=("unseen_category_count", "median"),
    )
    .reset_index()
)
print(summary.to_string(index=False))
