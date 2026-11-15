from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_results" / "applicability_domain_v3_retry"
records = pd.read_csv(OUT / "record_applicability.csv")
folds = pd.read_csv(OUT / "fold_applicability_summary.csv")
features = pd.read_csv(OUT / "feature_domain_summary.csv")
print("\n=== RECORD-LEVEL DEVIATION COUNTS ===")
for col in [
    "missing_feature_count",
    "missing_feature_fraction",
    "outside_training_minmax_count",
    "outside_training_q01q99_count",
    "unseen_category_count",
    "constant_deviation_count",
    "unsupported_feature_count",
]:
    if col in records.columns:
        print(f"\n{col}")
        print(records.groupby(
            ["target", "variant"]
        )[col].agg(["count", "mean", "max"]).to_string())
print("\n=== FOLD-LEVEL DEVIATION COUNTS ===")
for col in [
    "records_with_constant_deviations",
    "records_with_unsupported_feature_values",
    "records_with_unseen_categories",
    "records_with_numeric_range_violations",
]:
    if col in folds.columns:
        print(f"\n{col}")
        print(folds.groupby(
            ["target", "variant"]
        )[col].sum().to_string())
print("\n=== FEATURE-LEVEL EXCEPTIONS ===")
exceptions = features[
    features["training_status"].isin([
        "ALL_MISSING_IN_TRAIN",
        "CONSTANT_IN_TRAIN",
    ])
]
print(
    exceptions.groupby(
        ["variant", "feature", "training_status"]
    ).size().to_string()
)
print("\n=== RECORD COLUMNS ===")
print(list(records.columns))
print("\n=== DIAGNOSTIC COMPLETE ===")
