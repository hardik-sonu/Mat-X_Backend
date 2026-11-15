from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path("training_results/applicability_domain_v3_retry")
OUT = Path("training_results/applicability_domain_reason_audit_v1")
OUT.mkdir(parents=True, exist_ok=True)
records = pd.read_csv(ROOT / "record_applicability.csv")
features = pd.read_csv(ROOT / "feature_domain_summary.csv")
# Derive explicit reason flags from saved assessment fields.
records["reason_distance_q90"] = (
    records["nearest_training_distance"] > records["calibration_q90"]
)
records["reason_distance_q95"] = (
    records["calibration_q95"].notna()
    & (records["nearest_training_distance"] > records["calibration_q95"])
)
records["reason_training_minmax"] = (
    records["outside_training_minmax_count"] > 0
)
records["reason_training_q01q99"] = (
    records["outside_training_q01q99_count"] > 0
)
records["reason_unseen_category"] = (
    records["unseen_category_count"] > 0
)
records["reason_constant_deviation"] = (
    records["constant_reference_deviation_count"] > 0
)
records["reason_unsupported_value"] = (
    records["unsupported_feature_value_count"] > 0
)
minimum_observed = np.ceil(
    records["eligible_feature_count"] * 0.50
)
records["reason_insufficient_evidence"] = (
    records["eligible_features_observed"] < minimum_observed
)
reason_columns = [
    "reason_distance_q90",
    "reason_distance_q95",
    "reason_training_minmax",
    "reason_training_q01q99",
    "reason_unseen_category",
    "reason_constant_deviation",
    "reason_unsupported_value",
    "reason_insufficient_evidence",
]
# Counts are overlapping: one record can have multiple reasons.
group_columns = ["target", "variant", "applicability_status"]
rows = []
for keys, group in records.groupby(group_columns, dropna=False):
    row = dict(zip(group_columns, keys))
    row["record_count"] = len(group)
    for column in reason_columns:
        row[column.replace("reason_", "records_")] = int(
            group[column].sum()
        )
    row["mean_missing_feature_fraction"] = float(
        group["missing_feature_fraction"].mean()
    )
    row["mean_nearest_training_distance"] = float(
        group["nearest_training_distance"].mean()
    )
    row["mean_q90_threshold"] = float(
        group["calibration_q90"].mean()
    )
    rows.append(row)
reason_summary = pd.DataFrame(rows)
reason_summary.to_csv(
    OUT / "status_reason_summary.csv", index=False
)
# Overall reason frequencies by target and feature variant.
overall_rows = []
for keys, group in records.groupby(
    ["target", "variant"], dropna=False
):
    row = {
        "target": keys[0],
        "variant": keys[1],
        "record_count": len(group),
    }
    for column in reason_columns:
        row[column.replace("reason_", "records_")] = int(
            group[column].sum()
        )
        row[column.replace("reason_", "fraction_")] = float(
            group[column].mean()
        )
    overall_rows.append(row)
overall = pd.DataFrame(overall_rows)
overall.to_csv(
    OUT / "overall_reason_frequencies.csv", index=False
)
# Feature-level diagnostics: identify features that are unsupported
# or deviate from constant training references.
feature_columns = [
    "feature",
    "feature_type",
    "training_status",
    "training_valid_records",
    "training_missing_fraction",
    "training_unique_values",
    "eligible_for_distance",
    "test_valid_records",
    "test_missing_fraction",
    "unsupported_test_records",
    "constant_deviation_records",
    "target",
    "held_out_source",
    "variant",
]
available = [c for c in feature_columns if c in features.columns]
feature_output = features[available].copy()
feature_output.to_csv(
    OUT / "feature_level_domain_diagnostics.csv", index=False
)
print("\n=== STATUS REASON SUMMARY ===")
print(
    reason_summary.sort_values(
        ["target", "variant", "applicability_status"]
    ).to_string(index=False)
)
print("\n=== OVERALL REASON FREQUENCIES ===")
print(overall.to_string(index=False))
print("\n=== FEATURE-LEVEL EXCEPTIONS ===")
exception_columns = [
    c for c in [
        "target", "held_out_source", "variant", "feature",
        "training_status", "eligible_for_distance",
        "unsupported_test_records", "constant_deviation_records"
    ]
    if c in feature_output.columns
]
if exception_columns:
    exceptions = feature_output[
        (
            feature_output["training_status"].isin(
                ["ALL_MISSING_IN_TRAIN", "CONSTANT_IN_TRAIN"]
            )
        )
        | (
            feature_output.get(
                "unsupported_test_records",
                pd.Series(0, index=feature_output.index)
            ).fillna(0) > 0
        )
        | (
            feature_output.get(
                "constant_deviation_records",
                pd.Series(0, index=feature_output.index)
            ).fillna(0) > 0
        )
    ]
    print(exceptions[exception_columns].to_string(index=False))
print("\n=== OUTPUT FILES ===")
for path in sorted(OUT.glob("*.csv")):
    print(f"{path} | {path.stat().st_size:,} bytes")
print("\n=== REASON AUDIT COMPLETE ===")
