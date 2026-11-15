from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_results" / "applicability_domain_v3_retry"
records = pd.read_csv(OUT / "record_applicability.csv")
folds = pd.read_csv(OUT / "fold_applicability_summary.csv")
calibration = pd.read_csv(OUT / "calibration_reference_summary.csv")
print("\n=== STATUS DISTRIBUTION BY TARGET, SOURCE AND VARIANT ===")
status = (
    records.groupby(
        ["target", "held_out_source", "variant", "applicability_status"]
    )
    .size()
    .unstack(fill_value=0)
)
print(status.to_string())
print("\n=== STATUS PERCENTAGES BY SOURCE AND VARIANT ===")
status_pct = status.div(status.sum(axis=1), axis=0).mul(100).round(1)
print(status_pct.to_string())
print("\n=== FOLD-LEVEL DISTANCE AND EVIDENCE SUMMARY ===")
columns = [
    "target", "held_out_source", "variant",
    "eligible_feature_count",
    "train_grades", "calibration_grades", "test_grades",
    "calibration_q90", "calibration_q95",
    "test_distance_median", "test_distance_q95",
    "mean_eligible_evidence_fraction",
    "records_with_constant_deviations",
    "records_with_unsupported_feature_values",
    "records_with_unseen_categories",
    "records_with_numeric_range_violations",
]
print(folds[columns].to_string(index=False))
print("\n=== CALIBRATION THRESHOLD RATIO ===")
calibration["q95_to_q90_ratio"] = (
    calibration["calibration_q95"] /
    calibration["calibration_q90"].replace(0, float("nan"))
)
print(
    calibration[
        [
            "target", "held_out_source", "variant",
            "calibration_grade_count",
            "calibration_q90", "calibration_q95",
            "q95_to_q90_ratio",
        ]
    ].to_string(index=False)
)
print("\n=== SOURCE-LEVEL SUMMARY COMPLETE ===")
