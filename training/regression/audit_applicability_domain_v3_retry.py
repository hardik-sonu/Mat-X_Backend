from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_results" / "applicability_domain_v3_retry"
def read_csv_safe(name):
    path = OUT / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
records = read_csv_safe("record_applicability.csv")
folds = read_csv_safe("fold_applicability_summary.csv")
features = read_csv_safe("feature_domain_summary.csv")
splits = read_csv_safe("split_integrity_audit.csv")
failures = read_csv_safe("execution_failures.csv")
calibration = read_csv_safe("calibration_reference_summary.csv")
summary = json.loads(
    (OUT / "summary.json").read_text(encoding="utf-8")
)
checks = {}
checks["record_count_matches_summary"] = (
    len(records) == summary["record_rows"]
)
checks["assessment_count_matches_summary"] = (
    len(folds) == summary["successful_assessments"]
)
checks["no_execution_failures"] = failures.empty
checks["split_integrity_passed"] = bool(
    summary["split_integrity_passed"]
)
duplicate_columns = [
    "target", "held_out_source", "variant", "heat_id"
]
if all(c in records.columns for c in duplicate_columns):
    checks["no_duplicate_test_heat_rows"] = (
        not records.duplicated(duplicate_columns).any()
    )
observed_statuses = {
    str(k): int(v)
    for k, v in records["applicability_status"].value_counts().items()
}
checks["status_totals_match"] = (
    observed_statuses == summary["status_totals"]
)
distance_columns = [
    c for c in records.columns
    if "distance" in c.lower()
]
for col in distance_columns:
    if pd.api.types.is_numeric_dtype(records[col]):
        checks[f"{col}_finite"] = bool(
            np.isfinite(records[col].dropna()).all()
        )
overlap_columns = [
    c for c in splits.columns
    if "overlap" in c.lower()
]
if overlap_columns:
    checks["all_split_overlaps_zero"] = bool(
        (
            splits[overlap_columns]
            .fillna(0)
            .to_numpy() == 0
        ).all()
    )
if not calibration.empty and len(calibration) == len(folds):
    checks["calibration_rows_match_folds"] = True
else:
    checks["calibration_rows_match_folds"] = (
        len(calibration) == len(folds)
    )
print("\n=== INTEGRITY CHECKS ===")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'} | {name}")
print("\n=== OUTPUT COUNTS ===")
print(f"Records: {len(records):,}")
print(f"Fold/variant assessments: {len(folds)}")
print(f"Feature diagnostic rows: {len(features)}")
print(f"Split audit rows: {len(splits)}")
print(f"Execution failures: {len(failures)}")
print("\n=== STATUS COUNTS BY TARGET AND VARIANT ===")
if not records.empty:
    print(
        records.groupby(
            ["target", "variant", "applicability_status"]
        ).size().unstack(fill_value=0).to_string()
    )
print("\n=== CALIBRATION FILE COLUMNS ===")
print(list(calibration.columns))
print("\n=== CALIBRATION SUPPORT ===")
if not calibration.empty:
    print(calibration.to_string(index=False))
print("\n=== FEATURE TRAINING STATUS COUNTS ===")
if not features.empty:
    print(
        features.groupby(
            ["variant", "training_status"]
        ).size().unstack(fill_value=0).to_string()
    )
print("\n=== FEATURE DIAGNOSTIC COLUMNS ===")
print(list(features.columns))
print("\n=== FOLD SUMMARY COLUMNS ===")
print(list(folds.columns))
print("\n=== OVERALL AUDIT ===")
print(
    "ALL CHECKS PASSED"
    if all(checks.values())
    else "AUDIT REQUIRES INVESTIGATION"
)

