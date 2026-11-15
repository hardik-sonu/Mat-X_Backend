from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path("training_results")
PRED_ROOT = ROOT / "controlled_feature_ablation_v1" / "predictions"
APP_PATH = ROOT / "applicability_domain_v3_retry" / "record_applicability.csv"
OUT = ROOT / "applicability_prediction_error_validation_v1"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = [
    "target", "held_out_source", "variant",
    "source", "grade_id", "heat_id",
]
app = pd.read_csv(APP_PATH)
required_app = set(KEYS + ["applicability_status"])
missing_app = required_app - set(app.columns)
if missing_app:
    raise RuntimeError(f"Applicability file missing columns: {missing_app}")
app["variant"] = app["variant"].astype(str)
if app[KEYS].isna().any().any():
    raise RuntimeError("Null identity keys found in applicability records.")
if app.duplicated(KEYS).any():
    raise RuntimeError("Duplicate six-field keys in applicability records.")
prediction_frames = []
for path in sorted(PRED_ROOT.rglob("*.csv")):
    parts = path.relative_to(PRED_ROOT).parts
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected prediction path: {path}")
    target, held_out_source, filename = parts
    variant = Path(filename).stem
    df = pd.read_csv(path)
    required = {
        "source", "grade_id", "heat_id",
        "actual", "prediction", "absolute_error",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path} missing columns: {sorted(missing)}"
        )
    df["target"] = target
    df["held_out_source"] = held_out_source
    df["variant"] = variant
    prediction_frames.append(
        df[KEYS + ["actual", "prediction", "absolute_error"]]
    )
if not prediction_frames:
    raise RuntimeError("No prediction files found.")
pred = pd.concat(prediction_frames, ignore_index=True)
if pred[KEYS].isna().any().any():
    raise RuntimeError("Null identity keys found in predictions.")
if pred.duplicated(KEYS).any():
    duplicates = pred.loc[pred.duplicated(KEYS, keep=False)]
    duplicates.to_csv(OUT / "duplicate_prediction_keys.csv", index=False)
    raise RuntimeError("Duplicate six-field prediction keys found.")
print("Applicability records:", len(app))
print("Prediction records:", len(pred))
merged = app.merge(
    pred,
    on=KEYS,
    how="outer",
    validate="one_to_one",
    indicator=True,
)
merge_counts = merged["_merge"].value_counts().to_dict()
print("\n=== EXACT JOIN COUNTS ===")
print(merge_counts)
left_only = merged.loc[merged["_merge"] == "left_only"]
right_only = merged.loc[merged["_merge"] == "right_only"]
left_only.to_csv(OUT / "applicability_without_prediction.csv", index=False)
right_only.to_csv(OUT / "prediction_without_applicability.csv", index=False)
matched = merged.loc[merged["_merge"] == "both"].copy()
if len(matched) != len(app) or len(matched) != len(pred):
    raise RuntimeError(
        "Record-level matching incomplete. "
        "Do not proceed to error analysis."
    )
actual = pd.to_numeric(matched["actual"], errors="coerce")
prediction = pd.to_numeric(matched["prediction"], errors="coerce")
reported_error = pd.to_numeric(
    matched["absolute_error"], errors="coerce"
)
calculated_error = (actual - prediction).abs()
finite = (
    np.isfinite(actual.to_numpy())
    & np.isfinite(prediction.to_numpy())
    & np.isfinite(reported_error.to_numpy())
)
if not finite.all():
    raise RuntimeError("Non-finite actual/prediction/error values found.")
error_difference = (
    reported_error.to_numpy() - calculated_error.to_numpy()
)
error_consistent = np.isclose(
    reported_error.to_numpy(),
    calculated_error.to_numpy(),
    rtol=1e-7,
    atol=1e-8,
)
matched["calculated_absolute_error"] = calculated_error
matched["error_difference"] = error_difference
matched["error_consistent"] = error_consistent
print("\n=== ERROR FIELD INTEGRITY ===")
print("Finite numeric rows:", int(finite.sum()))
print("Consistent error rows:", int(error_consistent.sum()))
print("Inconsistent error rows:", int((~error_consistent).sum()))
print(
    "Maximum absolute error-field difference:",
    float(np.max(np.abs(error_difference)))
)
if not error_consistent.all():
    matched.loc[~error_consistent].to_csv(
        OUT / "inconsistent_error_rows.csv",
        index=False,
    )
    raise RuntimeError(
        "Stored absolute_error does not match actual-prediction."
    )
# Preserve the complete joined evidence for the next analysis.
matched.to_csv(OUT / "matched_applicability_predictions.csv", index=False)
# Summary of errors by applicability status.
status_summary = (
    matched.groupby(
        ["target", "held_out_source", "variant", "applicability_status"],
        dropna=False,
    )
    .agg(
        records=("actual", "size"),
        mae=("calculated_absolute_error", "mean"),
        median_absolute_error=("calculated_absolute_error", "median"),
        q90_absolute_error=(
            "calculated_absolute_error",
            lambda x: x.quantile(0.90),
        ),
        q95_absolute_error=(
            "calculated_absolute_error",
            lambda x: x.quantile(0.95),
        ),
    )
    .reset_index()
)
status_summary.to_csv(
    OUT / "error_by_applicability_status.csv",
    index=False,
)
# Overall status-level view, with target and variant kept separate above.
print("\n=== ERROR BY APPLICABILITY STATUS ===")
print(status_summary.to_string(index=False))
result = {
    "applicability_records": int(len(app)),
    "prediction_records": int(len(pred)),
    "matched_records": int(len(matched)),
    "applicability_without_prediction": int(len(left_only)),
    "prediction_without_applicability": int(len(right_only)),
    "duplicate_applicability_keys": int(app.duplicated(KEYS).sum()),
    "duplicate_prediction_keys": int(pred.duplicated(KEYS).sum()),
    "finite_numeric_rows": int(finite.sum()),
    "error_consistent_rows": int(error_consistent.sum()),
    "error_inconsistent_rows": int((~error_consistent).sum()),
    "maximum_error_field_difference": float(
        np.max(np.abs(error_difference))
    ),
    "exact_one_to_one_match_passed": bool(
        len(matched) == len(app) == len(pred)
        and len(left_only) == 0
        and len(right_only) == 0
    ),
    "error_integrity_passed": bool(error_consistent.all()),
    "outputs": [
        "matched_applicability_predictions.csv",
        "error_by_applicability_status.csv",
    ],
}
(OUT / "validation_summary.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)
print("\n=== VALIDATION SUMMARY ===")
print(json.dumps(result, indent=2))
print("\nOutput directory:", OUT.resolve())
print("\n=== RECORD-LEVEL VALIDATION COMPLETE ===")

