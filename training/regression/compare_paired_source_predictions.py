from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT = Path.cwd()
OLD = ROOT / "training_results" / "source_heldout_validation_v1"
NEW = ROOT / "training_results" / "common_feature_source_heldout_v1"
OUT = ROOT / "training_results" / "paired_feature_comparison_v1"
if OUT.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing output: {OUT}"
    )
OUT.mkdir(parents=True)
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
KEYS = ["source", "grade_id", "heat_id"]
comparison_rows = []
detail_frames = []
audit_rows = []
for target in TARGETS:
    for source in SOURCES:
        old_path = (
            OLD
            / f"{target}_baseline_heldout_{source}_predictions.csv"
        )
        new_path = (
            NEW
            / "predictions"
            / f"{target}__heldout__{source}.csv"
        )
        if not old_path.exists():
            audit_rows.append({
                "target": target,
                "source": source,
                "status": "OLD_PREDICTIONS_UNAVAILABLE",
            })
            continue
        if not new_path.exists():
            audit_rows.append({
                "target": target,
                "source": source,
                "status": "NEW_PREDICTIONS_UNAVAILABLE",
            })
            continue
        old = pd.read_csv(old_path)
        new = pd.read_csv(new_path)
        required = set(KEYS + ["actual", "prediction"])
        if not required.issubset(old.columns):
            raise ValueError(
                f"Missing columns in {old_path.name}: "
                f"{required - set(old.columns)}"
            )
        if not required.issubset(new.columns):
            raise ValueError(
                f"Missing columns in {new_path.name}: "
                f"{required - set(new.columns)}"
            )
        old = old[KEYS + ["actual", "prediction"]].copy()
        new = new[KEYS + ["actual", "prediction"]].copy()
        for frame in [old, new]:
            for col in KEYS:
                frame[col] = frame[col].astype(str).str.strip()
            frame["actual"] = pd.to_numeric(
                frame["actual"], errors="coerce"
            )
            frame["prediction"] = pd.to_numeric(
                frame["prediction"], errors="coerce"
            )
        old_duplicates = int(old.duplicated(KEYS).sum())
        new_duplicates = int(new.duplicated(KEYS).sum())
        if old_duplicates or new_duplicates:
            raise RuntimeError(
                f"Duplicate prediction identifiers: {target}/{source}; "
                f"old={old_duplicates}, new={new_duplicates}"
            )
        old = old.rename(columns={
            "actual": "actual_old",
            "prediction": "prediction_13_feature",
        })
        new = new.rename(columns={
            "actual": "actual_new",
            "prediction": "prediction_2_feature",
        })
        merged = old.merge(
            new,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        old_only = len(old) - len(merged)
        new_only = len(new) - len(merged)
        actual_difference = (
            merged["actual_old"] - merged["actual_new"]
        ).abs()
        actual_match = bool(
            len(merged) > 0
            and actual_difference.max() <= 1e-9
        )
        if not actual_match:
            raise RuntimeError(
                f"Actual target mismatch for {target}/{source}. "
                f"Maximum difference: "
                f"{actual_difference.max() if len(merged) else 'N/A'}"
            )
        if len(merged) == 0:
            audit_rows.append({
                "target": target,
                "source": source,
                "status": "NO_MATCHED_RECORDS",
                "old_records": len(old),
                "new_records": len(new),
            })
            continue
        actual = merged["actual_old"].to_numpy(dtype=float)
        pred_old = merged[
            "prediction_13_feature"
        ].to_numpy(dtype=float)
        pred_new = merged[
            "prediction_2_feature"
        ].to_numpy(dtype=float)
        if not (
            np.isfinite(actual).all()
            and np.isfinite(pred_old).all()
            and np.isfinite(pred_new).all()
        ):
            raise RuntimeError(
                f"Non-finite values in paired predictions: {target}/{source}"
            )
        error_old = np.abs(actual - pred_old)
        error_new = np.abs(actual - pred_new)
        mae_old = float(mean_absolute_error(actual, pred_old))
        mae_new = float(mean_absolute_error(actual, pred_new))
        rmse_old = float(
            np.sqrt(mean_squared_error(actual, pred_old))
        )
        rmse_new = float(
            np.sqrt(mean_squared_error(actual, pred_new))
        )
        r2_old = (
            float(r2_score(actual, pred_old))
            if len(actual) >= 2 and np.var(actual) > 0
            else None
        )
        r2_new = (
            float(r2_score(actual, pred_new))
            if len(actual) >= 2 and np.var(actual) > 0
            else None
        )
        old_better = int((error_old < error_new).sum())
        new_better = int((error_new < error_old).sum())
        ties = int((error_old == error_new).sum())
        row = {
            "target": target,
            "held_out_source": source,
            "matched_records": len(merged),
            "matched_grades": merged["grade_id"].nunique(),
            "old_records_unmatched": old_only,
            "new_records_unmatched": new_only,
            "actual_values_match": actual_match,
            "mae_13_feature": mae_old,
            "mae_2_feature": mae_new,
            "mae_difference_2_minus_13": mae_new - mae_old,
            "rmse_13_feature": rmse_old,
            "rmse_2_feature": rmse_new,
            "r2_13_feature": r2_old,
            "r2_2_feature": r2_new,
            "records_13_feature_lower_error": old_better,
            "records_2_feature_lower_error": new_better,
            "ties": ties,
            "13_feature_win_rate": old_better / len(merged),
            "2_feature_win_rate": new_better / len(merged),
        }
        comparison_rows.append(row)
        merged["target"] = target
        merged["held_out_source"] = source
        merged["absolute_error_13_feature"] = error_old
        merged["absolute_error_2_feature"] = error_new
        merged["absolute_error_difference_2_minus_13"] = (
            error_new - error_old
        )
        detail_frames.append(merged)
        audit_rows.append({
            "target": target,
            "source": source,
            "status": "PAIRED_SUCCESSFULLY",
            "old_records": len(old),
            "new_records": len(new),
            "matched_records": len(merged),
            "old_records_unmatched": old_only,
            "new_records_unmatched": new_only,
            "actual_values_match": actual_match,
        })
        print(
            f"{target:18s} | {source:27s} | "
            f"n={len(merged):4d} | "
            f"MAE 13f={mae_old:9.3f} | "
            f"MAE 2f={mae_new:9.3f} | "
            f"2f-13f={mae_new-mae_old:9.3f}"
        )
comparison = pd.DataFrame(comparison_rows)
audit = pd.DataFrame(audit_rows)
comparison.to_csv(
    OUT / "paired_comparison_metrics.csv",
    index=False,
)
audit.to_csv(
    OUT / "paired_record_audit.csv",
    index=False,
)
if detail_frames:
    details = pd.concat(detail_frames, ignore_index=True)
    details.to_csv(
        OUT / "paired_record_errors.csv",
        index=False,
    )
overall_rows = []
if not comparison.empty:
    for target, group in comparison.groupby("target"):
        detail = pd.concat(
            [
                frame.loc[frame["target"] == target]
                for frame in detail_frames
            ],
            ignore_index=True,
        )
        actual = detail["actual_old"].to_numpy(dtype=float)
        pred_old = detail[
            "prediction_13_feature"
        ].to_numpy(dtype=float)
        pred_new = detail[
            "prediction_2_feature"
        ].to_numpy(dtype=float)
        error_old = np.abs(actual - pred_old)
        error_new = np.abs(actual - pred_new)
        overall_rows.append({
            "target": target,
            "matched_records": len(detail),
            "matched_grades": detail["grade_id"].nunique(),
            "mae_13_feature": mean_absolute_error(actual, pred_old),
            "mae_2_feature": mean_absolute_error(actual, pred_new),
            "mae_difference_2_minus_13": (
                mean_absolute_error(actual, pred_new)
                - mean_absolute_error(actual, pred_old)
            ),
            "rmse_13_feature": np.sqrt(
                mean_squared_error(actual, pred_old)
            ),
            "rmse_2_feature": np.sqrt(
                mean_squared_error(actual, pred_new)
            ),
            "r2_13_feature": (
                r2_score(actual, pred_old)
                if len(actual) >= 2 and np.var(actual) > 0
                else None
            ),
            "r2_2_feature": (
                r2_score(actual, pred_new)
                if len(actual) >= 2 and np.var(actual) > 0
                else None
            ),
            "13_feature_lower_error_rate": float(
                (error_old < error_new).mean()
            ),
            "2_feature_lower_error_rate": float(
                (error_new < error_old).mean()
            ),
        })
overall = pd.DataFrame(overall_rows)
overall.to_csv(
    OUT / "paired_comparison_overall.csv",
    index=False,
)
summary = {
    "experiment": "Paired 13-feature versus 2-feature comparison",
    "matched_folds": len(comparison),
    "matched_records": (
        int(comparison["matched_records"].sum())
        if not comparison.empty else 0
    ),
    "all_actual_values_verified": (
        bool(comparison["actual_values_match"].all())
        if not comparison.empty else False
    ),
    "methodological_caveat": (
        "The 13-feature experiment uses a separate calibration partition, "
        "while the 2-feature experiment trains on all eligible non-held-out "
        "records. This is not a controlled feature-ablation comparison."
    ),
    "outputs": str(OUT),
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\n" + "=" * 100)
print("PAIRED COMPARISON BY SOURCE")
print("=" * 100)
if not comparison.empty:
    print(
        comparison[
            [
                "target",
                "held_out_source",
                "matched_records",
                "mae_13_feature",
                "mae_2_feature",
                "mae_difference_2_minus_13",
                "r2_13_feature",
                "r2_2_feature",
            ]
        ].to_string(index=False)
    )
print("\n" + "=" * 100)
print("OVERALL PAIRED COMPARISON")
print("=" * 100)
if not overall.empty:
    print(overall.to_string(index=False))
print("\n" + "=" * 100)
print("PAIRING AUDIT")
print("=" * 100)
print(audit.to_string(index=False))
print("\nSummary:")
print(json.dumps(summary, indent=2))
print("\nAll original datasets, models and experiment outputs preserved.")
