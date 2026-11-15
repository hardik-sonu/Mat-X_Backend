import json
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
sys.path.insert(0, str(ROOT / "training" / "regression"))
import audit_promoted_applicability_distance_calibration as ad
DATASET = ad.DATASET
SOURCE_REPORT = (
    ROOT / "training_results" / "applicability_promoted_reference_audit"
    / "promoted_applicability_distance_calibration_audit.json"
)
OUT = (
    ROOT / "training_results" / "applicability_promoted_reference_audit"
    / "held_out_test_validation_v1"
)
EXPECTED = {
    "tensile_strength": {"train": 1046, "test": 314, "train_grades": 449, "test_grades": 113},
    "yield_strength": {"train": 783, "test": 204, "train_grades": 446, "test_grades": 112},
}
def safe_float(value):
    return float(value) if np.isfinite(value) else None
def rate(mask):
    return float(np.mean(mask)) if len(mask) else None
def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files; refusing to overwrite: {OUT}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATASET.exists() or not SOURCE_REPORT.exists():
        raise FileNotFoundError("Dataset or distance-calibration report is missing.")
    with SOURCE_REPORT.open("r", encoding="utf-8") as handle:
        calibration_report = json.load(handle)
    df = pd.read_csv(DATASET)
    results = {}
    for target, config in ad.TARGETS.items():
        print(f"\n========== {target.upper()} ==========")
        clean = df.dropna(subset=[target, "grade_id"]).copy()
        if target == "yield_strength":
            ratio = (
                clean[target]
                / clean["tensile_strength"].replace(0, np.nan)
            )
            anomaly = (
                clean["grade_id"].astype(str).isin(
                    ["EMK0319", "EMK0320", "EMK0321"]
                )
                & (ratio < 0.03)
            )
            excluded = int(anomaly.sum())
            clean = clean.loc[~anomaly].copy()
        else:
            excluded = 0
        X = clean[ad.FEATURES].copy()
        y = clean[target].astype(float).copy()
        groups = clean["grade_id"].astype(str)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=42
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train = X.iloc[train_idx].copy().reset_index(drop=True)
        X_test = X.iloc[test_idx].copy().reset_index(drop=True)
        y_train = y.iloc[train_idx].to_numpy(dtype=float)
        y_test = y.iloc[test_idx].to_numpy(dtype=float)
        train_groups = groups.iloc[train_idx].astype(str).reset_index(drop=True)
        test_groups = groups.iloc[test_idx].astype(str).reset_index(drop=True)
        train_count = len(X_train)
        test_count = len(X_test)
        train_grade_count = int(train_groups.nunique())
        test_grade_count = int(test_groups.nunique())
        expected = EXPECTED[target]
        if train_count != expected["train"] or test_count != expected["test"]:
            raise RuntimeError(
                f"{target}: split record counts differ from expected: "
                f"train={train_count}, test={test_count}"
            )
        if (
            train_grade_count != expected["train_grades"]
            or test_grade_count != expected["test_grades"]
        ):
            raise RuntimeError(f"{target}: split grade counts differ from expected.")
        overlap = set(train_groups) & set(test_groups)
        if overlap:
            raise RuntimeError(
                f"{target}: train/test grade leakage detected: {sorted(overlap)[:10]}"
            )
        calibration = calibration_report[target]["methods"]["fixed_full_reference"]["summary"]
        q90 = calibration["q90"]
        q95 = calibration["q95"]
        if q90 is None or q95 is None:
            raise RuntimeError(f"{target}: missing calibrated distance thresholds.")
        reference = ad.fit_reference(X_train)
        distances = ad.distance_matrix(X_test, X_train, reference)
        nearest_distance = np.min(distances, axis=1)
        model_path = config["model"]
        model_hash_before = ad.sha256(model_path)
        model = joblib.load(model_path)
        predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        model_hash_after = ad.sha256(model_path)
        if model_hash_before != model_hash_after:
            raise RuntimeError(f"{target}: promoted model changed during audit.")
        absolute_error = np.abs(y_test - predictions)
        missing_count = X_test[ad.FEATURES].isna().sum(axis=1).to_numpy()
        q01_q99_count = np.zeros(test_count, dtype=int)
        min_max_count = np.zeros(test_count, dtype=int)
        for feature in reference["numeric"]:
            values = pd.to_numeric(
                X_test[feature], errors="coerce"
            ).to_numpy(dtype=float)
            item = reference["numeric_reference"][feature]
            valid = np.isfinite(values)
            q01_q99_count += (
                valid
                & ((values < item["q01"]) | (values > item["q99"]))
            ).astype(int)
            min_max_count += (
                valid
                & ((values < item["minimum"]) | (values > item["maximum"]))
            ).astype(int)
        unseen_category_count = np.zeros(test_count, dtype=int)
        for feature in reference["categorical"]:
            values = (
                X_test[feature].astype("string")
                .fillna(ad.MISSING_CATEGORY).astype(str).to_numpy()
            )
            known = reference["categorical_reference"][feature]
            unseen = np.array([
                value != ad.MISSING_CATEGORY and value not in known
                for value in values
            ])
            unseen_category_count += unseen.astype(int)
        q90_exceeded = nearest_distance > q90
        q95_exceeded = nearest_distance > q95
        distance_status = np.where(
            q95_exceeded,
            "ABOVE_Q95_SCREENING_THRESHOLD",
            np.where(
                q90_exceeded,
                "ABOVE_Q90_SCREENING_THRESHOLD",
                "AT_OR_BELOW_Q90_SCREENING_THRESHOLD",
            ),
        )
        record_frame = X_test.copy()
        record_frame.insert(0, "source_row_index", clean.index.to_numpy()[test_idx])
        record_frame.insert(1, "grade_id", test_groups.to_numpy())
        record_frame["actual_value"] = y_test
        record_frame["predicted_value"] = predictions
        record_frame["absolute_error"] = absolute_error
        record_frame["nearest_training_distance"] = nearest_distance
        record_frame["distance_status"] = distance_status
        record_frame["distance_q90_exceeded"] = q90_exceeded
        record_frame["distance_q95_exceeded"] = q95_exceeded
        record_frame["missing_feature_count"] = missing_count
        record_frame["numeric_outside_q01_q99_count"] = q01_q99_count
        record_frame["numeric_outside_min_max_count"] = min_max_count
        record_frame["unseen_category_count"] = unseen_category_count
        csv_path = OUT / f"{target}_held_out_record_audit.csv"
        record_frame.to_csv(csv_path, index=False)
        valid_corr = (
            np.isfinite(nearest_distance)
            & np.isfinite(absolute_error)
        )
        if valid_corr.sum() >= 3:
            correlation = spearmanr(
                nearest_distance[valid_corr],
                absolute_error[valid_corr],
            )
            spearman_rho = safe_float(correlation.statistic)
            spearman_p = safe_float(correlation.pvalue)
        else:
            spearman_rho = None
            spearman_p = None
        grade_frame = pd.DataFrame({
            "grade_id": test_groups,
            "nearest_distance": nearest_distance,
            "absolute_error": absolute_error,
        })
        grade_summary = (
            grade_frame.groupby("grade_id")
            .agg(
                records=("nearest_distance", "size"),
                maximum_distance=("nearest_distance", "max"),
                mean_distance=("nearest_distance", "mean"),
                mean_absolute_error=("absolute_error", "mean"),
                maximum_absolute_error=("absolute_error", "max"),
            )
            .reset_index()
        )
        grade_summary["maximum_distance_above_q90"] = (
            grade_summary["maximum_distance"] > q90
        )
        grade_summary["maximum_distance_above_q95"] = (
            grade_summary["maximum_distance"] > q95
        )
        grade_csv = OUT / f"{target}_held_out_grade_audit.csv"
        grade_summary.to_csv(grade_csv, index=False)
        summary = {
            "target": target,
            "split": {
                "method": "GroupShuffleSplit",
                "test_size": 0.20,
                "random_state": 42,
                "grouping": "grade_id",
                "training_records": train_count,
                "test_records": test_count,
                "training_grades": train_grade_count,
                "test_grades": test_grade_count,
                "train_test_grade_overlap_count": len(overlap),
                "yield_anomaly_records_excluded": excluded,
            },
            "distance_reference": {
                "method": "fixed_full_training_reference",
                "calibration_unit": "grade_id",
                "q90": q90,
                "q95": q95,
                "eligible_numeric_features": reference["numeric"],
                "eligible_categorical_features": reference["categorical"],
                "eligible_dimensions": (
                    len(reference["numeric"]) + len(reference["categorical"])
                ),
            },
            "held_out_distance_screening": {
                "records_above_q90": int(q90_exceeded.sum()),
                "records_above_q95": int(q95_exceeded.sum()),
                "fraction_above_q90": rate(q90_exceeded),
                "fraction_above_q95": rate(q95_exceeded),
                "distance_median": safe_float(np.median(nearest_distance)),
                "distance_mean": safe_float(np.mean(nearest_distance)),
                "distance_maximum": safe_float(np.max(nearest_distance)),
                "grades_with_maximum_distance_above_q90": int(
                    grade_summary["maximum_distance_above_q90"].sum()
                ),
                "grades_with_maximum_distance_above_q95": int(
                    grade_summary["maximum_distance_above_q95"].sum()
                ),
            },
            "input_evidence_diagnostics": {
                "records_with_missing_features": int((missing_count > 0).sum()),
                "fraction_with_missing_features": rate(missing_count > 0),
                "mean_missing_feature_count": safe_float(np.mean(missing_count)),
                "records_with_numeric_values_outside_q01_q99": int(
                    (q01_q99_count > 0).sum()
                ),
                "records_with_numeric_values_outside_training_min_max": int(
                    (min_max_count > 0).sum()
                ),
                "records_with_unseen_categories": int(
                    (unseen_category_count > 0).sum()
                ),
            },
            "prediction_error_analysis": {
                "mae": safe_float(np.mean(absolute_error)),
                "rmse": safe_float(np.sqrt(np.mean((y_test - predictions) ** 2))),
                "median_absolute_error": safe_float(np.median(absolute_error)),
                "spearman_distance_vs_absolute_error": {
                    "rho": spearman_rho,
                    "p_value": spearman_p,
                    "sample_count": int(valid_corr.sum()),
                },
                "interpretation": (
                    "Exploratory association only. Correlation does not establish "
                    "that distance predicts error or that thresholds calibrate error."
                ),
            },
            "integrity": {
                "promoted_model_sha256_before": model_hash_before,
                "promoted_model_sha256_after": model_hash_after,
                "promoted_model_unchanged": model_hash_before == model_hash_after,
                "source_dataset_modified": False,
                "promoted_model_modified": False,
            },
            "outputs": {
                "record_csv": str(csv_path),
                "grade_csv": str(grade_csv),
            },
            "interpretation_limit": (
                "Held-out evaluation of experimental applicability-distance screening. "
                "Not a probability of correctness, prediction interval, calibrated "
                "prediction uncertainty, or external-source validation."
            ),
        }
        results[target] = summary
        print(json.dumps(summary, indent=2))
    report_path = OUT / "held_out_applicability_test_validation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n========== HELD-OUT VALIDATION COMPLETE ==========")
    print("Report:", report_path)
    print("Output folder:", OUT)
    print("Promoted models unchanged:", all(
        item["integrity"]["promoted_model_unchanged"]
        for item in results.values()
    ))
    print("Original dataset and promoted models were not modified.")
if __name__ == "__main__":
    main()
