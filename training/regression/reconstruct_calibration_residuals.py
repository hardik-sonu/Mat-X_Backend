from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
EXPERIMENT = ROOT / "training_results" / "uncertainty_calibration_v1"
MODEL_DIR = EXPERIMENT / "models"
OUT = EXPERIMENT / "calibration_residual_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V",
    "Cu", "Al", "austenitize_T", "temper_T"
]
CATEGORICAL = ["quench_medium", "steel_family"]
CONFIGS = {
    "tensile_strength": {
        "trees": 400,
        "model": "tensile_strength_baseline_grade_grouped.joblib",
    },
    "yield_strength": {
        "trees": 500,
        "model": "yield_strength_baseline_grade_grouped.joblib",
    },
}
def reconstruct_split(X, y, groups):
    first = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_cal_idx, test_idx = next(
        first.split(X, y, groups=groups)
    )
    X_train_cal = X.iloc[train_cal_idx]
    y_train_cal = y.iloc[train_cal_idx]
    groups_train_cal = groups.iloc[train_cal_idx]
    second = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43
    )
    train_rel, calibration_rel = next(
        second.split(
            X_train_cal,
            y_train_cal,
            groups=groups_train_cal,
        )
    )
    train_idx = np.asarray(train_cal_idx)[train_rel]
    calibration_idx = np.asarray(train_cal_idx)[calibration_rel]
    return train_idx, calibration_idx, np.asarray(test_idx)
def prepare_target(df, target):
    work = df.copy()
    if target in ["tensile_strength", "yield_strength"]:
        work = work.dropna(subset=[target, "grade_id"]).copy()
        if target == "yield_strength":
            work[target] = pd.to_numeric(
                work[target], errors="coerce"
            )
            work["tensile_strength"] = pd.to_numeric(
                work["tensile_strength"], errors="coerce"
            )
            ratio = (
                work[target]
                / work["tensile_strength"].replace(0, np.nan)
            )
            anomaly_mask = (
                work["grade_id"].astype(str).isin(
                    ["EMK0319", "EMK0320", "EMK0321"]
                )
                & (ratio < 0.03)
            )
            print(
                "Yield anomaly rows excluded:",
                int(anomaly_mask.sum())
            )
            work = work.loc[~anomaly_mask].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target, "grade_id"]).copy()
    work["grade_id"] = work["grade_id"].astype(str)
    numeric_features = [c for c in NUMERIC if c in work.columns]
    categorical_features = [
        c for c in CATEGORICAL if c in work.columns
    ]
    for col in numeric_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    feature_columns = numeric_features + categorical_features
    X = work[feature_columns].copy()
    y = work[target].astype(float)
    groups = work["grade_id"]
    return work, X, y, groups, feature_columns
df = pd.read_csv(DATASET, low_memory=False)
for target, config in CONFIGS.items():
    print("\n" + "=" * 88)
    print("RECONSTRUCTING:", target)
    print("=" * 88)
    work, X, y, groups, feature_columns = prepare_target(df, target)
    train_idx, calibration_idx, test_idx = reconstruct_split(
        X, y, groups
    )
    X_cal = X.iloc[calibration_idx]
    y_cal = y.iloc[calibration_idx]
    g_cal = groups.iloc[calibration_idx]
    model_path = MODEL_DIR / config["model"]
    pipeline = joblib.load(model_path)
    predictions = np.asarray(
        pipeline.predict(X_cal), dtype=float
    )
    actual = y_cal.to_numpy(dtype=float)
    residuals = np.abs(actual - predictions)
    records = pd.DataFrame({
        "dataset_row_index": work.index.to_numpy()[calibration_idx],
        "grade_id": g_cal.to_numpy(),
        "actual": actual,
        "prediction": predictions,
        "absolute_residual": residuals,
    })
    for optional_col in ["source", "heat_id", "steel_family"]:
        if optional_col in work.columns:
            records[optional_col] = (
                work.iloc[calibration_idx][optional_col].to_numpy()
            )
    grade_stats = (
        records.groupby("grade_id")
        .agg(
            calibration_record_count=("absolute_residual", "size"),
            maximum_absolute_residual=("absolute_residual", "max"),
            median_absolute_residual=("absolute_residual", "median"),
            mean_absolute_residual=("absolute_residual", "mean"),
            minimum_absolute_residual=("absolute_residual", "min"),
            residual_std=("absolute_residual", "std"),
        )
        .reset_index()
    )
    records = records.merge(
        grade_stats[["grade_id", "calibration_record_count",
                     "maximum_absolute_residual"]],
        on="grade_id",
        how="left",
        validate="many_to_one",
    )
    records["residual_fraction_of_grade_max"] = (
        records["absolute_residual"]
        / records["maximum_absolute_residual"].replace(0, np.nan)
    )
    records["is_grade_max_residual"] = np.isclose(
        records["absolute_residual"],
        records["maximum_absolute_residual"],
        rtol=1e-10,
        atol=1e-10,
    )
    expected_path = EXPERIMENT / (
        f"{target}_baseline_calibration_grade_scores.csv"
    )
    expected = pd.read_csv(expected_path)
    expected["grade_id"] = expected["grade_id"].astype(str)
    comparison = grade_stats.merge(
        expected,
        on="grade_id",
        how="outer",
        suffixes=("_reconstructed", "_original"),
        indicator=True,
    )
    complete_match = (
        comparison["_merge"].eq("both").all()
        and len(comparison) == len(expected)
        and np.allclose(
            comparison["maximum_absolute_residual_reconstructed"],
            comparison["maximum_absolute_residual_original"],
            rtol=1e-8,
            atol=1e-8,
            equal_nan=False,
        )
    )
    records_path = OUT / f"{target}_individual_calibration_residuals.csv"
    grades_path = OUT / f"{target}_calibration_grade_diagnostics.csv"
    comparison_path = OUT / f"{target}_original_score_reconciliation.csv"
    records.to_csv(records_path, index=False)
    grade_stats.to_csv(grades_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    print("Training records:", len(train_idx))
    print("Calibration records:", len(calibration_idx))
    print("Calibration grades:", len(grade_stats))
    print("Test records:", len(test_idx))
    print("Features:", feature_columns)
    print("Original grade-score rows:", len(expected))
    print("Reconstructed grade-score rows:", len(grade_stats))
    print("EXACT SCORE RECONCILIATION:", complete_match)
    if not complete_match:
        print("\nRECONCILIATION DIFFERENCES:")
        print(
            comparison.loc[
                comparison["_merge"] != "both"
            ].head(20).to_string(index=False)
        )
        matched = comparison.loc[comparison["_merge"] == "both"]
        if len(matched):
            difference = (
                matched["maximum_absolute_residual_reconstructed"]
                - matched["maximum_absolute_residual_original"]
            ).abs()
            print("Maximum absolute score difference:", difference.max())
    print("\nTOP 15 GRADE-LEVEL RESIDUALS:")
    print(
        grade_stats.sort_values(
            "maximum_absolute_residual", ascending=False
        ).head(15).to_string(index=False)
    )
    print("\nCALIBRATION RECORD COUNT DISTRIBUTION:")
    print(
        grade_stats["calibration_record_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("\nTOP 20 INDIVIDUAL RESIDUALS:")
    print(
        records.sort_values(
            "absolute_residual", ascending=False
        )[
            [
                "grade_id", "actual", "prediction",
                "absolute_residual", "calibration_record_count",
                "maximum_absolute_residual",
                "is_grade_max_residual",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nSaved:")
    print(records_path)
    print(grades_path)
    print(comparison_path)
print("\nRECONSTRUCTION AND RECONCILIATION FINISHED.")
print("Diagnostic output directory:", OUT)
