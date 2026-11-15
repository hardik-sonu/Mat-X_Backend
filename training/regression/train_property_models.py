from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "training_results" / "regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUMERIC_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si",
    "V", "Cu", "Al", "austenitize_T", "temper_T",
]
CATEGORICAL_FEATURES = [
    "quench_medium",
    "steel_family",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = {
    "elongation": {
        "unit_status": "DATASET_UNIT_NOT_INDEPENDENTLY_VERIFIED",
        "scope": "Mixed EMK_spec and Kaggle sources",
    },
    "reduction_area": {
        "unit_status": "DATASET_UNIT_NOT_INDEPENDENTLY_VERIFIED",
        "scope": "EMK_spec only",
    },
    "impact_J_avg": {
        "unit_status": "IMPACT_TEST_UNIT_AND_CONDITIONS_REQUIRE_VERIFICATION",
        "scope": "EMK_spec only",
    },
}
df = pd.read_csv(DATA)
required = FEATURES + [
    "heat_id", "grade_id", "source",
    "elongation", "reduction_area", "impact_J_avg",
]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
print("\nMAT-X PROPERTY MODEL TRAINING")
print("=" * 75)
print("Dataset records:", len(df))
print("Validation: grade-held-out GroupShuffleSplit")
print("Random state: 42")
print("Model: ExtraTreesRegressor")
print("Features:", FEATURES)
for column in NUMERIC_FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")
for target, metadata in TARGETS.items():
    print("\n" + "=" * 75)
    print("TRAINING PROPERTY:", target)
    print("=" * 75)
    data = df.dropna(
        subset=[target, "grade_id", "heat_id"]
    ).copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    X = data[FEATURES].copy()
    y = data[target].astype(float)
    groups = data["grade_id"].astype(str)
    if groups.nunique() < 10:
        raise ValueError(
            f"Insufficient independent grades for {target}"
        )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_grades = set(groups.iloc[train_idx])
    test_grades = set(groups.iloc[test_idx])
    overlap = train_grades.intersection(test_grades)
    if overlap:
        raise RuntimeError(
            f"Grade leakage detected for {target}: {len(overlap)}"
        )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    baseline_prediction = np.full(
        len(y_test),
        float(y_train.median()),
    )
    baseline_mae = float(
        mean_absolute_error(y_test, baseline_prediction)
    )
    model_name = target.replace("_", "_")
    artifact_path = MODEL_DIR / (
        f"matx_{model_name}_extratrees_candidate.joblib"
    )
    report_path = OUTPUT_DIR / (
        f"matx_{model_name}_candidate_report.json"
    )
    predictions_path = OUTPUT_DIR / (
        f"matx_{model_name}_test_predictions.csv"
    )
    joblib.dump(model, artifact_path)
    result_df = pd.DataFrame({
        "heat_id": data.iloc[test_idx]["heat_id"].values,
        "grade_id": data.iloc[test_idx]["grade_id"].values,
        "source": data.iloc[test_idx]["source"].values,
        "actual": y_test.values,
        "predicted": predictions,
        "absolute_error": np.abs(y_test.values - predictions),
    })
    result_df.to_csv(predictions_path, index=False)
    source_counts = {
        str(k): int(v)
        for k, v in data["source"].value_counts().items()
    }
    report = {
        "engine": "MAT-X Property Reconstruction Candidate Model",
        "property": target,
        "model": "ExtraTreesRegressor",
        "model_parameters": {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "dataset": str(DATA),
        "total_dataset_records": int(len(data)),
        "training_records": int(len(train_idx)),
        "testing_records": int(len(test_idx)),
        "independent_grades": int(groups.nunique()),
        "training_grades": int(len(train_grades)),
        "testing_grades": int(len(test_grades)),
        "grade_overlap": int(len(overlap)),
        "source_distribution": source_counts,
        "features": FEATURES,
        "excluded_features": [
            "heat_id",
            "grade_id",
            "source",
            "tensile_strength",
            "yield_strength",
            "elongation",
            "reduction_area",
            "impact_J_avg",
            "hardness",
        ],
        "validation": "GRADE_HELD_OUT",
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "median_baseline_mae": baseline_mae,
        },
        "target_distribution": {
            "minimum": float(y.min()),
            "median": float(y.median()),
            "maximum": float(y.max()),
        },
        "unit_status": metadata["unit_status"],
        "dataset_scope": metadata["scope"],
        "impact_upper_boundary_review": (
            {
                "records_at_or_above_145": int((y >= 145).sum()),
                "records_equal_to_150": int((y == 150).sum()),
                "interpretation": (
                    "Upper-bound pattern requires source verification; "
                    "no values were removed or modified."
                ),
            }
            if target == "impact_J_avg"
            else None
        ),
        "uncertainty_status": (
            "PREDICTION_INTERVALS_NOT_YET_CALIBRATED"
        ),
        "external_validation": "NOT_PERFORMED",
        "candidate_artifact": str(artifact_path),
        "test_predictions": str(predictions_path),
        "promotion_status": "CANDIDATE_ONLY_NOT_PROMOTED",
    }
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )
    print("Records:", len(data))
    print("Training:", len(train_idx))
    print("Testing:", len(test_idx))
    print("Independent grades:", groups.nunique())
    print("Grade overlap:", len(overlap))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("Median baseline MAE:", round(baseline_mae, 4))
    print("Artifact:", artifact_path.name)
    print("Report:", report_path.name)
    print("Status: CANDIDATE ONLY")
print("\nALL THREE PROPERTY MODEL TRAINING RUNS COMPLETED.")
