from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "training_results" / "regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
BASE_CATEGORICAL = [
    "quench_medium",
    "steel_family",
]
MECHANICAL_PROPERTIES = [
    "tensile_strength",
    "yield_strength",
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
TARGETS = [
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
print("=" * 100)
print("MAT-X EVIDENCE-ENRICHED PROPERTY RECONSTRUCTION")
print("=" * 100)
df = pd.read_csv(DATA, dtype={"hardness": "string"})
required = (
    ["heat_id", "grade_id", "source"]
    + BASE_NUMERIC
    + BASE_CATEGORICAL
    + MECHANICAL_PROPERTIES
)
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
for col in BASE_NUMERIC + MECHANICAL_PROPERTIES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Dataset records:", len(df))
print("Dataset grades:", df["grade_id"].nunique())
print("Hardness excluded: unresolved scale/representation")
print("Source and grade identifiers excluded as predictors.")
def make_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ])
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                add_indicator=True,
            ),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    estimator = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])
def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
for target in TARGETS:
    print("\n" + "=" * 100)
    print("TARGET:", target)
    print("=" * 100)
    data = df.loc[df[target].notna()].copy().reset_index(drop=True)
    if data["grade_id"].isna().any():
        raise ValueError(f"{target}: Missing grade identifiers.")
    groups = data["grade_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(data, data[target], groups)
    )
    train_grades = set(groups.iloc[train_idx])
    test_grades = set(groups.iloc[test_idx])
    overlap = train_grades.intersection(test_grades)
    if overlap:
        raise AssertionError(
            f"Grade leakage detected for {target}: {len(overlap)}"
        )
    y_train = data.iloc[train_idx][target]
    y_test = data.iloc[test_idx][target]
    # Baseline: chemistry and process features only.
    baseline_numeric = BASE_NUMERIC.copy()
    baseline_categorical = BASE_CATEGORICAL.copy()
    # Enriched: add all other mechanical properties.
    evidence_features = [
        p for p in MECHANICAL_PROPERTIES if p != target
    ]
    enriched_numeric = (
        BASE_NUMERIC.copy() + evidence_features
    )
    baseline_model = make_pipeline(
        baseline_numeric,
        baseline_categorical,
    )
    enriched_model = make_pipeline(
        enriched_numeric,
        baseline_categorical,
    )
    X_train_baseline = data.iloc[train_idx][
        baseline_numeric + baseline_categorical
    ]
    X_test_baseline = data.iloc[test_idx][
        baseline_numeric + baseline_categorical
    ]
    X_train_enriched = data.iloc[train_idx][
        enriched_numeric + baseline_categorical
    ]
    X_test_enriched = data.iloc[test_idx][
        enriched_numeric + baseline_categorical
    ]
    print("Target records:", len(data))
    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("Training grades:", len(train_grades))
    print("Testing grades:", len(test_grades))
    print("Grade overlap:", len(overlap))
    print("\nAdditional evidence availability:")
    for feature in evidence_features:
        available = int(data[feature].notna().sum())
        print(
            f"{feature:<22} "
            f"{available}/{len(data)} "
            f"({available / len(data):.1%})"
        )
    complete_evidence = int(
        data[evidence_features].notna().all(axis=1).sum()
    )
    print(
        "Records with all additional evidence:",
        complete_evidence,
    )
    # Fit baseline and enriched models on the same training records.
    baseline_model.fit(X_train_baseline, y_train)
    enriched_model.fit(X_train_enriched, y_train)
    baseline_prediction = baseline_model.predict(X_test_baseline)
    enriched_prediction = enriched_model.predict(X_test_enriched)
    baseline_metrics = calculate_metrics(
        y_test,
        baseline_prediction,
    )
    enriched_metrics = calculate_metrics(
        y_test,
        enriched_prediction,
    )
    improvement = {
        metric: (
            baseline_metrics[metric] - enriched_metrics[metric]
        )
        for metric in ["mae", "rmse"]
    }
    improvement["mae_percent"] = (
        improvement["mae"] / baseline_metrics["mae"] * 100
        if baseline_metrics["mae"] else None
    )
    improvement["rmse_percent"] = (
        improvement["rmse"] / baseline_metrics["rmse"] * 100
        if baseline_metrics["rmse"] else None
    )
    improvement["r2_absolute"] = (
        enriched_metrics["r2"] - baseline_metrics["r2"]
    )
    enriched_name = f"matx_{target}_evidence_enriched_extratrees"
    baseline_name = f"matx_{target}_matched_baseline_extratrees"
    baseline_artifact = MODEL_DIR / f"{baseline_name}_candidate.joblib"
    enriched_artifact = MODEL_DIR / f"{enriched_name}_candidate.joblib"
    joblib.dump(baseline_model, baseline_artifact)
    joblib.dump(enriched_model, enriched_artifact)
    predictions_path = OUTPUT_DIR / (
        f"matx_{target}_evidence_enriched_test_predictions.csv"
    )
    predictions_df = pd.DataFrame({
        "heat_id": data.iloc[test_idx]["heat_id"].values,
        "grade_id": data.iloc[test_idx]["grade_id"].values,
        "source": data.iloc[test_idx]["source"].values,
        "actual": y_test.values,
        "baseline_prediction": baseline_prediction,
        "enriched_prediction": enriched_prediction,
        "baseline_absolute_error": np.abs(
            y_test.values - baseline_prediction
        ),
        "enriched_absolute_error": np.abs(
            y_test.values - enriched_prediction
        ),
    })
    predictions_df.to_csv(predictions_path, index=False)
    report = {
        "engine": "MAT-X Evidence-Enriched Property Reconstruction",
        "version": "1.0.0",
        "property": target,
        "dataset": str(DATA),
        "total_target_records": int(len(data)),
        "training_records": int(len(train_idx)),
        "testing_records": int(len(test_idx)),
        "independent_grades": int(groups.nunique()),
        "training_grades": int(len(train_grades)),
        "testing_grades": int(len(test_grades)),
        "grade_overlap": int(len(overlap)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation": "GRADE_HELD_OUT",
        "baseline": {
            "model": "ExtraTreesRegressor",
            "features": baseline_numeric + baseline_categorical,
            "metrics": baseline_metrics,
            "artifact": str(baseline_artifact),
        },
        "evidence_enriched": {
            "model": "ExtraTreesRegressor",
            "features": enriched_numeric + baseline_categorical,
            "additional_evidence": evidence_features,
            "metrics": enriched_metrics,
            "artifact": str(enriched_artifact),
        },
        "comparison": improvement,
        "missing_data_handling": {
            "numeric": "TRAINING_MEDIAN_IMPUTATION_WITH_INDICATORS",
            "categorical": "TRAINING_MODE_IMPUTATION_WITH_INDICATORS",
            "missing_values_modified_in_original_dataset": False,
        },
        "excluded_predictors": [
            "target_property",
            "heat_id",
            "grade_id",
            "source",
            "hardness",
            "condition",
            "split",
            "data_tier",
        ],
        "leakage_controls": [
            "Target property excluded from all predictors.",
            "Same target records used for baseline and enriched models.",
            "Same grade-held-out split used for both models.",
            "Grade identifiers excluded from model features.",
            "Source identifiers excluded from model features.",
            "Imputers fitted on training data only.",
        ],
        "deployment_scope": (
            "Conditional reconstruction: other measured mechanical "
            "properties must be available at prediction time."
        ),
        "uncertainty_status": "PREDICTION_INTERVALS_NOT_YET_CALIBRATED",
        "external_validation": "NOT_PERFORMED",
        "promotion_status": "CANDIDATE_ONLY_NOT_PROMOTED",
        "test_predictions": str(predictions_path),
    }
    report_path = OUTPUT_DIR / (
        f"matx_{target}_evidence_enriched_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )
    print("\nBASELINE METRICS")
    for key, value in baseline_metrics.items():
        print(f"{key.upper():<12}: {value:.5f}")
    print("\nEVIDENCE-ENRICHED METRICS")
    for key, value in enriched_metrics.items():
        print(f"{key.upper():<12}: {value:.5f}")
    print("\nIMPROVEMENT (POSITIVE IS BETTER FOR MAE/RMSE)")
    print(f"MAE improvement : {improvement['mae']:.5f}")
    print(f"MAE improvement %: {improvement['mae_percent']:.2f}%")
    print(f"RMSE improvement: {improvement['rmse']:.5f}")
    print(f"RMSE improvement %: {improvement['rmse_percent']:.2f}%")
    print(f"R2 improvement  : {improvement['r2_absolute']:.5f}")
    print("\nSAVED")
    print("Baseline artifact:", baseline_artifact.name)
    print("Enriched artifact:", enriched_artifact.name)
    print("Predictions:", predictions_path.name)
    print("Report:", report_path.name)
    print("Status: CANDIDATE ONLY")
print("\n" + "=" * 100)
print("ALL EVIDENCE-ENRICHED TRAINING RUNS COMPLETED.")
print("No existing model artifacts were overwritten.")
print("Original dataset was not modified.")
print("=" * 100)
