from pathlib import Path
import json
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
DATASET = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "training_results" / "regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "yield_strength"
NUMERIC_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T"
]
CATEGORICAL_FEATURES = ["quench_medium", "steel_family"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
def main():
    df = pd.read_csv(DATASET)
    required = FEATURES + [
        TARGET, "grade_id", "heat_id", "tensile_strength"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=[TARGET, "grade_id"]).copy()
    df["YS_UTS_ratio"] = (
        df[TARGET] / df["tensile_strength"].replace(0, np.nan)
    )
    anomaly_mask = (
        df["grade_id"].astype(str).isin(
            ["EMK0319", "EMK0320", "EMK0321"]
        )
        & (df["YS_UTS_ratio"] < 0.03)
    )
    excluded = df.loc[anomaly_mask].copy()
    clean = df.loc[~anomaly_mask].copy()
    X = clean[FEATURES]
    y = clean[TARGET].astype(float)
    groups = clean["grade_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    baseline = float(y.iloc[train_idx].median())
    baseline_predictions = np.full(len(test_idx), baseline)
    metrics = {
        "target": TARGET,
        "algorithm": "ExtraTreesRegressor",
        "training_records": len(train_idx),
        "test_records": len(test_idx),
        "training_grades": int(groups.iloc[train_idx].nunique()),
        "test_grades": int(groups.iloc[test_idx].nunique()),
        "excluded_anomaly_records": len(excluded),
        "MAE": float(mean_absolute_error(y.iloc[test_idx], predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y.iloc[test_idx], predictions))),
        "R2": float(r2_score(y.iloc[test_idx], predictions)),
        "median_baseline_MAE": float(
            mean_absolute_error(y.iloc[test_idx], baseline_predictions)
        ),
        "validation": "grade-held-out",
        "random_state": 42,
        "features": FEATURES,
        "anomaly_exclusion": (
            "EMK0319/0320/0321 records with YS/UTS ratio below 0.03"
        )
    }
    joblib.dump(
        model,
        MODEL_DIR / "steelbench_yield_extratrees_candidate.joblib"
    )
    pd.DataFrame({
        "actual": y.iloc[test_idx].to_numpy(),
        "predicted": predictions,
        "grade_id": groups.iloc[test_idx].to_numpy()
    }).to_csv(
        RESULT_DIR / "steelbench_yield_candidate_test_predictions.csv",
        index=False
    )
    excluded.to_csv(
        RESULT_DIR / "steelbench_yield_candidate_excluded_anomalies.csv",
        index=False
    )
    with open(
        RESULT_DIR / "steelbench_yield_candidate_training_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)
    print("\nYIELD MODEL TRAINING COMPLETED")
    print(json.dumps(metrics, indent=4))
if __name__ == "__main__":
    main()

