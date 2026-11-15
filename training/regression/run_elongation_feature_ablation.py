import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATA_PATH = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "regression"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
# Common chemistry/process features available across all three sources.
BASE_NUMERIC = ["C", "Mn", "V"]
BASE_CATEGORICAL = ["steel_family"]
MODEL_FEATURES = {
    "A_baseline": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
    },
    "B_plus_tensile": {
        "numeric": BASE_NUMERIC + ["tensile_strength"],
        "categorical": BASE_CATEGORICAL,
    },
    "C_plus_yield": {
        "numeric": BASE_NUMERIC + [
            "tensile_strength",
            "yield_strength",
        ],
        "categorical": BASE_CATEGORICAL,
    },
    "D_plus_ratio": {
        "numeric": BASE_NUMERIC + [
            "tensile_strength",
            "yield_strength",
            "yield_tensile_ratio",
        ],
        "categorical": BASE_CATEGORICAL,
    },
}
print("=" * 95)
print("MAT-X CONTROLLED FEATURE-ABLATION EXPERIMENT")
print("=" * 95)
# ------------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
required = [
    "source",
    "grade_id",
    "heat_id",
    "steel_family",
    "elongation",
    "C",
    "Mn",
    "V",
    "tensile_strength",
    "yield_strength",
]
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")
for col in [
    "elongation",
    "C",
    "Mn",
    "V",
    "tensile_strength",
    "yield_strength",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Use only records with a measured elongation and complete
# common evidence required by all four models.
df = df.loc[df["elongation"].notna()].copy()
required_features = [
    "C",
    "Mn",
    "V",
    "steel_family",
    "tensile_strength",
    "yield_strength",
]
complete_mask = df[required_features].notna().all(axis=1)
df = df.loc[complete_mask].copy()
# Ratio is derived from measured yield and tensile strength.
# Invalid/non-positive tensile values are not allowed.
if (df["tensile_strength"] <= 0).any():
    raise ValueError("Non-positive tensile strength found.")
df["yield_tensile_ratio"] = (
    df["yield_strength"] / df["tensile_strength"]
)
if not np.isfinite(
    df["yield_tensile_ratio"].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid yield/tensile ratio found.")
if df["source"].isna().any():
    raise ValueError("Missing source groups found.")
if df["steel_family"].isna().any():
    raise ValueError("Missing steel-family values found.")
df = df.reset_index(drop=True)
print("\nDATASET")
print("Eligible records:", len(df))
print("Sources:", sorted(df["source"].unique().tolist()))
print("Unique grades:", df["grade_id"].nunique())
print("Unique heat IDs:", df["heat_id"].nunique())
print("Missing common features:", int(
    df[required_features].isna().sum().sum()
))
print("\nSOURCE COUNTS")
print(df["source"].value_counts().to_string())
# ------------------------------------------------------------
# 2. MODEL FACTORY
# ------------------------------------------------------------
def build_model(numeric_features, categorical_features):
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    estimator = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("regressor", estimator),
        ]
    )
# ------------------------------------------------------------
# 3. SOURCE-HELD-OUT VALIDATION
# ------------------------------------------------------------
logo = LeaveOneGroupOut()
groups = df["source"].to_numpy()
y = df["elongation"].to_numpy(dtype=float)
prediction_rows = []
fold_rows = []
for fold_number, (train_idx, test_idx) in enumerate(
    logo.split(df, y, groups=groups),
    start=1,
):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    held_out_source = test_df["source"].iloc[0]
    if train_df["source"].isin(test_df["source"]).any():
        raise AssertionError("Source leakage detected.")
    train_grades = set(train_df["grade_id"])
    test_grades = set(test_df["grade_id"])
    grade_overlap = train_grades.intersection(test_grades)
    if grade_overlap:
        raise AssertionError(
            f"Grade leakage in fold {fold_number}: {grade_overlap}"
        )
    print("\n" + "-" * 95)
    print(f"FOLD {fold_number}: HELD-OUT SOURCE = {held_out_source}")
    print("Training records:", len(train_df))
    print("Test records:", len(test_df))
    print("Training sources:", sorted(train_df["source"].unique()))
    print("Grade overlap:", len(grade_overlap))
    fold_predictions = test_df[
        ["source", "grade_id", "heat_id", "steel_family", "elongation"]
    ].copy()
    fold_predictions = fold_predictions.rename(
        columns={"elongation": "actual"}
    )
    fold_predictions["fold"] = fold_number
    fold_predictions["held_out_source"] = held_out_source
    for model_name, feature_config in MODEL_FEATURES.items():
        numeric_features = feature_config["numeric"]
        categorical_features = feature_config["categorical"]
        selected_features = numeric_features + categorical_features
        model = build_model(
            numeric_features,
            categorical_features,
        )
        model.fit(
            train_df[selected_features],
            train_df["elongation"],
        )
        predictions = model.predict(
            test_df[selected_features]
        )
        fold_predictions[model_name + "_prediction"] = predictions
        actual = test_df["elongation"].to_numpy(dtype=float)
        mae = mean_absolute_error(actual, predictions)
        rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
        r2 = r2_score(actual, predictions)
        fold_rows.append({
            "fold": fold_number,
            "held_out_source": held_out_source,
            "model": model_name,
            "train_records": len(train_df),
            "test_records": len(test_df),
            "train_sources": sorted(
                train_df["source"].unique().tolist()
            ),
            "grade_overlap": len(grade_overlap),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        })
        print(
            f"{model_name:<20} "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f}"
        )
    prediction_rows.append(fold_predictions)
# ------------------------------------------------------------
# 4. POOLED METRICS AND PAIRED COMPARISON
# ------------------------------------------------------------
predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True,
)
fold_metrics = pd.DataFrame(fold_rows)
model_names = list(MODEL_FEATURES.keys())
actual = predictions_df["actual"].to_numpy(dtype=float)
pooled_rows = []
for model_name in model_names:
    pred = predictions_df[
        model_name + "_prediction"
    ].to_numpy(dtype=float)
    pooled_rows.append({
        "model": model_name,
        "records": len(pred),
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
    })
pooled = pd.DataFrame(pooled_rows)
baseline_mae = float(
    pooled.loc[pooled["model"] == "A_baseline", "MAE"].iloc[0]
)
baseline_rmse = float(
    pooled.loc[pooled["model"] == "A_baseline", "RMSE"].iloc[0]
)
pooled["MAE_improvement_vs_A_percent"] = (
    (baseline_mae - pooled["MAE"]) / baseline_mae * 100
)
pooled["RMSE_improvement_vs_A_percent"] = (
    (baseline_rmse - pooled["RMSE"]) / baseline_rmse * 100
)
print("\n" + "=" * 95)
print("POOLED SOURCE-HELD-OUT RESULTS")
print("=" * 95)
print(pooled.to_string(index=False))
print("\n" + "=" * 95)
print("HELD-OUT SOURCE RESULTS")
print("=" * 95)
print(
    fold_metrics[
        [
            "held_out_source",
            "model",
            "test_records",
            "MAE",
            "RMSE",
            "R2",
        ]
    ].to_string(index=False)
)
# ------------------------------------------------------------
# 5. PAIRED ERROR ANALYSIS
# ------------------------------------------------------------
for model_name in model_names:
    predictions_df[model_name + "_absolute_error"] = (
        predictions_df["actual"]
        - predictions_df[model_name + "_prediction"]
    ).abs()
baseline_error = predictions_df[
    "A_baseline_absolute_error"
]
paired_rows = []
for model_name in model_names[1:]:
    model_error = predictions_df[
        model_name + "_absolute_error"
    ]
    improvement = baseline_error - model_error
    paired_rows.append({
        "comparison": f"{model_name} vs A_baseline",
        "records": len(improvement),
        "mean_paired_improvement": float(improvement.mean()),
        "model_wins": int((improvement > 0).sum()),
        "baseline_wins": int((improvement < 0).sum()),
        "ties": int((improvement == 0).sum()),
        "MAE_improvement_percent": float(
            (baseline_error.mean() - model_error.mean())
            / baseline_error.mean() * 100
        ),
    })
paired = pd.DataFrame(paired_rows)
print("\n" + "=" * 95)
print("PAIRED ERROR COMPARISON AGAINST MODEL A")
print("=" * 95)
print(paired.to_string(index=False))
# ------------------------------------------------------------
# 6. SAVE REPORTS
# ------------------------------------------------------------
pred_path = OUT / "matx_elongation_feature_ablation_predictions.csv"
fold_path = OUT / "matx_elongation_feature_ablation_fold_metrics.csv"
pooled_path = OUT / "matx_elongation_feature_ablation_pooled_metrics.csv"
paired_path = OUT / "matx_elongation_feature_ablation_paired_comparison.csv"
report_path = OUT / "matx_elongation_feature_ablation_report.json"
predictions_df.to_csv(pred_path, index=False)
fold_metrics.to_csv(fold_path, index=False)
pooled.to_csv(pooled_path, index=False)
paired.to_csv(paired_path, index=False)
report = {
    "experiment": "MAT-X controlled elongation feature ablation",
    "experiment_version": "1.0",
    "target": "elongation",
    "validation": "LeaveOneGroupOut grouped by source",
    "source_groups": sorted(df["source"].unique().tolist()),
    "records": int(len(df)),
    "unique_grades": int(df["grade_id"].nunique()),
    "random_state": RANDOM_STATE,
    "estimator": {
        "name": "ExtraTreesRegressor",
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    },
    "models": {
        name: {
            "numeric_features": config["numeric"],
            "categorical_features": config["categorical"],
        }
        for name, config in MODEL_FEATURES.items()
    },
    "ratio_definition": "yield_strength / tensile_strength",
    "ratio_uses_measured_values": True,
    "source_and_grade_ids_used_as_features": False,
    "grade_overlap_across_each_fold": 0,
    "pooled_metrics": pooled.to_dict(orient="records"),
    "fold_metrics": fold_metrics.to_dict(orient="records"),
    "paired_comparisons": paired.to_dict(orient="records"),
    "promotion_status": "EXPERIMENTAL_VALIDATION_ONLY",
    "original_dataset_modified": False,
    "existing_models_modified": False,
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n" + "=" * 95)
print("OUTPUT FILES")
print("=" * 95)
for path in [
    pred_path,
    fold_path,
    pooled_path,
    paired_path,
    report_path,
]:
    print(path)
print("\nEXPERIMENT COMPLETE.")
print("Original dataset and existing models were not modified.")
