from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = ROOT / "training_results" / "regression"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "elongation"
NUMERIC_BASE = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL = ["quench_medium", "steel_family"]
ADDITIONAL_EVIDENCE = [
    "tensile_strength", "yield_strength", "reduction_area", "impact_J_avg",
]
EXPECTED_SOURCES = [
    "EMK_spec",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
print("=" * 78)
print("MAT-X ELONGATION: LEAVE-ONE-SOURCE-OUT VALIDATION")
print("=" * 78)
df = pd.read_csv(DATA)
required = (
    ["source", "grade_id", "heat_id", TARGET]
    + NUMERIC_BASE + CATEGORICAL + ADDITIONAL_EVIDENCE
)
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Required columns missing from dataset: {missing_columns}")
# Work on an in-memory copy. The original CSV is never modified.
df = df.copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.loc[df[TARGET].notna()].copy()
df = df.loc[df["source"].isin(EXPECTED_SOURCES)].copy()
for col in NUMERIC_BASE + ADDITIONAL_EVIDENCE:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["source"] = df["source"].astype(str)
df = df.reset_index(drop=True)
observed_sources = sorted(df["source"].unique().tolist())
if set(observed_sources) != set(EXPECTED_SOURCES):
    raise ValueError(
        f"Expected exactly {EXPECTED_SOURCES}; found {observed_sources}"
    )
print(f"\nEligible elongation records: {len(df)}")
print(f"Sources: {observed_sources}")
print("\nRecords by source:")
print(df.groupby("source").size().to_string())
# No source, grade, heat, or target-derived identifier is used as a feature.
baseline_features = NUMERIC_BASE + CATEGORICAL
enriched_features = NUMERIC_BASE + ADDITIONAL_EVIDENCE + CATEGORICAL
def build_model(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
logo = LeaveOneGroupOut()
X_groups = df["source"]
y = df[TARGET].astype(float)
prediction_rows = []
fold_reports = []
for fold_number, (train_idx, test_idx) in enumerate(
    logo.split(df, y, groups=X_groups), start=1
):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    held_out_source = test_df["source"].iloc[0]
    train_sources = sorted(train_df["source"].unique().tolist())
    if held_out_source in train_sources:
        raise AssertionError("Source leakage detected.")
    grade_overlap = set(train_df["grade_id"].dropna()) & set(
        test_df["grade_id"].dropna()
    )
    if grade_overlap:
        raise AssertionError(
            f"Grade overlap detected in {held_out_source}: {len(grade_overlap)}"
        )
    baseline = build_model(NUMERIC_BASE, CATEGORICAL)
    enriched = build_model(
        NUMERIC_BASE + ADDITIONAL_EVIDENCE,
        CATEGORICAL,
    )
    baseline.fit(train_df[baseline_features], train_df[TARGET])
    enriched.fit(train_df[enriched_features], train_df[TARGET])
    y_test = test_df[TARGET].to_numpy(dtype=float)
    baseline_pred = baseline.predict(test_df[baseline_features])
    enriched_pred = enriched.predict(test_df[enriched_features])
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    enriched_mae = mean_absolute_error(y_test, enriched_pred)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    enriched_rmse = float(np.sqrt(mean_squared_error(y_test, enriched_pred)))
    baseline_r2 = r2_score(y_test, baseline_pred)
    enriched_r2 = r2_score(y_test, enriched_pred)
    fold_report = {
        "fold": fold_number,
        "held_out_source": held_out_source,
        "training_sources": train_sources,
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "train_unique_grades": int(train_df["grade_id"].nunique()),
        "test_unique_grades": int(test_df["grade_id"].nunique()),
        "grade_overlap_count": int(len(grade_overlap)),
        "baseline": {
            "mae": float(baseline_mae),
            "rmse": baseline_rmse,
            "r2": float(baseline_r2),
        },
        "evidence_enriched": {
            "mae": float(enriched_mae),
            "rmse": enriched_rmse,
            "r2": float(enriched_r2),
        },
        "mae_improvement_percent": float(
            (baseline_mae - enriched_mae) / baseline_mae * 100
        ) if baseline_mae else None,
        "rmse_improvement_percent": float(
            (baseline_rmse - enriched_rmse) / baseline_rmse * 100
        ) if baseline_rmse else None,
    }
    fold_reports.append(fold_report)
    for i, (_, row) in enumerate(test_df.iterrows()):
        prediction_rows.append({
            "source": held_out_source,
            "grade_id": row["grade_id"],
            "heat_id": row["heat_id"],
            "actual_elongation": float(y_test[i]),
            "baseline_prediction": float(baseline_pred[i]),
            "enriched_prediction": float(enriched_pred[i]),
            "baseline_absolute_error": float(abs(y_test[i] - baseline_pred[i])),
            "enriched_absolute_error": float(abs(y_test[i] - enriched_pred[i])),
            "paired_absolute_error_improvement": float(
                abs(y_test[i] - baseline_pred[i])
                - abs(y_test[i] - enriched_pred[i])
            ),
        })
    print("\n" + "-" * 78)
    print(f"FOLD {fold_number}: HELD-OUT SOURCE = {held_out_source}")
    print(f"Training sources: {train_sources}")
    print(f"Train records: {len(train_df)} | Test records: {len(test_df)}")
    print(f"Grade overlap: {len(grade_overlap)}")
    print("\n                    BASELINE       ENRICHED")
    print(f"MAE                 {baseline_mae:10.4f}     {enriched_mae:10.4f}")
    print(f"RMSE                {baseline_rmse:10.4f}     {enriched_rmse:10.4f}")
    print(f"R2                  {baseline_r2:10.4f}     {enriched_r2:10.4f}")
    print(f"MAE improvement: {fold_report['mae_improvement_percent']:.2f}%")
predictions = pd.DataFrame(prediction_rows)
if len(predictions) != len(df):
    raise AssertionError(
        f"Out-of-fold prediction count {len(predictions)} != eligible rows {len(df)}"
    )
# Pooled out-of-fold metrics: every record is predicted by a model
# that was trained without that record's entire source.
actual = predictions["actual_elongation"].to_numpy()
base_pred = predictions["baseline_prediction"].to_numpy()
enriched_pred = predictions["enriched_prediction"].to_numpy()
pooled = {
    "records": int(len(predictions)),
    "baseline": {
        "mae": float(mean_absolute_error(actual, base_pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, base_pred))),
        "r2": float(r2_score(actual, base_pred)),
    },
    "evidence_enriched": {
        "mae": float(mean_absolute_error(actual, enriched_pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, enriched_pred))),
        "r2": float(r2_score(actual, enriched_pred)),
    },
    "mae_improvement_percent": float(
        (mean_absolute_error(actual, base_pred)
         - mean_absolute_error(actual, enriched_pred))
        / mean_absolute_error(actual, base_pred) * 100
    ),
    "rmse_improvement_percent": float(
        (np.sqrt(mean_squared_error(actual, base_pred))
         - np.sqrt(mean_squared_error(actual, enriched_pred)))
        / np.sqrt(mean_squared_error(actual, base_pred)) * 100
    ),
    "enriched_wins": int(
        (predictions["paired_absolute_error_improvement"] > 0).sum()
    ),
    "baseline_wins": int(
        (predictions["paired_absolute_error_improvement"] < 0).sum()
    ),
    "ties": int(
        (predictions["paired_absolute_error_improvement"] == 0).sum()
    ),
    "mean_paired_absolute_error_improvement": float(
        predictions["paired_absolute_error_improvement"].mean()
    ),
}
report = {
    "experiment": "MAT-X elongation leave-one-source-out validation",
    "validation_method": "LeaveOneGroupOut grouped by source",
    "target": TARGET,
    "sources": EXPECTED_SOURCES,
    "excluded_source": "NIMS: no eligible elongation target records",
    "baseline_features": baseline_features,
    "evidence_enriched_additional_features": ADDITIONAL_EVIDENCE,
    "model": "ExtraTreesRegressor",
    "parameters": {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "source_and_grade_identifiers_used_as_features": False,
    "original_dataset_modified": False,
    "model_artifacts_promoted": False,
    "folds": fold_reports,
    "pooled_out_of_source_metrics": pooled,
}
pred_path = OUT / "matx_elongation_source_held_out_predictions.csv"
report_path = OUT / "matx_elongation_source_held_out_report.json"
predictions.to_csv(pred_path, index=False)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n" + "=" * 78)
print("POOLED OUT-OF-SOURCE RESULTS")
print("=" * 78)
print(f"Records: {pooled['records']}")
print("\n                    BASELINE       ENRICHED")
print(f"MAE                 {pooled['baseline']['mae']:10.4f}     {pooled['evidence_enriched']['mae']:10.4f}")
print(f"RMSE                {pooled['baseline']['rmse']:10.4f}     {pooled['evidence_enriched']['rmse']:10.4f}")
print(f"R2                  {pooled['baseline']['r2']:10.4f}     {pooled['evidence_enriched']['r2']:10.4f}")
print(f"\nPooled MAE improvement: {pooled['mae_improvement_percent']:.2f}%")
print(f"Pooled RMSE improvement: {pooled['rmse_improvement_percent']:.2f}%")
print(
    f"Enriched wins: {pooled['enriched_wins']} | "
    f"Baseline wins: {pooled['baseline_wins']} | "
    f"Ties: {pooled['ties']}"
)
print(
    "Mean paired absolute-error improvement: "
    f"{pooled['mean_paired_absolute_error_improvement']:.4f}"
)
print("\nOUTPUT FILES")
print(pred_path)
print(report_path)
print("\nVALIDATION COMPLETE. NO MODEL WAS PROMOTED.")
