from pathlib import Path
import json
import math
import traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT = Path.cwd()
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "common_feature_source_heldout_v1"
MODEL_OUT = OUT / "models"
PREDICTION_OUT = OUT / "predictions"
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
FEATURES = ["C", "Mn"]
if OUT.exists():
    raise FileExistsError(
        f"Output directory already exists. "
        f"Refusing to overwrite: {OUT}"
    )
if not DATASET.exists():
    raise FileNotFoundError(DATASET)
MODEL_OUT.mkdir(parents=True)
PREDICTION_OUT.mkdir(parents=True)
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for col in FEATURES + TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
results = []
failures = []
skipped = []
print("=" * 88)
print("MAT-X COMMON-FEATURE SOURCE-HELD-OUT BENCHMARK")
print("=" * 88)
print("Features:", FEATURES)
print("Model: ExtraTreesRegressor")
print("Trees: 500")
print("Minimum samples per leaf: 2")
print("Random state: 42")
print("No imputation or source-specific features.")
print("No conformal calibration in this point-prediction experiment.")
for target in TARGETS:
    clean = df.dropna(
        subset=[
            target,
            "source",
            "grade_id",
            "heat_id",
            *FEATURES,
        ]
    ).copy()
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
        print(
            f"\nYield anomaly records excluded: {excluded}"
        )
    print("\n" + "-" * 88)
    print(f"TARGET: {target}")
    print("-" * 88)
    for held_out in SOURCES:
        test = clean.loc[
            clean["source"] == held_out
        ].copy()
        train = clean.loc[
            clean["source"] != held_out
        ].copy()
        if len(test) == 0:
            skipped.append({
                "target": target,
                "held_out_source": held_out,
                "reason": "No valid test labels.",
            })
            print(f"SKIP {held_out}: no test labels.")
            continue
        if len(train) == 0:
            skipped.append({
                "target": target,
                "held_out_source": held_out,
                "reason": "No valid training labels.",
            })
            print(f"SKIP {held_out}: no training labels.")
            continue
        train_grades = set(train["grade_id"].astype(str))
        test_grades = set(test["grade_id"].astype(str))
        overlap = train_grades.intersection(test_grades)
        if overlap:
            raise RuntimeError(
                f"Grade leakage detected for {target}, {held_out}: "
                f"{sorted(overlap)[:10]}"
            )
        if train[FEATURES].isna().any().any():
            raise RuntimeError("Missing training features detected.")
        if test[FEATURES].isna().any().any():
            raise RuntimeError("Missing test features detected.")
        name = f"{target}__heldout__{held_out}"
        print(
            f"\nRUN: {target} | held out {held_out}"
        )
        print(
            f"Train: {len(train)} records, "
            f"{len(train_grades)} grades"
        )
        print(
            f"Test: {len(test)} records, "
            f"{len(test_grades)} grades"
        )
        try:
            model = ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                train[FEATURES],
                train[target].astype(float),
            )
            predictions = model.predict(test[FEATURES])
            actual = test[target].to_numpy(dtype=float)
            mae = float(mean_absolute_error(actual, predictions))
            rmse = float(
                np.sqrt(mean_squared_error(actual, predictions))
            )
            r2 = (
                float(r2_score(actual, predictions))
                if len(actual) >= 2
                and np.var(actual) > 0
                else None
            )
            prediction_frame = pd.DataFrame({
                "source": test["source"].astype(str).to_numpy(),
                "grade_id": test["grade_id"].astype(str).to_numpy(),
                "heat_id": test["heat_id"].astype(str).to_numpy(),
                "actual": actual,
                "prediction": predictions,
                "residual": actual - predictions,
                "absolute_error": np.abs(actual - predictions),
            })
            prediction_path = (
                PREDICTION_OUT / f"{name}.csv"
            )
            model_path = MODEL_OUT / f"{name}.joblib"
            prediction_frame.to_csv(
                prediction_path,
                index=False,
            )
            joblib.dump(model, model_path)
            report = {
                "target": target,
                "variant": "common_C_Mn_only",
                "held_out_source": held_out,
                "status": "COMPLETED",
                "features": FEATURES,
                "train_records": len(train),
                "train_grades": len(train_grades),
                "test_records": len(test),
                "test_grades": len(test_grades),
                "grade_overlap_count": len(overlap),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "test_target_minimum": float(np.min(actual)),
                "test_target_median": float(np.median(actual)),
                "test_target_maximum": float(np.max(actual)),
                "model_path": str(model_path),
                "prediction_path": str(prediction_path),
                "limitations": [
                    "Experimental source-transfer benchmark only.",
                    "C and Mn feature units and source measurement conventions "
                    "have not yet been independently verified.",
                    "NIMS contains only four independent grade groups.",
                    "No uncertainty calibration is performed.",
                    "Not approved for engineering decisions.",
                ],
            }
            results.append(report)
            print(
                f"MAE={mae:.4f} | "
                f"RMSE={rmse:.4f} | "
                f"R2={r2 if r2 is not None else 'N/A'}"
            )
        except Exception as exc:
            failures.append({
                "target": target,
                "held_out_source": held_out,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            })
            print("FAILED:", exc)
summary = {
    "experiment": "MAT-X common-feature source-held-out benchmark",
    "features": FEATURES,
    "model": "ExtraTreesRegressor",
    "parameters": {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "successful_folds": len(results),
    "structurally_skipped_folds": len(skipped),
    "execution_failures": len(failures),
    "results": results,
    "skipped": skipped,
    "failures": failures,
    "interpretation_limitations": [
        "This is a diagnostic benchmark, not a promoted model.",
        "Two shared features do not represent complete material state.",
        "Cross-source chemistry measurement conventions remain to be verified.",
        "Source-held-out performance is empirical transfer evidence, "
        "not a guaranteed generalization bound.",
        "NIMS has only four independent grades.",
    ],
    "output_directory": str(OUT),
}
(OUT / "experiment_summary.json").write_text(
    json.dumps(summary, indent=2, default=str),
    encoding="utf-8",
)
if results:
    pd.DataFrame(results).drop(
        columns=["limitations"],
        errors="ignore",
    ).to_csv(
        OUT / "metrics.csv",
        index=False,
    )
print("\n" + "=" * 88)
print("COMMON-FEATURE BENCHMARK SUMMARY")
print("=" * 88)
print(json.dumps({
    "successful_folds": len(results),
    "structurally_skipped_folds": len(skipped),
    "execution_failures": len(failures),
    "output_directory": str(OUT),
}, indent=2))
if results:
    print("\nMETRICS")
    print(
        pd.DataFrame(results)[
            [
                "target",
                "held_out_source",
                "test_records",
                "test_grades",
                "mae",
                "rmse",
                "r2",
            ]
        ].to_string(index=False)
    )
print("\nExisting datasets and promoted models were not modified.")
