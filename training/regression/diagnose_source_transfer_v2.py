from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
ROOT = Path.cwd()
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
BASE = ROOT / "training_results" / "controlled_feature_ablation_v1"
PREDICTION_DIR = BASE / "predictions"
MODEL_DIR = BASE / "models"
OUT = (
    ROOT / "training_results"
    / "source_transfer_diagnostic_v2"
)
if OUT.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing output: {OUT}"
    )
if not DATASET.exists():
    raise FileNotFoundError(DATASET)
if not PREDICTION_DIR.exists() or not MODEL_DIR.exists():
    raise FileNotFoundError("Controlled ablation artifacts missing.")
OUT.mkdir(parents=True)
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
VARIANTS = [
    "full_13",
    "chemistry_2",
    "chemistry_3",
]
FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V",
    "Cu", "Al", "austenitize_T", "temper_T",
    "quench_medium", "steel_family",
]
NUMERIC_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V",
    "Cu", "Al", "austenitize_T", "temper_T",
]
# ---------------------------------------------------------
# 1. Read-only source feature audit
# ---------------------------------------------------------
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for col in NUMERIC_FEATURES + TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
sources = sorted(df["source"].dropna().unique())
missing_rows = []
distribution_rows = []
for source in sources:
    subset = df.loc[df["source"] == source]
    for feature in FEATURES:
        missing_rows.append({
            "source": source,
            "feature": feature,
            "records": len(subset),
            "missing_records": int(subset[feature].isna().sum()),
            "missing_fraction": float(subset[feature].isna().mean()),
            "unique_nonmissing": int(
                subset[feature].nunique(dropna=True)
            ),
        })
    for feature in NUMERIC_FEATURES:
        values = subset[feature].dropna()
        distribution_rows.append({
            "source": source,
            "feature": feature,
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "std": (
                float(values.std())
                if len(values) > 1 else None
            ),
            "median": (
                float(values.median())
                if len(values) else None
            ),
            "minimum": (
                float(values.min())
                if len(values) else None
            ),
            "maximum": (
                float(values.max())
                if len(values) else None
            ),
        })
missingness = pd.DataFrame(missing_rows)
distributions = pd.DataFrame(distribution_rows)
missingness.to_csv(
    OUT / "feature_missingness_by_source.csv",
    index=False,
)
distributions.to_csv(
    OUT / "numeric_feature_distributions.csv",
    index=False,
)
# ---------------------------------------------------------
# 2. Read prediction files from nested directories
# ---------------------------------------------------------
residual_rows = []
prediction_integrity = []
for target in TARGETS:
    target_dir = PREDICTION_DIR / target
    if not target_dir.exists():
        print("MISSING TARGET DIRECTORY:", target_dir)
        continue
    for source_dir in sorted(target_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for variant in VARIANTS:
            path = source_dir / f"{variant}.csv"
            if not path.exists():
                prediction_integrity.append({
                    "target": target,
                    "held_out_source": source,
                    "variant": variant,
                    "file_found": False,
                    "integrity_passed": False,
                    "reason": "Prediction file missing",
                })
                continue
            pred = pd.read_csv(path)
            required = {
                "source",
                "grade_id",
                "heat_id",
                "actual",
                "prediction",
            }
            missing_columns = required - set(pred.columns)
            if missing_columns:
                raise RuntimeError(
                    f"{path} missing columns: {missing_columns}"
                )
            actual = pd.to_numeric(
                pred["actual"], errors="coerce"
            ).to_numpy(dtype=float)
            prediction = pd.to_numeric(
                pred["prediction"], errors="coerce"
            ).to_numpy(dtype=float)
            if (
                len(pred) == 0
                or not np.isfinite(actual).all()
                or not np.isfinite(prediction).all()
            ):
                raise RuntimeError(
                    f"Invalid prediction values: {path}"
                )
            residual = prediction - actual
            residual_rows.append({
                "target": target,
                "held_out_source": source,
                "variant": variant,
                "records": len(pred),
                "grades": pred["grade_id"].nunique(),
                "mean_prediction_bias": float(
                    np.mean(residual)
                ),
                "median_prediction_bias": float(
                    np.median(residual)
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(residual))
                ),
                "root_mean_squared_error": float(
                    np.sqrt(np.mean(residual ** 2))
                ),
                "residual_std": float(
                    np.std(residual, ddof=1)
                ) if len(residual) > 1 else 0.0,
                "underprediction_fraction": float(
                    np.mean(residual < 0)
                ),
                "overprediction_fraction": float(
                    np.mean(residual > 0)
                ),
            })
            prediction_integrity.append({
                "target": target,
                "held_out_source": source,
                "variant": variant,
                "file_found": True,
                "integrity_passed": True,
                "records": len(pred),
                "grades": pred["grade_id"].nunique(),
                "reason": "Finite predictions and actuals",
            })
residuals = pd.DataFrame(residual_rows)
integrity = pd.DataFrame(prediction_integrity)
residuals.to_csv(
    OUT / "source_prediction_bias_and_errors.csv",
    index=False,
)
integrity.to_csv(
    OUT / "prediction_file_integrity.csv",
    index=False,
)
# ---------------------------------------------------------
# 3. Recover feature importance from fitted pipelines
# ---------------------------------------------------------
transformed_rows = []
aggregated_rows = []
importance_integrity = []
for target in TARGETS:
    for source in sources:
        model_path = (
            MODEL_DIR
            / f"{target}__{source}__full_13.joblib"
        )
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "named_steps"):
            importance_integrity.append({
                "target": target,
                "held_out_source": source,
                "status": "NOT_A_PIPELINE",
            })
            continue
        preprocessing = model.named_steps.get("preprocessing")
        estimator = model.named_steps.get("model")
        if preprocessing is None or estimator is None:
            continue
        if not hasattr(preprocessing, "get_feature_names_out"):
            continue
        if not hasattr(estimator, "feature_importances_"):
            continue
        transformed_names = list(
            preprocessing.get_feature_names_out()
        )
        importances = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )
        if len(transformed_names) != len(importances):
            raise RuntimeError(
                f"Feature importance dimension mismatch: {model_path}"
            )
        if not np.isfinite(importances).all():
            raise RuntimeError(
                f"Non-finite feature importance: {model_path}"
            )
        grouped = {}
        for name, importance in zip(
            transformed_names, importances
        ):
            original_feature = None
            # Match original feature names, including one-hot
            # encoded categories with underscores.
            for feature in sorted(
                FEATURES, key=len, reverse=True
            ):
                if (
                    name == f"numeric__{feature}"
                    or name == f"categorical__{feature}"
                    or name.startswith(
                        f"categorical__{feature}_"
                    )
                ):
                    original_feature = feature
                    break
            if original_feature is None:
                raise RuntimeError(
                    f"Cannot map transformed feature {name} "
                    f"in {model_path.name}"
                )
            transformed_rows.append({
                "target": target,
                "held_out_source": source,
                "transformed_feature": name,
                "original_feature": original_feature,
                "importance": float(importance),
            })
            grouped[original_feature] = (
                grouped.get(original_feature, 0.0)
                + float(importance)
            )
        total = sum(grouped.values())
        for feature, importance in grouped.items():
            aggregated_rows.append({
                "target": target,
                "held_out_source": source,
                "feature": feature,
                "importance": importance,
                "normalized_importance": (
                    importance / total if total > 0 else 0.0
                ),
            })
        importance_integrity.append({
            "target": target,
            "held_out_source": source,
            "status": "MAPPED",
            "transformed_feature_count": len(transformed_names),
            "original_feature_groups": len(grouped),
            "importance_sum": float(total),
        })
transformed_importance = pd.DataFrame(transformed_rows)
aggregated_importance = pd.DataFrame(aggregated_rows)
importance_audit = pd.DataFrame(importance_integrity)
transformed_importance.to_csv(
    OUT / "transformed_feature_importance.csv",
    index=False,
)
aggregated_importance.to_csv(
    OUT / "aggregated_original_feature_importance.csv",
    index=False,
)
importance_audit.to_csv(
    OUT / "feature_importance_integrity.csv",
    index=False,
)
# ---------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------
summary = {
    "experiment": "MAT-X source transfer diagnostic v2",
    "read_only": True,
    "dataset_records": len(df),
    "sources": {
        str(source): int(
            (df["source"] == source).sum()
        )
        for source in sources
    },
    "prediction_files_found": int(
        integrity["file_found"].sum()
    ) if not integrity.empty else 0,
    "prediction_integrity_passed": int(
        integrity["integrity_passed"].sum()
    ) if not integrity.empty else 0,
    "prediction_bias_rows": len(residuals),
    "feature_importance_models_mapped": int(
        (importance_audit["status"] == "MAPPED").sum()
    ) if not importance_audit.empty else 0,
    "importance_mapping_method": (
        "Transformed one-hot feature importances aggregated "
        "to original input feature groups."
    ),
    "limitations": [
        "Impurity-based feature importance is not causal evidence.",
        "Importance is model- and held-out-fold-specific.",
        "Prediction bias is descriptive.",
        "Source differences may reflect genuine population differences.",
        "No models were retrained or promoted.",
    ],
    "output_directory": str(OUT),
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
# ---------------------------------------------------------
# 5. Console report
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("MAT-X SOURCE TRANSFER DIAGNOSTIC V2")
print("=" * 100)
print("\nPREDICTION ERRORS AND BIAS")
if not residuals.empty:
    print(
        residuals[
            [
                "target",
                "held_out_source",
                "variant",
                "records",
                "mean_prediction_bias",
                "mean_absolute_error",
                "root_mean_squared_error",
            ]
        ].round(4).to_string(index=False)
    )
print("\nAGGREGATED FULL-13 FEATURE IMPORTANCE")
if not aggregated_importance.empty:
    print(
        aggregated_importance.sort_values(
            ["target", "held_out_source", "normalized_importance"],
            ascending=[True, True, False],
        ).to_string(index=False)
    )
print("\nFEATURE IMPORTANCE INTEGRITY")
if not importance_audit.empty:
    print(importance_audit.to_string(index=False))
print("\nPREDICTION FILE INTEGRITY")
if not integrity.empty:
    print(
        integrity.groupby(
            ["file_found", "integrity_passed"]
        ).size().to_string()
    )
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print("\nAll original datasets, models and prior experiment outputs preserved.")
