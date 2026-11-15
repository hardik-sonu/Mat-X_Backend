from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
ROOT = Path.cwd()
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
MODEL_DIR = (
    ROOT / "training_results"
    / "controlled_feature_ablation_v1"
)
OUT = (
    ROOT / "training_results"
    / "source_transfer_diagnostic_v1"
)
if OUT.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing output: {OUT}"
    )
if not DATASET.exists():
    raise FileNotFoundError(DATASET)
PREDICTION_DIR = MODEL_DIR / "predictions"
if not PREDICTION_DIR.exists():
    raise FileNotFoundError(PREDICTION_DIR)
OUT.mkdir(parents=True)
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
targets = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
features = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V",
    "Cu", "Al", "austenitize_T", "temper_T",
    "quench_medium", "steel_family",
]
numeric_features = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V",
    "Cu", "Al", "austenitize_T", "temper_T",
]
for col in numeric_features + targets:
    df[col] = pd.to_numeric(df[col], errors="coerce")
sources = sorted(df["source"].dropna().unique())
# ---------------------------------------------------------
# 1. Feature missingness by source
# ---------------------------------------------------------
missing_rows = []
for source in sources:
    subset = df.loc[df["source"] == source]
    for feature in features:
        missing_rows.append({
            "source": source,
            "feature": feature,
            "records": len(subset),
            "missing_records": int(subset[feature].isna().sum()),
            "missing_fraction": float(
                subset[feature].isna().mean()
            ),
            "unique_nonmissing": int(
                subset[feature].nunique(dropna=True)
            ),
        })
missingness = pd.DataFrame(missing_rows)
missingness.to_csv(
    OUT / "feature_missingness_by_source.csv",
    index=False,
)
# ---------------------------------------------------------
# 2. Source-specific feature distributions
# ---------------------------------------------------------
distribution_rows = []
for source in sources:
    subset = df.loc[df["source"] == source]
    for feature in numeric_features:
        values = subset[feature].dropna()
        distribution_rows.append({
            "source": source,
            "feature": feature,
            "count": len(values),
            "mean": (
                float(values.mean()) if len(values) else None
            ),
            "std": (
                float(values.std()) if len(values) > 1 else None
            ),
            "minimum": (
                float(values.min()) if len(values) else None
            ),
            "median": (
                float(values.median()) if len(values) else None
            ),
            "maximum": (
                float(values.max()) if len(values) else None
            ),
        })
pd.DataFrame(distribution_rows).to_csv(
    OUT / "numeric_feature_distributions.csv",
    index=False,
)
# ---------------------------------------------------------
# 3. Prediction bias and residuals by source
# ---------------------------------------------------------
residual_rows = []
for target in targets:
    for source in sources:
        path = (
            PREDICTION_DIR
            / f"{target}__{source}.csv"
        )
        # Controlled-ablation filenames are target__heldout__source.
        path = (
            PREDICTION_DIR
            / f"{target}__heldout__{source}.csv"
        )
        if not path.exists():
            print(f"NO PREDICTION FILE: {target} | {source}")
            continue
        pred = pd.read_csv(path)
        required = [
            "source",
            "grade_id",
            "heat_id",
            "actual",
            "prediction",
        ]
        if not set(required).issubset(pred.columns):
            raise RuntimeError(
                f"Unexpected prediction schema: {path}"
            )
        actual = pred["actual"].to_numpy(dtype=float)
        prediction = pred["prediction"].to_numpy(dtype=float)
        residual = prediction - actual
        residual_rows.append({
            "target": target,
            "source": source,
            "records": len(pred),
            "grades": pred["grade_id"].nunique(),
            "mean_prediction_bias": float(np.mean(residual)),
            "median_prediction_bias": float(np.median(residual)),
            "mean_absolute_error": float(np.mean(np.abs(residual))),
            "root_mean_squared_error": float(
                np.sqrt(np.mean(residual ** 2))
            ),
            "residual_std": float(np.std(residual, ddof=1)),
            "underprediction_fraction": float(
                np.mean(residual < 0)
            ),
            "overprediction_fraction": float(
                np.mean(residual > 0)
            ),
        })
residuals = pd.DataFrame(residual_rows)
residuals.to_csv(
    OUT / "source_prediction_bias.csv",
    index=False,
)
# ---------------------------------------------------------
# 4. Feature importance from controlled full_13 models
# ---------------------------------------------------------
importance_rows = []
model_dir = MODEL_DIR / "models"
if model_dir.exists():
    model_files = sorted(model_dir.glob("*.joblib"))
    for model_path in model_files:
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            if not hasattr(model, "named_steps"):
                continue
        estimator = (
            model.named_steps["model"]
            if hasattr(model, "named_steps")
            and "model" in model.named_steps
            else model
        )
        if not hasattr(estimator, "feature_importances_"):
            continue
        values = estimator.feature_importances_
        # The pipeline may expand categorical variables through
        # one-hot encoding, so raw input names cannot safely be
        # paired with transformed importances.
        if len(values) != len(features):
            print(
                "TRANSFORMED FEATURE SPACE:",
                model_path.name,
                "importance_count=",
                len(values),
            )
            continue
        for feature, importance in zip(features, values):
            importance_rows.append({
                "model_file": model_path.name,
                "feature": feature,
                "importance": float(importance),
            })
pd.DataFrame(importance_rows).to_csv(
    OUT / "feature_importance_if_raw_alignment_valid.csv",
    index=False,
)
# ---------------------------------------------------------
# 5. Summary and integrity
# ---------------------------------------------------------
summary = {
    "experiment": "MAT-X source transfer diagnostic v1",
    "read_only": True,
    "dataset_records": len(df),
    "sources": {
        str(source): int(
            (df["source"] == source).sum()
        )
        for source in sources
    },
    "targets": targets,
    "features_audited": features,
    "prediction_bias_rows": len(residuals),
    "limitations": [
        "Feature importance is omitted when transformed feature "
        "dimensions cannot be safely mapped to original columns.",
        "Prediction bias is descriptive and source-specific.",
        "This audit does not establish causal relationships.",
        "No model is retrained or promoted.",
    ],
    "output_directory": str(OUT),
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\n" + "=" * 90)
print("SOURCE TRANSFER DIAGNOSTIC COMPLETE")
print("=" * 90)
print("\nFEATURE MISSINGNESS")
print(
    missingness.pivot(
        index="feature",
        columns="source",
        values="missing_fraction",
    ).round(3).to_string()
)
print("\nSOURCE-SPECIFIC PREDICTION BIAS")
if not residuals.empty:
    print(residuals.to_string(index=False))
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print("\nOriginal datasets, promoted models and prior results preserved.")
