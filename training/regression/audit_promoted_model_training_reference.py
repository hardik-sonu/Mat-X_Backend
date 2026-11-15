import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
MODEL_DIR = ROOT / "models"
OUT = ROOT / "training_results" / "applicability_model_alignment_audit"
OUT.mkdir(parents=True, exist_ok=True)
NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T"
]
CATEGORICAL = ["quench_medium", "steel_family"]
FEATURES = NUMERIC + CATEGORICAL
TARGETS = {
    "tensile_strength": {
        "model": MODEL_DIR / "steelbench_tensile_extratrees.joblib",
        "expected_train": 1046,
        "expected_test": 314,
    },
    "yield_strength": {
        "model": MODEL_DIR / "steelbench_yield_extratrees.joblib",
        "expected_train": 783,
        "expected_test": 204,
    },
}
def clean_population(df, target):
    required = FEATURES + [target, "grade_id"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise RuntimeError(f"{target}: missing columns: {missing}")
    clean = df.dropna(subset=[target, "grade_id"]).copy()
    excluded = 0
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
    return clean, excluded
def compare_values(actual, expected, tolerance=1e-10):
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if pd.isna(a) and pd.isna(e):
            continue
        try:
            if not np.isclose(float(a), float(e), atol=tolerance, rtol=0):
                return False
        except (TypeError, ValueError):
            if str(a) != str(e):
                return False
    return True
def audit_target(df, target, config):
    clean, excluded = clean_population(df, target)
    X = clean[FEATURES]
    y = clean[target].astype(float)
    groups = clean["grade_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )
    X_train = X.iloc[train_idx]
    model = joblib.load(config["model"])
    if not hasattr(model, "named_steps"):
        raise TypeError(f"{target}: promoted artifact is not a Pipeline")
    preprocessing = model.named_steps["preprocessing"]
    numeric_pipeline = preprocessing.named_transformers_["numeric"]
    categorical_pipeline = preprocessing.named_transformers_["categorical"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["encoder"]
    reconstructed_numeric = (
        X_train[NUMERIC].apply(pd.to_numeric, errors="coerce")
        .median()
        .to_numpy(dtype=float)
    )
    promoted_numeric = np.asarray(
        numeric_imputer.statistics_, dtype=float
    )
    reconstructed_modes = []
    for column in CATEGORICAL:
        values = X_train[column].dropna()
        if values.empty:
            reconstructed_modes.append(None)
        else:
            reconstructed_modes.append(values.mode().iloc[0])
    promoted_modes = list(categorical_imputer.statistics_)
    reconstructed_categories = []
    for column, mode in zip(CATEGORICAL, reconstructed_modes):
        values = X_train[column].dropna()
        if values.empty:
            values = pd.Series([mode])
        reconstructed_categories.append(
            np.sort(values.astype(str).unique())
        )
    promoted_categories = [
        np.sort(np.asarray(categories).astype(str))
        for categories in encoder.categories_
    ]
    numeric_match = compare_values(
        reconstructed_numeric, promoted_numeric
    )
    mode_match = compare_values(
        reconstructed_modes, promoted_modes
    )
    category_match = (
        len(reconstructed_categories) == len(promoted_categories)
        and all(
            np.array_equal(a, b)
            for a, b in zip(
                reconstructed_categories, promoted_categories
            )
        )
    )
    result = {
        "target": target,
        "model_path": str(config["model"]),
        "clean_records": int(len(clean)),
        "excluded_anomaly_records": excluded,
        "reconstructed_training_records": int(len(train_idx)),
        "reconstructed_test_records": int(len(test_idx)),
        "reconstructed_training_grades": int(
            groups.iloc[train_idx].nunique()
        ),
        "reconstructed_test_grades": int(
            groups.iloc[test_idx].nunique()
        ),
        "expected_training_records": config["expected_train"],
        "expected_test_records": config["expected_test"],
        "training_count_match": len(train_idx) == config["expected_train"],
        "test_count_match": len(test_idx) == config["expected_test"],
        "numeric_imputer_match": bool(numeric_match),
        "categorical_imputer_match": bool(mode_match),
        "categorical_encoder_categories_match": bool(category_match),
        "overall_preprocessing_match": bool(
            numeric_match and mode_match and category_match
        ),
        "reconstructed_numeric_medians": dict(
            zip(NUMERIC, reconstructed_numeric.tolist())
        ),
        "promoted_numeric_medians": dict(
            zip(NUMERIC, promoted_numeric.tolist())
        ),
        "reconstructed_categorical_modes": dict(
            zip(CATEGORICAL, reconstructed_modes)
        ),
        "promoted_categorical_modes": dict(
            zip(CATEGORICAL, promoted_modes)
        ),
        "reconstructed_categories": {
            name: values.tolist()
            for name, values in zip(
                CATEGORICAL, reconstructed_categories
            )
        },
        "promoted_categories": {
            name: values.tolist()
            for name, values in zip(
                CATEGORICAL, promoted_categories
            )
        },
    }
    return result
def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    df = pd.read_csv(DATASET)
    results = {}
    for target, config in TARGETS.items():
        print(f"\nAuditing {target}...", flush=True)
        results[target] = audit_target(df, target, config)
        r = results[target]
        print("Training records:", r["reconstructed_training_records"])
        print("Test records:", r["reconstructed_test_records"])
        print("Training count match:", r["training_count_match"])
        print("Test count match:", r["test_count_match"])
        print("Numeric imputer match:", r["numeric_imputer_match"])
        print("Categorical imputer match:", r["categorical_imputer_match"])
        print("Encoder categories match:", r["categorical_encoder_categories_match"])
        print("Overall preprocessing match:", r["overall_preprocessing_match"])
    report = {
        "audit": "PROMOTED_MODEL_TRAINING_REFERENCE_RECONSTRUCTION",
        "method": "Exact GroupShuffleSplit reconstruction and fitted preprocessing comparison",
        "source_dataset": str(DATASET),
        "models_modified": False,
        "dataset_modified": False,
        "results": results,
    }
    output = OUT / "promoted_model_alignment_report.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n========== FINAL AUDIT ==========")
    for target, result in results.items():
        print(
            target,
            "PREPROCESSING_MATCH=",
            result["overall_preprocessing_match"]
        )
    print("\nReport:", output)
if __name__ == "__main__":
    main()
