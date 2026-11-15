import json
import math
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from app.core.paths import STEELBENCH_PATH
ROOT = Path(__file__).resolve().parents[2]
DATASET = STEELBENCH_PATH
AUDIT_DIR = ROOT / "training_results" / "applicability_promoted_reference_audit"
MODEL_DIR = ROOT / "models"
NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL = ["quench_medium", "steel_family"]
FEATURES = NUMERIC + CATEGORICAL
MISSING_CATEGORY = "__MATX_MISSING_CATEGORY__"
TARGETS = {
    "tensile_strength": {
        "expected_records": 1046,
        "expected_grades": 449,
        "model": "steelbench_tensile_extratrees.joblib",
        "scores": "tensile_strength_fixed_full_reference_grade_scores.csv",
        "output": "applicability_tensile_reference_v1.json",
        "expected_q90": 0.4804347170203321,
        "expected_q95": 0.8860731658536412,
        "excluded_anomalies": 0,
    },
    "yield_strength": {
        "expected_records": 783,
        "expected_grades": 446,
        "model": "steelbench_yield_extratrees.joblib",
        "scores": "yield_strength_fixed_full_reference_grade_scores.csv",
        "output": "applicability_yield_reference_v1.json",
        "expected_q90": 0.4936579409568959,
        "expected_q95": 0.8198871602602545,
        "excluded_anomalies": 13,
    },
}
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def finite_sample_threshold(values, confidence):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    rank = int(math.ceil((n + 1) * confidence))
    if n == 0 or rank > n:
        raise RuntimeError(
            f"Cannot calculate finite-sample threshold: n={n}, rank={rank}"
        )
    return float(np.sort(values)[rank - 1]), rank
def fit_reference(frame):
    numeric_reference = {}
    categorical_reference = {}
    eligible_numeric = []
    eligible_categorical = []
    for feature in NUMERIC:
        values = pd.to_numeric(
            frame[feature], errors="coerce"
        ).to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            continue
        median = float(np.median(valid))
        q25 = float(np.quantile(valid, 0.25))
        q75 = float(np.quantile(valid, 0.75))
        q01 = float(np.quantile(valid, 0.01))
        q99 = float(np.quantile(valid, 0.99))
        minimum = float(np.min(valid))
        maximum = float(np.max(valid))
        iqr = q75 - q25
        std = float(np.std(valid))
        tolerance = np.finfo(float).eps * max(1.0, abs(median)) * 100
        if np.unique(valid).size <= 1 or std <= tolerance:
            continue
        scale = iqr
        if not np.isfinite(scale) or scale <= tolerance:
            scale = std
        if not np.isfinite(scale) or scale <= tolerance:
            continue
        eligible_numeric.append(feature)
        numeric_reference[feature] = {
            "median": median,
            "scale": float(scale),
            "minimum": minimum,
            "maximum": maximum,
            "q01": q01,
            "q99": q99,
        }
    for feature in CATEGORICAL:
        values = (
            frame[feature]
            .astype("string")
            .fillna(MISSING_CATEGORY)
            .astype(str)
        )
        observed = values[values != MISSING_CATEGORY]
        known = sorted(set(observed))
        if len(known) > 1:
            eligible_categorical.append(feature)
            categorical_reference[feature] = known
    dimensions = len(eligible_numeric) + len(eligible_categorical)
    if dimensions != 13:
        raise RuntimeError(
            f"Expected 13 eligible dimensions, got {dimensions}"
        )
    return {
        "numeric": eligible_numeric,
        "categorical": eligible_categorical,
        "numeric_reference": numeric_reference,
        "categorical_reference": categorical_reference,
    }
def json_safe_record(row):
    result = {}
    for feature in NUMERIC:
        value = pd.to_numeric(
            pd.Series([row[feature]]), errors="coerce"
        ).iloc[0]
        result[feature] = (
            float(value) if pd.notna(value) and np.isfinite(value) else None
        )
    for feature in CATEGORICAL:
        value = row[feature]
        result[feature] = None if pd.isna(value) else str(value)
    result["grade_id"] = str(row["grade_id"])
    return result
def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    source_hash = sha256(DATASET)
    df = pd.read_csv(DATASET)
    print("Dataset SHA256:", source_hash)
    for target, config in TARGETS.items():
        print(f"\n========== {target.upper()} ==========")
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
        X = clean[FEATURES].copy()
        groups = clean["grade_id"].astype(str)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42,
        )
        train_idx, _ = next(splitter.split(X, groups=groups))
        train = X.iloc[train_idx].copy().reset_index(drop=True)
        train["grade_id"] = groups.iloc[train_idx].to_numpy()
        records = len(train)
        grades = int(train["grade_id"].nunique())
        if records != config["expected_records"]:
            raise RuntimeError(
                f"Training record mismatch: {records}"
            )
        if grades != config["expected_grades"]:
            raise RuntimeError(
                f"Training grade mismatch: {grades}"
            )
        if excluded != config["excluded_anomalies"]:
            raise RuntimeError(
                f"Anomaly exclusion mismatch: {excluded}"
            )
        reference = fit_reference(train)
        score_path = AUDIT_DIR / config["scores"]
        model_path = MODEL_DIR / config["model"]
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        scores = pd.read_csv(score_path)
        maxima = scores["maximum_nearest_distance"].to_numpy(dtype=float)
        q90, rank90 = finite_sample_threshold(maxima, 0.90)
        q95, rank95 = finite_sample_threshold(maxima, 0.95)
        if not np.isclose(
            q90, config["expected_q90"], rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"q90 mismatch: {q90}")
        if not np.isclose(
            q95, config["expected_q95"], rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"q95 mismatch: {q95}")
        if len(scores) != grades:
            raise RuntimeError("Grade-score count mismatch.")
        model_hash = sha256(model_path)
        artifact = {
            "artifact_version": "1.0",
            "status": "EXPERIMENTAL_RUNTIME_SCREENING_REFERENCE",
            "target": target,
            "algorithm": {
                "name": "MIXED_NUMERIC_CATEGORICAL_NEAREST_TRAINING_RECORD",
                "numeric_scaling": "training_median_and_IQR_with_STD_fallback",
                "numeric_missing_handling": "training_median",
                "categorical_distance": "exact_mismatch_indicator",
                "distance": "sqrt(sum_squared_numeric_and_categorical_distance / dimensions)",
                "nearest_reference": "minimum_distance_to_training_records",
                "calibration_unit": "grade_id",
                "calibration_score": "maximum_record_to_nearest_other_grade_distance",
                "threshold_method": "finite_sample_order_statistic",
            },
            "features": {
                "numeric": reference["numeric"],
                "categorical": reference["categorical"],
                "all": FEATURES,
            },
            "reference_statistics": {
                "numeric": reference["numeric_reference"],
                "categorical": reference["categorical_reference"],
                "missing_category_token": MISSING_CATEGORY,
            },
            "thresholds": {
                "q90": q90,
                "q90_rank": rank90,
                "q95": q95,
                "q95_rank": rank95,
                "calibration_grades": int(len(scores)),
                "calibration_records": int(scores["records"].sum()),
            },
            "training_reference": {
                "records": records,
                "grades": grades,
                "excluded_yield_anomaly_records": excluded,
                "split": {
                    "method": "GroupShuffleSplit",
                    "test_size": 0.20,
                    "random_state": 42,
                    "group_column": "grade_id",
                },
                "records_data": [
                    json_safe_record(row)
                    for _, row in train.iterrows()
                ],
            },
            "provenance": {
                "source_dataset": str(DATASET),
                "source_dataset_sha256": source_hash,
                "promoted_model": str(model_path),
                "promoted_model_sha256": model_hash,
                "calibration_scores_file": str(score_path),
                "promoted_model_modified": False,
                "source_dataset_modified": False,
            },
            "interpretation_limit": (
                "Experimental applicability-distance screening only. "
                "Not predictive uncertainty, probability of correctness, "
                "prediction-error calibration, or external validation."
            ),
        }
        output_path = MODEL_DIR / config["output"]
        output_path.write_text(
            json.dumps(artifact, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        # Reopen and verify the saved artifact.
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if len(saved["training_reference"]["records_data"]) != records:
            raise RuntimeError("Saved reference record count mismatch.")
        print("Training records:", records)
        print("Training grades:", grades)
        print("Eligible dimensions:", len(reference["numeric"]) + len(reference["categorical"]))
        print("Calibration grades:", len(scores))
        print("q90:", q90, "rank:", rank90)
        print("q95:", q95, "rank:", rank95)
        print("Model SHA256:", model_hash)
        print("Artifact:", output_path)
        print("Artifact bytes:", output_path.stat().st_size)
        print("Saved artifact verification: PASS")
    print("\nREFERENCE ARTIFACT GENERATION: COMPLETE")
    print("Original datasets and promoted model artifacts were not modified.")
if __name__ == "__main__":
    main()
