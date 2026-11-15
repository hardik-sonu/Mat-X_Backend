import json
import math
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATASET = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = ROOT / "training_results" / "applicability_promoted_reference_audit"
OUT.mkdir(parents=True, exist_ok=True)
NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL = ["quench_medium", "steel_family"]
FEATURES = NUMERIC + CATEGORICAL
MISSING_CATEGORY = "__MATX_MISSING_CATEGORY__"
TARGETS = {
    "tensile_strength": {
        "expected_train": 1046,
        "expected_grades": 449,
        "model": ROOT / "models" / "steelbench_tensile_extratrees.joblib",
    },
    "yield_strength": {
        "expected_train": 783,
        "expected_grades": 446,
        "model": ROOT / "models" / "steelbench_yield_extratrees.joblib",
    },
}
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def finite_sample_threshold(values, confidence):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return None, None
    rank = int(math.ceil((n + 1) * confidence))
    if rank > n:
        return None, rank
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
            frame[feature].astype("string")
            .fillna(MISSING_CATEGORY).astype(str)
        )
        observed = values[values != MISSING_CATEGORY]
        known = set(observed)
        if len(known) > 1:
            eligible_categorical.append(feature)
            categorical_reference[feature] = known
    if not eligible_numeric and not eligible_categorical:
        raise RuntimeError("No eligible features in reference.")
    return {
        "numeric": eligible_numeric,
        "categorical": eligible_categorical,
        "numeric_reference": numeric_reference,
        "categorical_reference": categorical_reference,
    }
def transform_numeric(frame, reference):
    features = reference["numeric"]
    if not features:
        return np.empty((len(frame), 0), dtype=float)
    matrix = np.zeros((len(frame), len(features)), dtype=float)
    for j, feature in enumerate(features):
        values = pd.to_numeric(
            frame[feature], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        valid = np.isfinite(values)
        values[~valid] = reference["numeric_reference"][feature]["median"]
        item = reference["numeric_reference"][feature]
        matrix[:, j] = (values - item["median"]) / item["scale"]
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite numeric transformation.")
    return matrix
def transform_categories(frame, reference):
    result = {}
    for feature in reference["categorical"]:
        result[feature] = (
            frame[feature].astype("string")
            .fillna(MISSING_CATEGORY).astype(str).to_numpy()
        )
    return result
def distance_matrix(frame_a, frame_b, reference):
    numeric_a = transform_numeric(frame_a, reference)
    numeric_b = transform_numeric(frame_b, reference)
    categorical_a = transform_categories(frame_a, reference)
    categorical_b = transform_categories(frame_b, reference)
    dimensions = numeric_a.shape[1] + len(categorical_a)
    if dimensions == 0:
        raise RuntimeError("No dimensions available for distance.")
    if numeric_a.shape[1]:
        squared = cdist(numeric_a, numeric_b, metric="sqeuclidean")
    else:
        squared = np.zeros((len(frame_a), len(frame_b)), dtype=float)
    for feature in categorical_a:
        squared += (
            categorical_a[feature][:, None]
            != categorical_b[feature][None, :]
        ).astype(float)
    return np.sqrt(squared / dimensions)
def grade_calibration(train, mode):
    rows = []
    grades = sorted(train["grade_id"].astype(str).unique())
    full_reference = fit_reference(train) if mode == "fixed_full_reference" else None
    for index, grade in enumerate(grades, start=1):
        query_mask = train["grade_id"].astype(str).to_numpy() == grade
        query = train.loc[query_mask]
        reference_frame = train.loc[~query_mask]
        if len(query) == 0 or len(reference_frame) == 0:
            raise RuntimeError(f"Empty query/reference for grade {grade}")
        if mode == "leave_one_grade_out_refit":
            reference = fit_reference(reference_frame)
        else:
            reference = full_reference
        distances = distance_matrix(query, reference_frame, reference)
        nearest = np.min(distances, axis=1)
        if not np.isfinite(nearest).all():
            raise RuntimeError(f"Non-finite distances for grade {grade}")
        rows.append({
            "grade_id": grade,
            "records": int(len(query)),
            "reference_records": int(len(reference_frame)),
            "eligible_numeric_features": len(reference["numeric"]),
            "eligible_categorical_features": len(reference["categorical"]),
            "eligible_dimensions": (
                len(reference["numeric"]) + len(reference["categorical"])
            ),
            "mean_nearest_distance": float(np.mean(nearest)),
            "median_nearest_distance": float(np.median(nearest)),
            "maximum_nearest_distance": float(np.max(nearest)),
            "minimum_nearest_distance": float(np.min(nearest)),
        })
        if index % 100 == 0:
            print(f"  {mode}: processed {index}/{len(grades)} grades")
    return pd.DataFrame(rows)
def summarize(frame):
    maxima = frame["maximum_nearest_distance"].to_numpy(dtype=float)
    q90, rank90 = finite_sample_threshold(maxima, 0.90)
    q95, rank95 = finite_sample_threshold(maxima, 0.95)
    return {
        "calibration_unit": "grade_id",
        "calibration_score": "maximum record-to-nearest-other-grade distance",
        "grade_count": int(len(frame)),
        "record_count": int(frame["records"].sum()),
        "q90": q90,
        "q90_rank": rank90,
        "q95": q95,
        "q95_rank": rank95,
        "grade_maximum_distance_median": float(np.median(maxima)),
        "grade_maximum_distance_mean": float(np.mean(maxima)),
        "grade_maximum_distance_min": float(np.min(maxima)),
        "grade_maximum_distance_max": float(np.max(maxima)),
        "eligible_dimensions_min": int(frame["eligible_dimensions"].min()),
        "eligible_dimensions_max": int(frame["eligible_dimensions"].max()),
        "eligible_dimensions_unique": sorted(
            frame["eligible_dimensions"].unique().astype(int).tolist()
        ),
    }
def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    df = pd.read_csv(DATASET)
    results = {}
    for target, config in TARGETS.items():
        print(f"\n========== {target.upper()} ==========")
        clean = df.dropna(subset=[target, "grade_id"]).copy()
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
        else:
            excluded = 0
        X = clean[FEATURES].copy()
        groups = clean["grade_id"].astype(str)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=42
        )
        train_idx, _ = next(splitter.split(X, groups=groups))
        train = X.iloc[train_idx].copy().reset_index(drop=True)
        train["grade_id"] = groups.iloc[train_idx].to_numpy()
        train_records = len(train)
        train_grades = int(train["grade_id"].nunique())
        if train_records != config["expected_train"]:
            raise RuntimeError(
                f"{target}: training count mismatch: {train_records}"
            )
        if train_grades != config["expected_grades"]:
            raise RuntimeError(
                f"{target}: grade count mismatch: {train_grades}"
            )
        model_path = config["model"]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model_hash_before = sha256(model_path)
        print("Training records:", train_records)
        print("Training grades:", train_grades)
        print("Yield anomaly records excluded:", excluded)
        print("Promoted model SHA256:", model_hash_before)
        modes = {}
        for mode in (
            "leave_one_grade_out_refit",
            "fixed_full_reference",
        ):
            print(f"\nRunning {mode}...")
            frame = grade_calibration(train, mode)
            summary = summarize(frame)
            csv_path = OUT / f"{target}_{mode}_grade_scores.csv"
            frame.to_csv(csv_path, index=False)
            modes[mode] = {
                "summary": summary,
                "grade_scores_csv": str(csv_path),
            }
            print(json.dumps(summary, indent=2))
        a = modes["leave_one_grade_out_refit"]["summary"]
        b = modes["fixed_full_reference"]["summary"]
        comparison = {}
        for key in ("q90", "q95"):
            av, bv = a[key], b[key]
            comparison[key] = {
                "leave_one_grade_out_refit": av,
                "fixed_full_reference": bv,
                "absolute_difference": (
                    abs(av - bv) if av is not None and bv is not None else None
                ),
                "relative_difference_fraction": (
                    abs(av - bv) / abs(bv)
                    if av is not None and bv not in (None, 0)
                    else None
                ),
            }
        model_hash_after = sha256(model_path)
        if model_hash_before != model_hash_after:
            raise RuntimeError(f"Promoted model hash changed: {target}")
        results[target] = {
            "training_records": train_records,
            "training_grades": train_grades,
            "yield_anomaly_records_excluded": excluded,
            "promoted_model_sha256_before": model_hash_before,
            "promoted_model_sha256_after": model_hash_after,
            "model_unchanged": model_hash_before == model_hash_after,
            "methods": modes,
            "threshold_comparison": comparison,
            "interpretation_limit": (
                "Experimental applicability-distance screening only. "
                "Not predictive uncertainty, probability of correctness, "
                "prediction-error calibration, or external validation."
            ),
        }
    report_path = OUT / "promoted_applicability_distance_calibration_audit.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n========== AUDIT COMPLETE ==========")
    print("Report:", report_path)
    print("Models unchanged:", all(
        item["model_unchanged"] for item in results.values()
    ))
    print("No promoted models or source datasets were written.")
if __name__ == "__main__":
    main()
