from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "applicability_domain_v2"
RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.25
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
BASE_NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si",
    "V", "Cu", "Al", "austenitize_T", "temper_T",
]
BASE_CATEGORICAL = [
    "quench_medium",
    "steel_family",
]
VARIANTS = {
    "full_13": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
    },
    "chemistry_2": {
        "numeric": ["C", "Mn"],
        "categorical": [],
    },
    "chemistry_3": {
        "numeric": ["C", "Mn", "V"],
        "categorical": [],
    },
}
YIELD_ANOMALY_GRADES = [
    "EMK0319",
    "EMK0320",
    "EMK0321",
]
MISSING_CATEGORY = "__MATX_MISSING_CATEGORY__"
def log(message):
    print(message, flush=True)
def load_dataset():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    df = pd.read_csv(DATASET)
    required = {
        "source", "grade_id", "heat_id",
        "tensile_strength", "yield_strength",
        "elongation",
        *BASE_NUMERIC,
        *BASE_CATEGORICAL,
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )
    for column in BASE_NUMERIC + TARGETS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )
    return df
def eligible_population(df, target):
    clean = df.dropna(
        subset=[target, "source", "grade_id", "heat_id"]
    ).copy()
    excluded = 0
    if target == "yield_strength":
        ratio = (
            clean[target]
            / clean["tensile_strength"].replace(0, np.nan)
        )
        anomaly = (
            clean["grade_id"].astype(str).isin(
                YIELD_ANOMALY_GRADES
            )
            & (ratio < 0.03)
        )
        excluded = int(anomaly.sum())
        clean = clean.loc[~anomaly].copy()
    clean["source"] = clean["source"].astype(str)
    clean["grade_id"] = clean["grade_id"].astype(str)
    clean["heat_id"] = clean["heat_id"].astype(str)
    if clean.duplicated(
        subset=["source", "grade_id", "heat_id"]
    ).any():
        raise RuntimeError(
            f"Duplicate source/grade/heat identifiers for {target}."
        )
    return clean, excluded
def prepare_numeric(train, other, numeric_features):
    train_raw = train[numeric_features].apply(
        pd.to_numeric, errors="coerce"
    )
    other_raw = other[numeric_features].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = {}
    scales = {}
    ranges = {}
    train_matrix = np.zeros(
        (len(train), len(numeric_features)), dtype=float
    )
    other_matrix = np.zeros(
        (len(other), len(numeric_features)), dtype=float
    )
    for j, feature in enumerate(numeric_features):
        values = train_raw[feature].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            median = 0.0
            q01 = q25 = q75 = q99 = 0.0
            minimum = maximum = 0.0
            scale = 1.0
            all_missing = True
        else:
            median = float(np.median(valid))
            q01, q25, q75, q99 = np.quantile(
                valid, [0.01, 0.25, 0.75, 0.99]
            )
            minimum = float(np.min(valid))
            maximum = float(np.max(valid))
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(valid))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            all_missing = False
        medians[feature] = median
        scales[feature] = scale
        ranges[feature] = {
            "minimum": minimum,
            "maximum": maximum,
            "q01": float(q01),
            "q25": float(q25),
            "q75": float(q75),
            "q99": float(q99),
            "median": median,
            "robust_scale_iqr_fallback_std": scale,
            "all_training_values_missing": all_missing,
        }
        train_values = train_raw[feature].to_numpy(dtype=float)
        other_values = other_raw[feature].to_numpy(dtype=float)
        train_values = np.where(
            np.isfinite(train_values), train_values, median
        )
        other_values = np.where(
            np.isfinite(other_values), other_values, median
        )
        train_matrix[:, j] = (train_values - median) / scale
        other_matrix[:, j] = (other_values - median) / scale
    return (
        train_matrix,
        other_matrix,
        train_raw,
        other_raw,
        ranges,
    )
def prepare_categories(train, other, categorical_features):
    train_values = {}
    other_values = {}
    known_categories = {}
    for feature in categorical_features:
        tr = train[feature].astype("string")
        te = other[feature].astype("string")
        tr = tr.fillna(MISSING_CATEGORY).astype(str)
        te = te.fillna(MISSING_CATEGORY).astype(str)
        train_values[feature] = tr.to_numpy()
        other_values[feature] = te.to_numpy()
        known = sorted(
            value for value in set(train_values[feature])
            if value != MISSING_CATEGORY
        )
        known_categories[feature] = known
    return train_values, other_values, known_categories
def mixed_distance_matrix(
    numeric_a,
    numeric_b,
    categorical_a,
    categorical_b,
    numeric_count,
    categorical_count,
):
    dimensions = numeric_count + categorical_count
    if dimensions == 0:
        raise RuntimeError("No features available for distance.")
    if numeric_count:
        distance_squared = cdist(
            numeric_a,
            numeric_b,
            metric="sqeuclidean",
        )
    else:
        distance_squared = np.zeros(
            (len(categorical_a[next(iter(categorical_a))])
             if categorical_a else 0,
             len(categorical_b[next(iter(categorical_b))])
             if categorical_b else 0),
            dtype=float,
        )
    for feature in categorical_a:
        distance_squared += (
            categorical_a[feature][:, None]
            != categorical_b[feature][None, :]
        ).astype(float)
    return np.sqrt(distance_squared / dimensions)
def nearest_other_grade_distances(
    train_numeric,
    train_categories,
    train_grades,
):
    matrix = mixed_distance_matrix(
        train_numeric,
        train_numeric,
        train_categories,
        train_categories,
        train_numeric.shape[1],
        len(train_categories),
    )
    same_grade = (
        train_grades[:, None] == train_grades[None, :]
    )
    matrix[same_grade] = np.inf
    nearest_indices = np.argmin(matrix, axis=1)
    nearest_distances = matrix[
        np.arange(len(train_grades)), nearest_indices
    ]
    if not np.isfinite(nearest_distances).all():
        raise RuntimeError(
            "Could not calculate finite cross-grade training distances."
        )
    return nearest_distances, nearest_indices
def classify_record(
    distance,
    q95,
    q99,
    missing_fraction,
    available_count,
    total_features,
    outside_minmax_count,
    outside_q01q99_count,
    unseen_category_count,
):
    if available_count < int(np.ceil(total_features * 0.50)):
        return "INSUFFICIENT_EVIDENCE"
    if distance > q99 or outside_minmax_count >= 2:
        return "OUT_OF_DOMAIN"
    if (
        distance > q95
        or outside_minmax_count > 0
        or outside_q01q99_count > 0
        or unseen_category_count > 0
    ):
        return "BORDERLINE"
    return "IN_DOMAIN"
def assess_variant(
    train,
    calibration,
    test,
    target,
    held_out_source,
    variant_name,
    config,
):
    numeric_features = config["numeric"]
    categorical_features = config["categorical"]
    feature_columns = numeric_features + categorical_features
    (
        train_numeric,
        test_numeric,
        train_raw,
        test_raw,
        numeric_ranges,
    ) = prepare_numeric(
        train, test, numeric_features
    )
    (
        train_categories,
        test_categories,
        known_categories,
    ) = prepare_categories(
        train, test, categorical_features
    )
    train_grades = train["grade_id"].to_numpy(dtype=str)
    (
        _,
        calibration_numeric,
        _,
        _,
        _,
    ) = prepare_numeric(
        train, calibration, numeric_features
    )
    (
        _,
        calibration_categories,
        _,
    ) = prepare_categories(
        train, calibration, categorical_features
    )
    calibration_distance_matrix = mixed_distance_matrix(
        calibration_numeric,
        train_numeric,
        calibration_categories,
        train_categories,
        len(numeric_features),
        len(categorical_features),
    )
    calibration_nearest_distances = np.min(
        calibration_distance_matrix, axis=1
    )
    if (
        len(calibration_nearest_distances) == 0
        or not np.isfinite(calibration_nearest_distances).all()
    ):
        raise RuntimeError(
            "Invalid calibration-to-training distances."
        )
    q95, q99 = np.quantile(
        calibration_nearest_distances, [0.95, 0.99]
    )
    if not np.isfinite([q95, q99]).all():
        raise RuntimeError("Invalid training distance thresholds.")
    test_distance_matrix = mixed_distance_matrix(
        test_numeric,
        train_numeric,
        test_categories,
        train_categories,
        len(numeric_features),
        len(categorical_features),
    )
    nearest_indices = np.argmin(
        test_distance_matrix, axis=1
    )
    nearest_distances = test_distance_matrix[
        np.arange(len(test)), nearest_indices
    ]
    nearest_grades = train_grades[nearest_indices]
    all_features = feature_columns
    total_features = len(all_features)
    missing_mask = test[all_features].isna().to_numpy()
    missing_counts = missing_mask.sum(axis=1)
    available_counts = total_features - missing_counts
    missing_fractions = missing_counts / total_features
    outside_minmax = np.zeros(len(test), dtype=int)
    outside_q01q99 = np.zeros(len(test), dtype=int)
    feature_rows = []
    for feature in numeric_features:
        limits = numeric_ranges[feature]
        values = test_raw[feature].to_numpy(dtype=float)
        valid = np.isfinite(values)
        below_min = valid & (values < limits["minimum"])
        above_max = valid & (values > limits["maximum"])
        below_q01 = valid & (values < limits["q01"])
        above_q99 = valid & (values > limits["q99"])
        outside_minmax += (below_min | above_max).astype(int)
        outside_q01q99 += (below_q01 | above_q99).astype(int)
        feature_rows.append({
            "target": target,
            "held_out_source": held_out_source,
            "variant": variant_name,
            "feature": feature,
            "feature_type": "numeric",
            "training_min": limits["minimum"],
            "training_max": limits["maximum"],
            "training_q01": limits["q01"],
            "training_q99": limits["q99"],
            "training_median": limits["median"],
            "training_scale": limits[
                "robust_scale_iqr_fallback_std"
            ],
            "training_missing_fraction": float(
                train[feature].isna().mean()
            ),
            "test_missing_fraction": float(
                test[feature].isna().mean()
            ),
            "test_outside_training_minmax": int(
                (below_min | above_max).sum()
            ),
            "test_outside_training_q01q99": int(
                (below_q01 | above_q99).sum()
            ),
            "known_categories": None,
            "unseen_test_values": 0,
        })
    unseen_counts = np.zeros(len(test), dtype=int)
    for feature in categorical_features:
        known = set(known_categories[feature])
        values = test_categories[feature]
        unseen = np.array([
            value != MISSING_CATEGORY and value not in known
            for value in values
        ])
        unseen_counts += unseen.astype(int)
        feature_rows.append({
            "target": target,
            "held_out_source": held_out_source,
            "variant": variant_name,
            "feature": feature,
            "feature_type": "categorical",
            "training_min": None,
            "training_max": None,
            "training_q01": None,
            "training_q99": None,
            "training_median": None,
            "training_scale": None,
            "training_missing_fraction": float(
                train[feature].isna().mean()
            ),
            "test_missing_fraction": float(
                test[feature].isna().mean()
            ),
            "test_outside_training_minmax": 0,
            "test_outside_training_q01q99": 0,
            "known_categories": json.dumps(
                known_categories[feature]
            ),
            "unseen_test_values": int(unseen.sum()),
        })
    statuses = []
    for i in range(len(test)):
        statuses.append(
            classify_record(
                float(nearest_distances[i]),
                float(q95),
                float(q99),
                float(missing_fractions[i]),
                int(available_counts[i]),
                total_features,
                int(outside_minmax[i]),
                int(outside_q01q99[i]),
                int(unseen_counts[i]),
            )
        )
    result = pd.DataFrame({
        "target": target,
        "held_out_source": held_out_source,
        "variant": variant_name,
        "source": test["source"].to_numpy(),
        "grade_id": test["grade_id"].to_numpy(),
        "heat_id": test["heat_id"].to_numpy(),
        "nearest_training_distance": nearest_distances,
        "training_distance_q95": float(q95),
        "training_distance_q99": float(q99),
        "nearest_training_grade": nearest_grades,
        "missing_feature_count": missing_counts,
        "total_feature_count": total_features,
        "missing_feature_fraction": missing_fractions,
        "outside_training_minmax_count": outside_minmax,
        "outside_training_q01q99_count": outside_q01q99,
        "unseen_category_count": unseen_counts,
        "applicability_status": statuses,
    })
    if not np.isfinite(
        result["nearest_training_distance"].to_numpy()
    ).all():
        raise RuntimeError("Non-finite applicability distances.")
    if result["heat_id"].duplicated().any():
        raise RuntimeError("Duplicate heat IDs in applicability output.")
    summary = {
        "target": target,
        "held_out_source": held_out_source,
        "variant": variant_name,
        "features": feature_columns,
        "train_records": len(train),
        "test_records": len(test),
        "train_grades": int(train["grade_id"].nunique()),
        "test_grades": int(test["grade_id"].nunique()),
        "calibration_reference_rows": len(calibration_nearest_distances),
        "training_distance_q95": float(q95),
        "training_distance_q99": float(q99),
        "test_distance_median": float(
            np.median(nearest_distances)
        ),
        "test_distance_q95": float(
            np.quantile(nearest_distances, 0.95)
        ),
        "mean_missing_feature_fraction": float(
            missing_fractions.mean()
        ),
        "records_with_unseen_categories": int(
            (unseen_counts > 0).sum()
        ),
        "records_with_numeric_range_violations": int(
            (outside_minmax > 0).sum()
        ),
        "status_counts": {
            str(k): int(v)
            for k, v in result[
                "applicability_status"
            ].value_counts().items()
        },
    }
    return result, summary, feature_rows
def main():
    if OUT.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {OUT}"
        )
    OUT.mkdir(parents=True)
    log("MAT-X APPLICABILITY-DOMAIN ASSESSMENT v1")
    log(f"Dataset: {DATASET}")
    log(f"Output: {OUT}")
    df = load_dataset()
    all_records = []
    fold_summaries = []
    feature_summaries = []
    split_audits = []
    failures = []
    for target in TARGETS:
        clean, excluded = eligible_population(df, target)
        log(
            f"\nTARGET: {target} | eligible={len(clean)} "
            f"| grades={clean['grade_id'].nunique()} "
            f"| yield_anomalies_excluded={excluded}"
        )
        for held_out_source in SOURCES:
            test = clean.loc[
                clean["source"] == held_out_source
            ].copy()
            train_cal = clean.loc[
                clean["source"] != held_out_source
            ].copy()
            if test.empty or train_cal.empty:
                log(
                    f"SKIP {target}/{held_out_source}: "
                    "missing test or training population."
                )
                continue
            if train_cal["grade_id"].nunique() < 5:
                log(
                    f"SKIP {target}/{held_out_source}: "
                    "fewer than five training grades."
                )
                continue
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=CALIBRATION_FRACTION,
                random_state=(
                    RANDOM_STATE + SOURCES.index(held_out_source)
                ),
            )
            train_idx, calibration_idx = next(
                splitter.split(
                    np.zeros((len(train_cal), 1)),
                    train_cal[target].to_numpy(dtype=float),
                    groups=train_cal["grade_id"].to_numpy(),
                )
            )
            train = train_cal.iloc[train_idx].copy()
            calibration = train_cal.iloc[calibration_idx].copy()
            train_grades = set(train["grade_id"])
            calibration_grades = set(calibration["grade_id"])
            test_grades = set(test["grade_id"])
            overlaps = {
                "train_calibration": len(
                    train_grades & calibration_grades
                ),
                "train_test": len(train_grades & test_grades),
                "calibration_test": len(
                    calibration_grades & test_grades
                ),
            }
            if any(overlaps.values()):
                raise RuntimeError(
                    f"Grade leakage detected: {target}/{held_out_source}"
                )
            split_audits.append({
                "target": target,
                "held_out_source": held_out_source,
                "train_records": len(train),
                "calibration_records": len(calibration),
                "test_records": len(test),
                "train_grades": len(train_grades),
                "calibration_grades": len(calibration_grades),
                "test_grades": len(test_grades),
                **{
                    f"{key}_overlap": value
                    for key, value in overlaps.items()
                },
            })
            log(
                f"\nFOLD {target} | {held_out_source} | "
                f"train={len(train)} | test={len(test)}"
            )
            for variant_name, config in VARIANTS.items():
                try:
                    result, summary, feature_rows = assess_variant(
                        train=train,
                        calibration=calibration,
                        test=test,
                        target=target,
                        held_out_source=held_out_source,
                        variant_name=variant_name,
                        config=config,
                    )
                    all_records.append(result)
                    fold_summaries.append(summary)
                    feature_summaries.extend(feature_rows)
                    log(
                        f"  {variant_name:12s} | "
                        f"distance q95={summary['training_distance_q95']:.4f} "
                        f"| statuses={summary['status_counts']}"
                    )
                except Exception as exc:
                    failures.append({
                        "target": target,
                        "held_out_source": held_out_source,
                        "variant": variant_name,
                        "error": repr(exc),
                    })
                    log(
                        f"  FAILURE {variant_name}: {exc}"
                    )
    if not all_records:
        raise RuntimeError("No applicability assessments completed.")
    records = pd.concat(all_records, ignore_index=True)
    summaries = pd.DataFrame(fold_summaries)
    feature_table = pd.DataFrame(feature_summaries)
    split_table = pd.DataFrame(split_audits)
    failure_table = pd.DataFrame(failures)
    records_path = OUT / "record_applicability.csv"
    summaries_path = OUT / "fold_applicability_summary.csv"
    features_path = OUT / "feature_domain_summary.csv"
    splits_path = OUT / "split_integrity_audit.csv"
    failures_path = OUT / "execution_failures.csv"
    records.to_csv(records_path, index=False)
    summaries.to_csv(summaries_path, index=False)
    feature_table.to_csv(features_path, index=False)
    split_table.to_csv(splits_path, index=False)
    failure_table.to_csv(failures_path, index=False)
    status_totals = {
        str(k): int(v)
        for k, v in records[
            "applicability_status"
        ].value_counts().items()
    }
    summary_json = {
        "experiment": "MAT-X Applicability-Domain Assessment v1",
        "dataset": str(DATASET),
        "method": {
            "numeric_scaling": (
                "Training-only median imputation and robust IQR scaling; "
                "standard deviation fallback; unit fallback for constants."
            ),
            "categorical_distance": (
                "Per-feature mismatch distance; missing categories treated "
                "as an explicit category."
            ),
            "reference_distance": (
                "Each calibration record's nearest training record."
            ),
            "test_distance": (
                "Each held-out record's nearest training record."
            ),
            "thresholds": (
                "95th and 99th percentiles of calibration-to-training "
                "nearest-neighbour distances."
            ),
            "range_checks": (
                "Training-only min/max and 1st/99th percentile ranges."
            ),
            "status_rules": {
                "INSUFFICIENT_EVIDENCE": (
                    "Fewer than 50% of model input features observed."
                ),
                "OUT_OF_DOMAIN": (
                    "Distance above training 99th percentile or at least "
                    "two numeric features outside training min/max."
                ),
                "BORDERLINE": (
                    "Distance above 95th percentile, any numeric range "
                    "violation, or any unseen category."
                ),
                "IN_DOMAIN": (
                    "No triggered screening condition."
                ),
            },
        },
        "targets": TARGETS,
        "variants": {
            name: {
                "numeric": config["numeric"],
                "categorical": config["categorical"],
            }
            for name, config in VARIANTS.items()
        },
        "successful_assessments": len(fold_summaries),
        "failed_assessments": len(failures),
        "record_rows": len(records),
        "split_integrity_passed": bool(
            len(split_table) > 0
            and (
                split_table[
                    [
                        "train_calibration_overlap",
                        "train_test_overlap",
                        "calibration_test_overlap",
                    ]
                ].to_numpy() == 0
            ).all()
        ),
        "status_totals": status_totals,
        "limitations": [
            "Applicability status is an empirical screening result, not a "
            "probability of prediction correctness.",
            "Distance thresholds are specific to each target, source fold, "
            "and feature variant.",
            "The nearest-neighbour reference uses training records and "
            "does not establish engineering validity.",
            "Median imputation is used for distance computation; missing "
            "evidence is separately reported.",
            "Feature ranges are descriptive and do not encode physical "
            "or metallurgical feasibility.",
            "Source-held-out results measure transfer to different "
            "dataset populations.",
            "NIMS has only four independent grades in this dataset; "
            "its grade-level conclusions are unstable.",
            "No model was retrained, promoted, or modified.",
            "No raw dataset was modified.",
        ],
        "outputs": {
            "records": str(records_path),
            "fold_summary": str(summaries_path),
            "feature_summary": str(features_path),
            "split_audit": str(splits_path),
            "failures": str(failures_path),
        },
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )
    log("\n" + "=" * 72)
    log("APPLICABILITY-DOMAIN ASSESSMENT COMPLETE")
    log("=" * 72)
    log(json.dumps(summary_json, indent=2))
    log("\nOutput files:")
    for path in [
        summary_path,
        records_path,
        summaries_path,
        features_path,
        splits_path,
        failures_path,
    ]:
        log(str(path))
if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*",
    )
    main()
