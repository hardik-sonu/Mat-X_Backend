from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupShuffleSplit
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "applicability_domain_v3_retry"
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
        "source",
        "grade_id",
        "heat_id",
        *TARGETS,
        *BASE_NUMERIC,
        *BASE_CATEGORICAL,
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing dataset columns: {sorted(missing)}"
        )
    for column in BASE_NUMERIC + TARGETS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )
    df["source"] = df["source"].astype(str)
    df["grade_id"] = df["grade_id"].astype(str)
    df["heat_id"] = df["heat_id"].astype(str)
    return df
def eligible_population(df, target):
    clean = df.dropna(
        subset=[
            target,
            "source",
            "grade_id",
            "heat_id",
        ]
    ).copy()
    excluded = 0
    if target == "yield_strength":
        ratio = (
            clean[target]
            / clean["tensile_strength"].replace(
                0, np.nan
            )
        )
        anomaly = (
            clean["grade_id"].isin(
                YIELD_ANOMALY_GRADES
            )
            & (ratio < 0.03)
        )
        excluded = int(anomaly.sum())
        clean = clean.loc[~anomaly].copy()
    return clean, excluded
def finite_sample_threshold(values, confidence):
    """
    Finite-sample order-statistic threshold.
    If the requested conformal rank exceeds the available
    calibration grades, the threshold is not estimable.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return None, None
    rank = int(math.ceil((n + 1) * confidence))
    if rank > n:
        return None, rank
    ordered = np.sort(values)
    return float(ordered[rank - 1]), rank
def prepare_feature_reference(train, numeric_features,
                               categorical_features):
    """
    Feature eligibility and preprocessing use training data only.
    """
    numeric_reference = {}
    categorical_reference = {}
    eligible_numeric = []
    eligible_categorical = []
    feature_rows = []
    for feature in numeric_features:
        values = pd.to_numeric(
            train[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            status = "ALL_MISSING_IN_TRAIN"
            reference = None
        else:
            median = float(np.median(valid))
            q01 = float(np.quantile(valid, 0.01))
            q25 = float(np.quantile(valid, 0.25))
            q75 = float(np.quantile(valid, 0.75))
            q99 = float(np.quantile(valid, 0.99))
            minimum = float(np.min(valid))
            maximum = float(np.max(valid))
            iqr = q75 - q25
            std = float(np.std(valid))
            tolerance = (
                np.finfo(float).eps
                * max(1.0, abs(median))
                * 100
            )
            if np.unique(valid).size <= 1 or std <= tolerance:
                status = "CONSTANT_IN_TRAIN"
                reference = {
                    "median": median,
                    "minimum": minimum,
                    "maximum": maximum,
                    "q01": q01,
                    "q99": q99,
                }
            else:
                scale = iqr
                if not np.isfinite(scale) or scale <= tolerance:
                    scale = std
                if not np.isfinite(scale) or scale <= tolerance:
                    status = "CONSTANT_IN_TRAIN"
                    reference = {
                        "median": median,
                        "minimum": minimum,
                        "maximum": maximum,
                        "q01": q01,
                        "q99": q99,
                    }
                else:
                    status = "VARIABLE_IN_TRAIN"
                    reference = {
                        "median": median,
                        "scale": float(scale),
                        "minimum": minimum,
                        "maximum": maximum,
                        "q01": q01,
                        "q99": q99,
                    }
                    eligible_numeric.append(feature)
                    numeric_reference[feature] = reference
        feature_rows.append({
            "feature": feature,
            "feature_type": "numeric",
            "training_status": status,
            "training_valid_records": int(len(valid)),
            "training_missing_fraction": float(
                train[feature].isna().mean()
            ),
            "training_unique_values": int(
                len(np.unique(valid))
            ),
            "training_min": (
                reference.get("minimum")
                if reference else None
            ),
            "training_max": (
                reference.get("maximum")
                if reference else None
            ),
            "training_q01": (
                reference.get("q01")
                if reference else None
            ),
            "training_q99": (
                reference.get("q99")
                if reference else None
            ),
            "training_median": (
                reference.get("median")
                if reference else None
            ),
            "training_scale": (
                reference.get("scale")
                if reference else None
            ),
        })
    for feature in categorical_features:
        values = (
            train[feature]
            .astype("string")
            .fillna(MISSING_CATEGORY)
            .astype(str)
        )
        observed = values[
            values != MISSING_CATEGORY
        ]
        known = sorted(set(observed))
        if len(known) == 0:
            status = "ALL_MISSING_IN_TRAIN"
        elif len(known) == 1:
            status = "ONE_CATEGORY_IN_TRAIN"
        else:
            status = "VARIABLE_IN_TRAIN"
            eligible_categorical.append(feature)
            categorical_reference[feature] = set(known)
        feature_rows.append({
            "feature": feature,
            "feature_type": "categorical",
            "training_status": status,
            "training_valid_records": int(len(observed)),
            "training_missing_fraction": float(
                train[feature].isna().mean()
            ),
            "training_unique_values": int(len(known)),
            "training_min": None,
            "training_max": None,
            "training_q01": None,
            "training_q99": None,
            "training_median": None,
            "training_scale": None,
            "known_categories": json.dumps(known),
        })
    return (
        eligible_numeric,
        eligible_categorical,
        numeric_reference,
        categorical_reference,
        feature_rows,
    )
def transform_numeric(frame, features, reference):
    if not features:
        return np.empty((len(frame), 0), dtype=float)
    matrix = np.zeros(
        (len(frame), len(features)),
        dtype=float,
    )
    for j, feature in enumerate(features):
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        ).to_numpy(dtype=float, copy=True)
        valid = np.isfinite(values)
        values[~valid] = reference[feature]["median"]
        matrix[:, j] = (
            values - reference[feature]["median"]
        ) / reference[feature]["scale"]
    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Non-finite transformed numeric features."
        )
    return matrix
def transform_categories(frame, features):
    result = {}
    for feature in features:
        result[feature] = (
            frame[feature]
            .astype("string")
            .fillna(MISSING_CATEGORY)
            .astype(str)
            .to_numpy()
        )
    return result
def mixed_distance_matrix(
    numeric_a,
    numeric_b,
    categorical_a,
    categorical_b,
):
    dimensions = (
        numeric_a.shape[1]
        + len(categorical_a)
    )
    if dimensions == 0:
        raise RuntimeError(
            "No variable training features available."
        )
    if numeric_a.shape[1]:
        squared = cdist(
            numeric_a,
            numeric_b,
            metric="sqeuclidean",
        )
    else:
        squared = np.zeros(
            (len(next(iter(categorical_a.values()))),
             len(next(iter(categorical_b.values())))),
            dtype=float,
        )
    for feature in categorical_a:
        squared += (
            categorical_a[feature][:, None]
            != categorical_b[feature][None, :]
        ).astype(float)
    return np.sqrt(squared / dimensions)
def calculate_unsupported_evidence(
    test,
    numeric_features,
    categorical_features,
    numeric_reference,
    categorical_reference,
):
    """
    Reports deviations from training-constant or unsupported
    features separately from the distance calculation.
    """
    n = len(test)
    unsupported_counts = np.zeros(n, dtype=int)
    constant_deviation_counts = np.zeros(n, dtype=int)
    unseen_counts = np.zeros(n, dtype=int)
    outside_minmax = np.zeros(n, dtype=int)
    outside_q01q99 = np.zeros(n, dtype=int)
    all_missing_training_values = np.zeros(n, dtype=int)
    details = []
    for feature in numeric_features:
        values = pd.to_numeric(
            test[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        train_reference = numeric_reference.get(feature)
        if train_reference is None:
            count = int(valid.sum())
            unsupported_counts += valid.astype(int)
            all_missing_training_values += valid.astype(int)
            details.append({
                "feature": feature,
                "feature_type": "numeric",
                "training_status": "ALL_MISSING_IN_TRAIN",
                "test_valid_records": count,
                "test_missing_fraction": float(
                    (~valid).mean()
                ),
                "unsupported_test_records": count,
                "constant_deviation_records": 0,
                "unseen_category_records": 0,
                "outside_training_minmax": 0,
                "outside_training_q01q99": 0,
            })
            continue
        # Features not eligible for distance are audited separately.
        if feature not in numeric_reference:
            # The caller supplies the complete feature list below;
            # this branch is retained as a defensive guard.
            continue
        minimum = train_reference["minimum"]
        maximum = train_reference["maximum"]
        q01 = train_reference["q01"]
        q99 = train_reference["q99"]
        below_min = valid & (values < minimum)
        above_max = valid & (values > maximum)
        below_q01 = valid & (values < q01)
        above_q99 = valid & (values > q99)
        outside_minmax += (
            below_min | above_max
        ).astype(int)
        outside_q01q99 += (
            below_q01 | above_q99
        ).astype(int)
        details.append({
            "feature": feature,
            "feature_type": "numeric",
            "training_status": "VARIABLE_IN_TRAIN",
            "test_valid_records": int(valid.sum()),
            "test_missing_fraction": float((~valid).mean()),
            "unsupported_test_records": 0,
            "constant_deviation_records": 0,
            "unseen_category_records": 0,
            "outside_training_minmax": int(
                (below_min | above_max).sum()
            ),
            "outside_training_q01q99": int(
                (below_q01 | above_q99).sum()
            ),
        })
    for feature in BASE_NUMERIC:
        if feature not in numeric_features:
            continue
        if feature in numeric_reference:
            continue
        values = pd.to_numeric(
            test[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        train_values = pd.to_numeric(
            # The reference for constant features is supplied
            # separately by the assessment function.
            pd.Series([], dtype=float),
            errors="coerce",
        )
        # Constant feature handling is completed by the caller.
    for feature in categorical_features:
        values = (
            test[feature]
            .astype("string")
            .fillna(MISSING_CATEGORY)
            .astype(str)
            .to_numpy()
        )
        known = categorical_reference.get(feature)
        if known is None:
            observed = values != MISSING_CATEGORY
            count = int(observed.sum())
            unsupported_counts += observed.astype(int)
            details.append({
                "feature": feature,
                "feature_type": "categorical",
                "training_status": "ALL_MISSING_IN_TRAIN",
                "test_valid_records": count,
                "test_missing_fraction": float(
                    (~observed).mean()
                ),
                "unsupported_test_records": count,
                "constant_deviation_records": 0,
                "unseen_category_records": 0,
                "outside_training_minmax": 0,
                "outside_training_q01q99": 0,
            })
            continue
        unseen = np.array([
            value != MISSING_CATEGORY and value not in known
            for value in values
        ])
        unseen_counts += unseen.astype(int)
        details.append({
            "feature": feature,
            "feature_type": "categorical",
            "training_status": "VARIABLE_IN_TRAIN",
            "test_valid_records": int(
                (values != MISSING_CATEGORY).sum()
            ),
            "test_missing_fraction": float(
                (values == MISSING_CATEGORY).mean()
            ),
            "unsupported_test_records": 0,
            "constant_deviation_records": 0,
            "unseen_category_records": int(unseen.sum()),
            "outside_training_minmax": 0,
            "outside_training_q01q99": 0,
        })
    return {
        "unsupported_counts": unsupported_counts,
        "constant_deviation_counts": constant_deviation_counts,
        "unseen_counts": unseen_counts,
        "outside_minmax": outside_minmax,
        "outside_q01q99": outside_q01q99,
        "all_missing_training_values": all_missing_training_values,
        "details": details,
    }
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
    all_features = numeric_features + categorical_features
    (
        eligible_numeric,
        eligible_categorical,
        numeric_reference,
        categorical_reference,
        feature_rows,
    ) = prepare_feature_reference(
        train,
        numeric_features,
        categorical_features,
    )
    eligible_count = (
        len(eligible_numeric)
        + len(eligible_categorical)
    )
    total_features = len(all_features)
    # Add fold and variant context to the feature table.
    for row in feature_rows:
        row.update({
            "target": target,
            "held_out_source": held_out_source,
            "variant": variant_name,
            "eligible_for_distance": (
                row["feature"] in eligible_numeric
                or row["feature"] in eligible_categorical
            ),
        })
    if eligible_count == 0:
        raise RuntimeError(
            "No eligible features with training variation."
        )
    train_numeric = transform_numeric(
        train,
        eligible_numeric,
        numeric_reference,
    )
    calibration_numeric = transform_numeric(
        calibration,
        eligible_numeric,
        numeric_reference,
    )
    test_numeric = transform_numeric(
        test,
        eligible_numeric,
        numeric_reference,
    )
    train_categories = transform_categories(
        train,
        eligible_categorical,
    )
    calibration_categories = transform_categories(
        calibration,
        eligible_categorical,
    )
    test_categories = transform_categories(
        test,
        eligible_categorical,
    )
    # Calibration record -> nearest training record.
    calibration_matrix = mixed_distance_matrix(
        calibration_numeric,
        train_numeric,
        calibration_categories,
        train_categories,
    )
    calibration_nearest = np.min(
        calibration_matrix,
        axis=1,
    )
    if not np.isfinite(calibration_nearest).all():
        raise RuntimeError(
            "Invalid calibration distances."
        )
    # Grade-grouped calibration: maximum distance per grade.
    calibration_reference = pd.DataFrame({
        "grade_id": calibration["grade_id"].to_numpy(),
        "distance": calibration_nearest,
    })
    grade_maxima = (
        calibration_reference.groupby("grade_id")["distance"]
        .max()
        .to_numpy(dtype=float)
    )
    q90, rank90 = finite_sample_threshold(
        grade_maxima, 0.90
    )
    q95, rank95 = finite_sample_threshold(
        grade_maxima, 0.95
    )
    test_matrix = mixed_distance_matrix(
        test_numeric,
        train_numeric,
        test_categories,
        train_categories,
    )
    nearest_indices = np.argmin(
        test_matrix,
        axis=1,
    )
    nearest_distances = test_matrix[
        np.arange(len(test)),
        nearest_indices,
    ]
    train_grades = train["grade_id"].to_numpy(dtype=str)
    nearest_grades = train_grades[nearest_indices]
    # Feature evidence availability.
    all_missing = test[all_features].isna().to_numpy()
    missing_counts = all_missing.sum(axis=1)
    missing_fraction = (
        missing_counts / total_features
    )
    eligible_columns = (
        eligible_numeric + eligible_categorical
    )
    eligible_missing = (
        test[eligible_columns].isna().to_numpy()
    )
    eligible_missing_counts = eligible_missing.sum(axis=1)
    eligible_observed_counts = (
        eligible_count - eligible_missing_counts
    )
    evidence_fraction = (
        eligible_observed_counts / eligible_count
    )
    # Range and category checks on eligible features.
    outside_minmax = np.zeros(len(test), dtype=int)
    outside_q01q99 = np.zeros(len(test), dtype=int)
    unseen_counts = np.zeros(len(test), dtype=int)
    for feature in eligible_numeric:
        values = pd.to_numeric(
            test[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(values)
        reference = numeric_reference[feature]
        outside_minmax += (
            valid
            & (
                (values < reference["minimum"])
                | (values > reference["maximum"])
            )
        ).astype(int)
        outside_q01q99 += (
            valid
            & (
                (values < reference["q01"])
                | (values > reference["q99"])
            )
        ).astype(int)
    for feature in eligible_categorical:
        values = test_categories[feature]
        known = categorical_reference[feature]
        unseen = np.array([
            value != MISSING_CATEGORY and value not in known
            for value in values
        ])
        unseen_counts += unseen.astype(int)
    # Unsupported features and constant-feature deviations.
    unsupported_counts = np.zeros(len(test), dtype=int)
    constant_deviation_counts = np.zeros(len(test), dtype=int)
    for feature in numeric_features:
        if feature in eligible_numeric:
            continue
        train_values = pd.to_numeric(
            train[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        test_values = pd.to_numeric(
            test[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        train_valid = train_values[np.isfinite(train_values)]
        test_valid = np.isfinite(test_values)
        if len(train_valid) == 0:
            unsupported_counts += test_valid.astype(int)
            feature_rows.append({
                "target": target,
                "held_out_source": held_out_source,
                "variant": variant_name,
                "feature": feature,
                "feature_type": "numeric",
                "training_status": "ALL_MISSING_IN_TRAIN",
                "eligible_for_distance": False,
                "training_valid_records": 0,
                "training_missing_fraction": 1.0,
                "training_unique_values": 0,
                "test_valid_records": int(test_valid.sum()),
                "test_missing_fraction": float(
                    (~test_valid).mean()
                ),
                "unsupported_test_records": int(test_valid.sum()),
                "constant_deviation_records": 0,
            })
        else:
            reference_value = float(train_valid[0])
            deviations = (
                test_valid
                & ~np.isclose(
                    test_values,
                    reference_value,
                    rtol=1e-9,
                    atol=1e-12,
                )
            )
            constant_deviation_counts += deviations.astype(int)
            feature_rows.append({
                "target": target,
                "held_out_source": held_out_source,
                "variant": variant_name,
                "feature": feature,
                "feature_type": "numeric",
                "training_status": "CONSTANT_IN_TRAIN",
                "eligible_for_distance": False,
                "training_valid_records": int(len(train_valid)),
                "training_missing_fraction": float(
                    train[feature].isna().mean()
                ),
                "training_unique_values": int(
                    len(np.unique(train_valid))
                ),
                "constant_training_value": reference_value,
                "test_valid_records": int(test_valid.sum()),
                "test_missing_fraction": float(
                    (~test_valid).mean()
                ),
                "unsupported_test_records": 0,
                "constant_deviation_records": int(
                    deviations.sum()
                ),
            })
    for feature in categorical_features:
        if feature in eligible_categorical:
            continue
        train_values = (
            train[feature]
            .astype("string")
            .dropna()
            .astype(str)
        )
        test_values = (
            test[feature]
            .astype("string")
            .fillna(MISSING_CATEGORY)
            .astype(str)
            .to_numpy()
        )
        if len(train_values) == 0:
            observed = test_values != MISSING_CATEGORY
            unsupported_counts += observed.astype(int)
            status = "ALL_MISSING_IN_TRAIN"
            deviation = np.zeros(len(test), dtype=bool)
            reference_value = None
        else:
            reference_value = str(train_values.iloc[0])
            observed = test_values != MISSING_CATEGORY
            deviation = (
                observed
                & (test_values != reference_value)
            )
            constant_deviation_counts += deviation.astype(int)
            status = "ONE_CATEGORY_IN_TRAIN"
        feature_rows.append({
            "target": target,
            "held_out_source": held_out_source,
            "variant": variant_name,
            "feature": feature,
            "feature_type": "categorical",
            "training_status": status,
            "eligible_for_distance": False,
            "training_valid_records": int(len(train_values)),
            "training_missing_fraction": float(
                train[feature].isna().mean()
            ),
            "training_unique_values": int(
                train_values.nunique()
            ),
            "constant_training_value": reference_value,
            "test_valid_records": int(observed.sum()),
            "test_missing_fraction": float((~observed).mean()),
            "unsupported_test_records": int(
                observed.sum()
            ) if len(train_values) == 0 else 0,
            "constant_deviation_records": int(
                deviation.sum()
            ),
        })
    if not np.isfinite(nearest_distances).all():
        raise RuntimeError(
            "Non-finite test applicability distances."
        )
    # Status assignment.
    statuses = []
    minimum_observed = int(
        math.ceil(eligible_count * 0.50)
    )
    for i in range(len(test)):
        if q90 is None:
            status = "UNASSESSABLE_DOMAIN"
        elif eligible_observed_counts[i] < minimum_observed:
            status = "INSUFFICIENT_EVIDENCE"
        elif q95 is not None and nearest_distances[i] > q95:
            status = "OUT_OF_DOMAIN"
        elif (
            nearest_distances[i] > q90
            or outside_minmax[i] > 0
            or outside_q01q99[i] > 0
            or unseen_counts[i] > 0
            or constant_deviation_counts[i] > 0
            or unsupported_counts[i] > 0
        ):
            status = "BORDERLINE"
        else:
            status = "IN_DOMAIN"
        statuses.append(status)
    result = pd.DataFrame({
        "target": target,
        "held_out_source": held_out_source,
        "variant": variant_name,
        "source": test["source"].to_numpy(),
        "grade_id": test["grade_id"].to_numpy(),
        "heat_id": test["heat_id"].to_numpy(),
        "nearest_training_distance": nearest_distances,
        "nearest_training_grade": nearest_grades,
        "calibration_grade_count": len(grade_maxima),
        "calibration_q90": q90,
        "calibration_q95": q95,
        "eligible_feature_count": eligible_count,
        "eligible_features_observed": eligible_observed_counts,
        "eligible_feature_evidence_fraction": evidence_fraction,
        "missing_feature_count": missing_counts,
        "total_feature_count": total_features,
        "missing_feature_fraction": missing_fraction,
        "outside_training_minmax_count": outside_minmax,
        "outside_training_q01q99_count": outside_q01q99,
        "unseen_category_count": unseen_counts,
        "constant_reference_deviation_count": constant_deviation_counts,
        "unsupported_feature_value_count": unsupported_counts,
        "applicability_status": statuses,
    })
    if result["heat_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate heat IDs in held-out applicability results."
        )
    summary = {
        "target": target,
        "held_out_source": held_out_source,
        "variant": variant_name,
        "features_requested": total_features,
        "eligible_numeric_features": eligible_numeric,
        "eligible_categorical_features": eligible_categorical,
        "eligible_feature_count": eligible_count,
        "train_records": len(train),
        "calibration_records": len(calibration),
        "test_records": len(test),
        "train_grades": int(train["grade_id"].nunique()),
        "calibration_grades": int(calibration["grade_id"].nunique()),
        "test_grades": int(test["grade_id"].nunique()),
        "calibration_grade_count": int(len(grade_maxima)),
        "calibration_q90": q90,
        "calibration_q90_rank": rank90,
        "calibration_q95": q95,
        "calibration_q95_rank": rank95,
        "test_distance_median": float(
            np.median(nearest_distances)
        ),
        "test_distance_q95": float(
            np.quantile(nearest_distances, 0.95)
        ),
        "mean_missing_feature_fraction": float(
            missing_fraction.mean()
        ),
        "mean_eligible_evidence_fraction": float(
            evidence_fraction.mean()
        ),
        "records_with_constant_deviations": int(
            (constant_deviation_counts > 0).sum()
        ),
        "records_with_unsupported_feature_values": int(
            (unsupported_counts > 0).sum()
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
            "Refusing to overwrite existing output: "
            f"{OUT}"
        )
    OUT.mkdir(parents=True)
    log("MAT-X APPLICABILITY-DOMAIN ASSESSMENT V3")
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
            f"\nTARGET: {target} | "
            f"eligible={len(clean)} | "
            f"grades={clean['grade_id'].nunique()} | "
            f"yield_anomalies_excluded={excluded}"
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
                    RANDOM_STATE
                    + SOURCES.index(held_out_source)
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
            calibration = train_cal.iloc[
                calibration_idx
            ].copy()
            train_grades = set(train["grade_id"])
            calibration_grades = set(
                calibration["grade_id"]
            )
            test_grades = set(test["grade_id"])
            overlaps = {
                "train_calibration": len(
                    train_grades & calibration_grades
                ),
                "train_test": len(
                    train_grades & test_grades
                ),
                "calibration_test": len(
                    calibration_grades & test_grades
                ),
            }
            if any(overlaps.values()):
                raise RuntimeError(
                    "Grade leakage detected: "
                    f"{target}/{held_out_source}"
                )
            split_audits.append({
                "target": target,
                "held_out_source": held_out_source,
                "train_records": len(train),
                "calibration_records": len(calibration),
                "test_records": len(test),
                "train_grades": len(train_grades),
                "calibration_grades": len(
                    calibration_grades
                ),
                "test_grades": len(test_grades),
                **{
                    f"{key}_overlap": value
                    for key, value in overlaps.items()
                },
            })
            log(
                f"\nFOLD {target} | {held_out_source} | "
                f"train={len(train)} | "
                f"calibration={len(calibration)} | "
                f"test={len(test)}"
            )
            for variant_name, config in VARIANTS.items():
                try:
                    result, summary, feature_rows = (
                        assess_variant(
                            train=train,
                            calibration=calibration,
                            test=test,
                            target=target,
                            held_out_source=held_out_source,
                            variant_name=variant_name,
                            config=config,
                        )
                    )
                    all_records.append(result)
                    fold_summaries.append(summary)
                    feature_summaries.extend(
                        feature_rows
                    )
                    log(
                        f"  {variant_name:12s} | "
                        f"eligible={summary['eligible_feature_count']} | "
                        f"q90={summary['calibration_q90']} | "
                        f"q95={summary['calibration_q95']} | "
                        f"statuses={summary['status_counts']}"
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
        raise RuntimeError(
            "No applicability assessments completed."
        )
    records = pd.concat(
        all_records,
        ignore_index=True,
    )
    summaries = pd.DataFrame(fold_summaries)
    feature_table = pd.DataFrame(feature_summaries)
    split_table = pd.DataFrame(split_audits)
    failure_table = pd.DataFrame(failures)
    # Every assessed heat must occur exactly once per fold/variant.
    duplicate_keys = records.duplicated(
        subset=[
            "target",
            "held_out_source",
            "variant",
            "heat_id",
        ]
    )
    if duplicate_keys.any():
        raise RuntimeError(
            "Duplicate target/fold/variant/heat combinations."
        )
    # Verify split separation.
    overlap_columns = [
        "train_calibration_overlap",
        "train_test_overlap",
        "calibration_test_overlap",
    ]
    split_integrity_passed = bool(
        len(split_table) > 0
        and (
            split_table[overlap_columns].to_numpy() == 0
        ).all()
    )
    if not split_integrity_passed:
        raise RuntimeError(
            "Split integrity audit failed."
        )
    records_path = OUT / "record_applicability.csv"
    summaries_path = OUT / "fold_applicability_summary.csv"
    features_path = OUT / "feature_domain_summary.csv"
    splits_path = OUT / "split_integrity_audit.csv"
    failures_path = OUT / "execution_failures.csv"
    statuses_path = OUT / "status_counts.csv"
    calibration_path = OUT / "calibration_reference_summary.csv"
    records.to_csv(records_path, index=False)
    summaries.to_csv(summaries_path, index=False)
    feature_table.to_csv(features_path, index=False)
    split_table.to_csv(splits_path, index=False)
    failure_table.to_csv(failures_path, index=False)
    status_table = (
        records.groupby(
            [
                "target",
                "held_out_source",
                "variant",
                "applicability_status",
            ]
        )
        .size()
        .reset_index(name="record_count")
    )
    status_table.to_csv(
        statuses_path,
        index=False,
    )
    calibration_table = summaries[
        [
            "target",
            "held_out_source",
            "variant",
            "calibration_grades",
            "calibration_grade_count",
            "calibration_q90",
            "calibration_q90_rank",
            "calibration_q95",
            "calibration_q95_rank",
            "eligible_feature_count",
        ]
    ].copy()
    calibration_table.to_csv(
        calibration_path,
        index=False,
    )
    status_totals = {
        str(k): int(v)
        for k, v in records[
            "applicability_status"
        ].value_counts().items()
    }
    summary_json = {
        "experiment": (
            "MAT-X Applicability-Domain Assessment v3"
        ),
        "dataset": str(DATASET),
        "method": {
            "folds": (
                "Source-held-out folds with grade-grouped "
                "training/calibration split; same split procedure "
                "and random states as v2."
            ),
            "feature_eligibility": (
                "Only features with observed training variation "
                "participate in distance calculations."
            ),
            "numeric_scaling": (
                "Training-only median imputation and robust IQR "
                "scaling, with standard-deviation fallback."
            ),
            "categorical_distance": (
                "Mismatch distance for categorical features with "
                "at least two observed training categories."
            ),
            "constant_features": (
                "Excluded from distance; held-out deviations "
                "reported separately."
            ),
            "all_missing_training_features": (
                "Excluded from distance; held-out observed values "
                "reported as unsupported evidence."
            ),
            "reference_distance": (
                "Calibration record to nearest training record."
            ),
            "group_calibration": (
                "Maximum calibration-record distance per calibration "
                "grade."
            ),
            "thresholds": (
                "Finite-sample order statistics at 90% and 95% "
                "confidence using calibration grade maxima. A "
                "threshold is not estimable if its required rank "
                "exceeds the available calibration grades."
            ),
            "range_checks": (
                "Training-only min/max and 1st/99th percentile "
                "ranges for variable numeric features."
            ),
            "distance_normalization": (
                "Square root of mean squared numeric/categorical "
                "feature distances over eligible dimensions."
            ),
            "status_rules": {
                "UNASSESSABLE_DOMAIN": (
                    "No usable calibration q90 reference."
                ),
                "INSUFFICIENT_EVIDENCE": (
                    "Fewer than half of eligible features observed."
                ),
                "OUT_OF_DOMAIN": (
                    "Nearest-training distance exceeds the "
                    "calibration grade-group q95 threshold."
                ),
                "BORDERLINE": (
                    "Distance exceeds q90, or feature range, "
                    "category, constant-reference, or unsupported "
                    "evidence deviations are present."
                ),
                "IN_DOMAIN": (
                    "Sufficient evidence, distance within q90, "
                    "and no detected feature deviations."
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
        "split_integrity_passed": split_integrity_passed,
        "status_totals": status_totals,
        "limitations": [
            "Applicability labels are empirical screening outcomes, "
            "not probabilities of prediction correctness.",
            "Calibration uses grade-level maxima and can be conservative.",
            "The finite-sample thresholds require sufficient "
            "independent calibration grades.",
            "NIMS has only four independent grades in the dataset; "
            "its grade-level conclusions remain unstable.",
            "The mixed distance is a screening metric, not a "
            "metallurgical law or physical similarity measure.",
            "Median imputation is used for distance calculation; "
            "missing evidence is separately reported.",
            "Constant and all-missing training features cannot "
            "establish a learned reference distribution.",
            "Source-held-out results measure transfer across "
            "different dataset populations.",
            "No test target values or prediction errors are used "
            "to assign applicability statuses.",
            "No model was retrained, promoted, or modified.",
            "No raw dataset was modified.",
        ],
        "outputs": {
            "records": str(records_path),
            "fold_summary": str(summaries_path),
            "feature_summary": str(features_path),
            "split_audit": str(splits_path),
            "failures": str(failures_path),
            "status_counts": str(statuses_path),
            "calibration_reference": str(calibration_path),
        },
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )
    log("\n" + "=" * 76)
    log("MAT-X APPLICABILITY-DOMAIN ASSESSMENT V3 COMPLETE")
    log("=" * 76)
    log(json.dumps(summary_json, indent=2))
    log("\nFOLD SUMMARY")
    log(
        summaries[
            [
                "target",
                "held_out_source",
                "variant",
                "eligible_feature_count",
                "calibration_grade_count",
                "calibration_q90",
                "calibration_q95",
                "test_distance_median",
            ]
        ].to_string(index=False)
    )
    log("\nOUTPUT FILES")
    for path in [
        summary_path,
        records_path,
        summaries_path,
        features_path,
        splits_path,
        failures_path,
        statuses_path,
        calibration_path,
    ]:
        log(f"{path} | {path.stat().st_size:,} bytes")
if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )
    main()


