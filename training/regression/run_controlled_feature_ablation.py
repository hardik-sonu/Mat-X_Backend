from pathlib import Path
import json
import math
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = ROOT / "training_results" / "controlled_feature_ablation_v1"
PRED_OUT = OUT / "predictions"
MODEL_OUT = OUT / "models"
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
NUMERIC_13 = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL_13 = [
    "quench_medium",
    "steel_family",
]
VARIANTS = {
    "full_13": {
        "numeric": NUMERIC_13,
        "categorical": CATEGORICAL_13,
        "description": "11 numerical + 2 categorical features",
    },
    "chemistry_2": {
        "numeric": ["C", "Mn"],
        "categorical": [],
        "description": "C + Mn",
    },
    "chemistry_3": {
        "numeric": ["C", "Mn", "V"],
        "categorical": [],
        "description": "C + Mn + V; diagnostic only",
    },
}
RANDOM_STATE = 42
CALIBRATION_FRACTION = 0.25
CONFIDENCE_LEVELS = [0.90, 0.95]
if not DATASET.exists():
    raise FileNotFoundError(DATASET)
if OUT.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUT}"
    )
OUT.mkdir(parents=True)
PRED_OUT.mkdir()
MODEL_OUT.mkdir()
df = pd.read_csv(DATASET, low_memory=False)
required = [
    "source", "grade_id", "heat_id",
    *NUMERIC_13, *CATEGORICAL_13, *TARGETS,
]
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise RuntimeError(
        f"Missing required dataset columns: {missing_columns}"
    )
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for col in NUMERIC_13 + TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[["source", "grade_id", "heat_id"]].isna().any().any():
    raise RuntimeError("Missing source, grade_id or heat_id values.")
# Verify shared chemistry feature availability.
for col in ["C", "Mn", "V"]:
    missing = int(df[col].isna().sum())
    if missing:
        raise RuntimeError(
            f"{col} has {missing} missing values. "
            "Do not compare feature sets on unequal records."
        )
def build_pipeline(numeric_features, categorical_features, trees):
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])
def conformal_quantile(scores, confidence):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * confidence))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])
def grade_max_scores(actual, predicted, grades):
    frame = pd.DataFrame({
        "grade_id": np.asarray(grades).astype(str),
        "absolute_residual": np.abs(
            np.asarray(actual, dtype=float)
            - np.asarray(predicted, dtype=float)
        ),
    })
    return (
        frame.groupby("grade_id")["absolute_residual"]
        .max()
        .to_numpy(dtype=float)
    )
comparison_rows = []
prediction_audits = []
split_audits = []
failures = []
print("=" * 100)
print("MAT-X CONTROLLED FEATURE ABLATION V1")
print("=" * 100)
print("Variants:", list(VARIANTS))
print("Sources:", SOURCES)
print("Identical target populations and grade-grouped splits.")
print("No original datasets or promoted models will be modified.")
for target in TARGETS:
    print("\n" + "=" * 100)
    print("TARGET:", target)
    print("=" * 100)
    # Identical eligible population for every feature variant.
    clean = df.dropna(
        subset=[target, "source", "grade_id", "heat_id"]
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
        print("Yield anomaly records excluded:", excluded)
    clean["source"] = clean["source"].astype(str)
    clean["grade_id"] = clean["grade_id"].astype(str)
    clean["heat_id"] = clean["heat_id"].astype(str)
    # Check unique identifiers before splitting.
    if clean.duplicated(
        subset=["source", "grade_id", "heat_id"]
    ).any():
        raise RuntimeError(
            f"Duplicate source/grade/heat identifiers for {target}."
        )
    trees = 400 if target == "tensile_strength" else 500
    print("Eligible records:", len(clean))
    print("Eligible grades:", clean["grade_id"].nunique())
    print("Trees:", trees)
    for held_out_source in SOURCES:
        test = clean.loc[
            clean["source"] == held_out_source
        ].copy()
        train_cal = clean.loc[
            clean["source"] != held_out_source
        ].copy()
        if test.empty:
            print("SKIP", held_out_source, "- no test labels")
            continue
        if train_cal.empty:
            print("SKIP", held_out_source, "- no training labels")
            continue
        if train_cal["grade_id"].nunique() < 5:
            print("SKIP", held_out_source, "- insufficient grades")
            continue
        # Generate ONE split, reused by every feature variant.
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
        if train_grades & calibration_grades:
            raise RuntimeError("Training/calibration grade leakage.")
        if train_grades & test_grades:
            raise RuntimeError("Training/test grade leakage.")
        if calibration_grades & test_grades:
            raise RuntimeError("Calibration/test grade leakage.")
        split_audits.append({
            "target": target,
            "held_out_source": held_out_source,
            "train_records": len(train),
            "calibration_records": len(calibration),
            "test_records": len(test),
            "train_grades": len(train_grades),
            "calibration_grades": len(calibration_grades),
            "test_grades": len(test_grades),
            "train_calibration_overlap": len(
                train_grades & calibration_grades
            ),
            "train_test_overlap": len(train_grades & test_grades),
            "calibration_test_overlap": len(
                calibration_grades & test_grades
            ),
        })
        print(
            f"\nFOLD: {held_out_source} | "
            f"train={len(train)} | "
            f"calibration={len(calibration)} | "
            f"test={len(test)}"
        )
        predictions_by_variant = {}
        for variant_name, config in VARIANTS.items():
            numeric_features = config["numeric"]
            categorical_features = config["categorical"]
            feature_columns = (
                numeric_features + categorical_features
            )
            try:
                pipeline = build_pipeline(
                    numeric_features,
                    categorical_features,
                    trees,
                )
                pipeline.fit(
                    train[feature_columns],
                    train[target].astype(float),
                )
                calibration_predictions = pipeline.predict(
                    calibration[feature_columns]
                )
                test_predictions = pipeline.predict(
                    test[feature_columns]
                )
                actual_cal = calibration[target].to_numpy(
                    dtype=float
                )
                actual_test = test[target].to_numpy(
                    dtype=float
                )
                if not np.isfinite(test_predictions).all():
                    raise RuntimeError(
                        "Non-finite test predictions."
                    )
                if not np.isfinite(calibration_predictions).all():
                    raise RuntimeError(
                        "Non-finite calibration predictions."
                    )
                # Grade-max calibration scores.
                calibration_scores = grade_max_scores(
                    actual_cal,
                    calibration_predictions,
                    calibration["grade_id"],
                )
                if len(calibration_scores) == 0:
                    raise RuntimeError(
                        "No calibration grade scores."
                    )
                q_values = {
                    confidence: conformal_quantile(
                        calibration_scores,
                        confidence,
                    )
                    for confidence in CONFIDENCE_LEVELS
                }
                mae = float(
                    mean_absolute_error(
                        actual_test,
                        test_predictions,
                    )
                )
                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            actual_test,
                            test_predictions,
                        )
                    )
                )
                r2 = (
                    float(r2_score(actual_test, test_predictions))
                    if len(actual_test) >= 2
                    and np.var(actual_test) > 0
                    else None
                )
                prediction_frame = pd.DataFrame({
                    "source": test["source"].to_numpy(),
                    "grade_id": test["grade_id"].to_numpy(),
                    "heat_id": test["heat_id"].to_numpy(),
                    "actual": actual_test,
                    "prediction": test_predictions,
                    "absolute_error": np.abs(
                        actual_test - test_predictions
                    ),
                })
                for confidence, q in q_values.items():
                    key = int(confidence * 100)
                    lower = test_predictions - q
                    upper = test_predictions + q
                    covered = (
                        (actual_test >= lower)
                        & (actual_test <= upper)
                    )
                    prediction_frame[f"lower_{key}"] = lower
                    prediction_frame[f"upper_{key}"] = upper
                    prediction_frame[f"covered_{key}"] = covered
                    prediction_frame[f"width_{key}"] = (
                        upper - lower
                    )
                variant_dir = (
                    PRED_OUT
                    / target
                    / held_out_source
                )
                variant_dir.mkdir(parents=True, exist_ok=True)
                prediction_path = (
                    variant_dir / f"{variant_name}.csv"
                )
                prediction_frame.to_csv(
                    prediction_path,
                    index=False,
                )
                predictions_by_variant[variant_name] = (
                    test_predictions.copy()
                )
                result = {
                    "target": target,
                    "held_out_source": held_out_source,
                    "variant": variant_name,
                    "features": feature_columns,
                    "description": config["description"],
                    "train_records": len(train),
                    "calibration_records": len(calibration),
                    "test_records": len(test),
                    "train_grades": len(train_grades),
                    "calibration_grades": len(calibration_grades),
                    "test_grades": len(test_grades),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "q90": q_values[0.90],
                    "q95": q_values[0.95],
                    "record_coverage_90": float(
                        prediction_frame["covered_90"].mean()
                    ),
                    "record_coverage_95": float(
                        prediction_frame["covered_95"].mean()
                    ),
                    "mean_interval_width_90": float(
                        prediction_frame["width_90"].mean()
                    ),
                    "mean_interval_width_95": float(
                        prediction_frame["width_95"].mean()
                    ),
                    "prediction_file": str(prediction_path),
                    "status": "COMPLETED",
                }
                for confidence in CONFIDENCE_LEVELS:
                    key = int(confidence * 100)
                    grade_coverage = (
                        prediction_frame
                        .groupby("grade_id")[f"covered_{key}"]
                        .all()
                    )
                    result[
                        f"whole_grade_coverage_{key}"
                    ] = float(grade_coverage.mean())
                comparison_rows.append(result)
                print(
                    f"  {variant_name:12s} | "
                    f"MAE={mae:9.3f} | "
                    f"RMSE={rmse:9.3f} | "
                    f"R2={r2 if r2 is not None else float('nan'):8.3f}"
                )
                # Store model only inside this experiment.
                model_path = (
                    MODEL_OUT
                    / f"{target}__{held_out_source}__{variant_name}.joblib"
                )
                joblib.dump(pipeline, model_path)
                prediction_audits.append({
                    "target": target,
                    "held_out_source": held_out_source,
                    "variant": variant_name,
                    "prediction_rows": len(prediction_frame),
                    "actual_values_finite": bool(
                        np.isfinite(actual_test).all()
                    ),
                    "predictions_finite": bool(
                        np.isfinite(test_predictions).all()
                    ),
                    "all_intervals_ordered": bool(
                        (
                            prediction_frame["lower_90"]
                            <= prediction_frame["upper_90"]
                        ).all()
                        and (
                            prediction_frame["lower_95"]
                            <= prediction_frame["upper_95"]
                        ).all()
                    ),
                })
            except Exception as exc:
                failures.append({
                    "target": target,
                    "held_out_source": held_out_source,
                    "variant": variant_name,
                    "error": repr(exc),
                })
                print(
                    f"  FAILED {variant_name}: {repr(exc)}"
                )
        # Verify all variants used the same test population.
        if predictions_by_variant:
            lengths = {
                name: len(values)
                for name, values in predictions_by_variant.items()
            }
            if len(set(lengths.values())) != 1:
                raise RuntimeError(
                    f"Unequal test prediction counts: {lengths}"
                )
results = pd.DataFrame(comparison_rows)
splits = pd.DataFrame(split_audits)
audits = pd.DataFrame(prediction_audits)
failure_frame = pd.DataFrame(failures)
results.to_csv(
    OUT / "controlled_ablation_metrics.csv",
    index=False,
)
splits.to_csv(
    OUT / "shared_split_audit.csv",
    index=False,
)
audits.to_csv(
    OUT / "prediction_integrity_audit.csv",
    index=False,
)
failure_frame.to_csv(
    OUT / "execution_failures.csv",
    index=False,
)
overall_rows = []
if not results.empty:
    for target, group in results.groupby("target"):
        # Pool the same held-out predictions across source folds.
        prediction_frames = []
        for _, row in group.iterrows():
            prediction_path = Path(row["prediction_file"])
            prediction_frames.append(
                pd.read_csv(prediction_path)
            )
        pooled = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        actual = pooled["actual"].to_numpy(dtype=float)
        for variant in VARIANTS:
            # Rebuild pooled predictions from the stored CSVs.
            variant_frames = []
            for _, row in group.loc[
                group["variant"] == variant
            ].iterrows():
                variant_frames.append(
                    pd.read_csv(row["prediction_file"])
                )
            if not variant_frames:
                continue
            variant_data = pd.concat(
                variant_frames,
                ignore_index=True,
            )
            variant_actual = variant_data[
                "actual"
            ].to_numpy(dtype=float)
            variant_prediction = variant_data[
                "prediction"
            ].to_numpy(dtype=float)
            overall_rows.append({
                "target": target,
                "variant": variant,
                "records": len(variant_data),
                "grades": variant_data["grade_id"].nunique(),
                "mae": float(
                    mean_absolute_error(
                        variant_actual,
                        variant_prediction,
                    )
                ),
                "rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            variant_actual,
                            variant_prediction,
                        )
                    )
                ),
                "r2": (
                    float(
                        r2_score(
                            variant_actual,
                            variant_prediction,
                        )
                    )
                    if len(variant_actual) >= 2
                    and np.var(variant_actual) > 0
                    else None
                ),
            })
overall = pd.DataFrame(overall_rows)
overall.to_csv(
    OUT / "controlled_ablation_overall.csv",
    index=False,
)
summary = {
    "experiment": "MAT-X controlled feature ablation v1",
    "dataset": str(DATASET),
    "variants": VARIANTS,
    "targets": TARGETS,
    "sources": SOURCES,
    "split_protocol": {
        "held_out_source": "Entire source reserved for final testing",
        "training_calibration_split": "75/25 GroupShuffleSplit",
        "group": "grade_id",
        "random_state": RANDOM_STATE,
        "source_adjusted_seed": True,
        "identical_split_across_variants": True,
    },
    "model": {
        "type": "ExtraTreesRegressor",
        "tensile_trees": 400,
        "other_target_trees": 500,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    },
    "successful_runs": len(results),
    "execution_failures": len(failures),
    "split_audit_passed": bool(
        not splits.empty
        and (
            splits[
                [
                    "train_calibration_overlap",
                    "train_test_overlap",
                    "calibration_test_overlap",
                ]
            ].to_numpy() == 0
        ).all()
    ),
    "prediction_integrity_passed": bool(
        not audits.empty
        and audits[
            [
                "actual_values_finite",
                "predictions_finite",
                "all_intervals_ordered",
            ]
        ].all().all()
    ),
    "limitations": [
        "Source-held-out results measure cross-source transfer, not random within-source generalization.",
        "Conformal coverage under source shift is empirical and is not guaranteed.",
        "V is included only as a diagnostic because it is constant in three sources.",
        "No variant is automatically promoted for engineering use.",
        "Human engineering judgment and independent validation remain necessary.",
    ],
    "output_directory": str(OUT),
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2, default=str),
    encoding="utf-8",
)
print("\n" + "=" * 100)
print("CONTROLLED ABLATION RESULTS")
print("=" * 100)
if not results.empty:
    print(
        results[
            [
                "target",
                "held_out_source",
                "variant",
                "test_records",
                "mae",
                "rmse",
                "r2",
                "record_coverage_90",
                "record_coverage_95",
            ]
        ].to_string(index=False)
    )
print("\n" + "=" * 100)
print("POOLED RESULTS")
print("=" * 100)
if not overall.empty:
    print(overall.to_string(index=False))
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(json.dumps(summary, indent=2, default=str))
print("\nOriginal datasets and promoted models preserved.")
