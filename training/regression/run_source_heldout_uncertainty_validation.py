from pathlib import Path
import json
import math
import traceback
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
OUT = ROOT / "training_results" / "source_heldout_validation_v1"
MODEL_OUT = OUT / "models"
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
BASE_NUMERIC = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T"
]
BASE_CATEGORICAL = ["quench_medium", "steel_family"]
PROPERTY_TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
CONFIDENCE_LEVELS = [0.90, 0.95]
RANDOM_STATE = 42
def safe_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)}")
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
def build_pipeline(numeric_features, categorical_features, enriched, trees):
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=enriched,
                ),
            )
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )
    if categorical_features:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    add_indicator=enriched,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
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
def grade_max_scores(actual, predicted, groups, sources):
    frame = pd.DataFrame({
        "grade_id": groups.astype(str).to_numpy(),
        "source": sources.astype(str).to_numpy(),
        "absolute_residual": np.abs(
            np.asarray(actual, dtype=float)
            - np.asarray(predicted, dtype=float)
        ),
    })
    return (
        frame.groupby(["source", "grade_id"])["absolute_residual"]
        .max()
        .reset_index(name="maximum_absolute_residual")
    )
def evaluate_fold(
    df,
    target,
    variant,
    numeric_features,
    categorical_features,
    trees,
    held_out_source,
):
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(
        subset=[target, "grade_id", "source", "heat_id"]
    ).copy()
    work["grade_id"] = work["grade_id"].astype(str)
    work["heat_id"] = work["heat_id"].astype(str)
    work["source"] = work["source"].astype(str)
    for col in numeric_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    # Entire source is reserved exclusively for final testing.
    test = work.loc[work["source"] == held_out_source].copy()
    train_cal = work.loc[work["source"] != held_out_source].copy()
    if test.empty:
        raise RuntimeError(
            f"No test records for held-out source {held_out_source}"
        )
    if train_cal["grade_id"].nunique() < 5:
        raise RuntimeError("Insufficient training/calibration grades.")
    feature_columns = numeric_features + categorical_features
    X_train_cal = train_cal[feature_columns].copy()
    y_train_cal = train_cal[target].astype(float)
    groups_train_cal = train_cal["grade_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE + SOURCES.index(held_out_source),
    )
    train_rel, calibration_rel = next(
        splitter.split(
            X_train_cal,
            y_train_cal,
            groups=groups_train_cal,
        )
    )
    train = train_cal.iloc[train_rel].copy()
    calibration = train_cal.iloc[calibration_rel].copy()
    train_grades = set(train["grade_id"])
    calibration_grades = set(calibration["grade_id"])
    test_grades = set(test["grade_id"])
    if train_grades & calibration_grades:
        raise RuntimeError("Training/calibration grade leakage.")
    if train_grades & test_grades:
        raise RuntimeError("Training/test grade leakage.")
    if calibration_grades & test_grades:
        raise RuntimeError("Calibration/test grade leakage.")
    X_train = train[feature_columns]
    y_train = train[target].astype(float)
    X_cal = calibration[feature_columns]
    y_cal = calibration[target].astype(float)
    X_test = test[feature_columns]
    y_test = test[target].astype(float)
    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        enriched=(variant == "evidence_enriched"),
        trees=trees,
    )
    pipeline.fit(X_train, y_train)
    calibration_predictions = pipeline.predict(X_cal)
    test_predictions = pipeline.predict(X_test)
    calibration_scores = grade_max_scores(
        y_cal,
        calibration_predictions,
        calibration["grade_id"],
        calibration["source"],
    )
    if calibration_scores.empty:
        raise RuntimeError("No calibration grade scores generated.")
    prediction_frame = pd.DataFrame({
        "source": test["source"].to_numpy(),
        "grade_id": test["grade_id"].to_numpy(),
        "heat_id": test["heat_id"].to_numpy(),
        "actual": y_test.to_numpy(),
        "prediction": test_predictions,
        "absolute_error": np.abs(
            y_test.to_numpy() - test_predictions
        ),
    })
    intervals = {}
    for confidence in CONFIDENCE_LEVELS:
        key = str(int(confidence * 100))
        q = conformal_quantile(
            calibration_scores["maximum_absolute_residual"],
            confidence,
        )
        lower = test_predictions - q
        upper = test_predictions + q
        covered = (
            (y_test.to_numpy() >= lower)
            & (y_test.to_numpy() <= upper)
        )
        grade_coverage_frame = pd.DataFrame({
            "grade_id": test["grade_id"].to_numpy(),
            "covered": covered,
        })
        record_coverage = float(covered.mean())
        whole_grade_coverage = float(
            grade_coverage_frame.groupby("grade_id")["covered"]
            .all()
            .mean()
        )
        mean_width = (
            float(np.mean(upper - lower))
            if math.isfinite(q)
            else None
        )
        prediction_frame[f"lower_{key}"] = lower
        prediction_frame[f"upper_{key}"] = upper
        intervals[key] = {
            "quantile": q,
            "record_coverage": record_coverage,
            "whole_grade_coverage": whole_grade_coverage,
            "mean_interval_width": mean_width,
        }
    prefix = f"{target}_{variant}_heldout_{held_out_source}"
    model_path = MODEL_OUT / f"{prefix}.joblib"
    prediction_path = OUT / f"{prefix}_predictions.csv"
    calibration_path = OUT / f"{prefix}_calibration_scores.csv"
    joblib.dump(pipeline, model_path)
    prediction_frame.to_csv(prediction_path, index=False)
    calibration_scores.to_csv(calibration_path, index=False)
    report = {
        "target": target,
        "variant": variant,
        "held_out_source": held_out_source,
        "model": "ExtraTreesRegressor",
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "calibration_scores_path": str(calibration_path),
        "feature_columns": feature_columns,
        "train_records": len(train),
        "calibration_records": len(calibration),
        "test_records": len(test),
        "train_grades": len(train_grades),
        "calibration_grades": len(calibration_grades),
        "test_grades": len(test_grades),
        "train_sources": sorted(train["source"].unique().tolist()),
        "calibration_sources": sorted(
            calibration["source"].unique().tolist()
        ),
        "test_sources": sorted(test["source"].unique().tolist()),
        "grade_overlap_checks": {
            "train_calibration_disjoint": True,
            "train_test_disjoint": True,
            "calibration_test_disjoint": True,
        },
        "held_out_source_excluded_from_training": (
            held_out_source not in set(train["source"])
            and held_out_source not in set(calibration["source"])
        ),
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_test, test_predictions))
        ),
        "r2": float(r2_score(y_test, test_predictions)),
        "calibration_grade_score_count": len(calibration_scores),
        "calibration_method": (
            "Group conformal calibration using maximum absolute "
            "residual per calibration grade"
        ),
        "intervals": intervals,
        "validation_scope": (
            "Entire source held out from training and calibration"
        ),
        "production_status": "EXPERIMENTAL_NOT_PROMOTED",
        "limitations": [
            "Only four source datasets are available.",
            "NIMS contains only four grade groups.",
            "Cross-source exchangeability is not guaranteed.",
            "Source-held-out coverage is an empirical domain-transfer test, "
            "not a guarantee of future coverage.",
            "No external laboratory validation has been performed.",
            "Evidence-enriched models require additional measured-property "
            "inputs at inference.",
        ],
    }
    report_path = OUT / f"{prefix}_report.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=safe_json,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return report
def main():
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    # Prevent accidental overwriting of a previous experiment.
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUT}\n"
            "Preserve the existing results and choose a new version."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET, low_memory=False)
    required = [
        "source", "grade_id", "heat_id"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing identity columns: {missing}")
    for col in required:
        df[col] = df[col].astype("string").str.strip()
    if df[required].isna().any().any():
        raise RuntimeError(
            "Missing source/grade/heat identifiers detected. "
            "Resolve identity quality before validation."
        )
    observed_sources = set(df["source"].unique())
    if observed_sources != set(SOURCES):
        raise RuntimeError(
            f"Unexpected source set: {sorted(observed_sources)}"
        )
    print("=" * 78)
    print("MAT-X SOURCE-HELD-OUT UNCERTAINTY VALIDATION")
    print("=" * 78)
    print(f"Dataset: {DATASET}")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUT}")
    print("\nSource distribution:")
    print(df.groupby("source").agg(
        records=("source", "size"),
        grades=("grade_id", "nunique"),
        heats=("heat_id", "nunique"),
    ).to_string())
    results = []
    failures = []
    for target in PROPERTY_TARGETS:
        if target not in df.columns:
            print(f"\nSKIP {target}: target column absent.")
            continue
        clean = df.copy()
        clean[target] = pd.to_numeric(clean[target], errors="coerce")
        # Preserve the established yield-anomaly exclusion.
        if target == "yield_strength":
            clean["tensile_strength"] = pd.to_numeric(
                clean["tensile_strength"], errors="coerce"
            )
            ratio = (
                clean[target]
                / clean["tensile_strength"].replace(0, np.nan)
            )
            anomaly_mask = (
                clean["grade_id"].astype(str).isin(
                    ["EMK0319", "EMK0320", "EMK0321"]
                )
                & (ratio < 0.03)
            )
            excluded = int(anomaly_mask.sum())
            clean = clean.loc[~anomaly_mask].copy()
            print(
                f"\nYield anomaly records excluded: {excluded}"
            )
        numeric_features = [
            col for col in BASE_NUMERIC if col in df.columns
        ]
        categorical_features = [
            col for col in BASE_CATEGORICAL if col in df.columns
        ]
        variants = [("baseline", numeric_features)]
        if target not in ["tensile_strength", "yield_strength"]:
            other_properties = [
                col for col in PROPERTY_TARGETS
                if col != target and col in df.columns
            ]
            variants.append((
                "evidence_enriched",
                numeric_features + other_properties,
            ))
        for variant, variant_numeric in variants:
            trees = 400 if target == "tensile_strength" else 500
            for held_out_source in SOURCES:
                name = f"{target} | {variant} | held out {held_out_source}"
                print(f"\nRUNNING: {name}")
                try:
                    report = evaluate_fold(
                        clean,
                        target,
                        variant,
                        variant_numeric,
                        categorical_features,
                        trees,
                        held_out_source,
                    )
                    results.append(report)
                    print(
                        f"SUCCESS | records={report['test_records']} "
                        f"| grades={report['test_grades']} "
                        f"| MAE={report['mae']:.4f} "
                        f"| R2={report['r2']:.4f}"
                    )
                    for level in ["90", "95"]:
                        interval = report["intervals"][level]
                        print(
                            f"  {level}% coverage: "
                            f"records={interval['record_coverage']:.4f}, "
                            f"whole_grade="
                            f"{interval['whole_grade_coverage']:.4f}, "
                            f"width={interval['mean_interval_width']}"
                        )
                except Exception as exc:
                    failures.append({
                        "target": target,
                        "variant": variant,
                        "held_out_source": held_out_source,
                        "reason": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
                    print(f"FAILED: {name}: {exc}")
    summary_rows = []
    for report in results:
        row = {
            "target": report["target"],
            "variant": report["variant"],
            "held_out_source": report["held_out_source"],
            "mae": report["mae"],
            "rmse": report["rmse"],
            "r2": report["r2"],
            "train_records": report["train_records"],
            "calibration_records": report["calibration_records"],
            "test_records": report["test_records"],
            "train_grades": report["train_grades"],
            "calibration_grades": report["calibration_grades"],
            "test_grades": report["test_grades"],
            "calibration_grade_score_count": (
                report["calibration_grade_score_count"]
            ),
        }
        for level in ["90", "95"]:
            interval = report["intervals"][level]
            row[f"q{level}"] = interval["quantile"]
            row[f"record_coverage_{level}"] = (
                interval["record_coverage"]
            )
            row[f"whole_grade_coverage_{level}"] = (
                interval["whole_grade_coverage"]
            )
            row[f"mean_width_{level}"] = (
                interval["mean_interval_width"]
            )
        summary_rows.append(row)
    summary_path = OUT / "source_heldout_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            summary_path,
            index=False,
        )
    final_summary = {
        "experiment": "MAT-X source-held-out uncertainty validation v1",
        "dataset": str(DATASET),
        "sources": SOURCES,
        "successful_variants": len(results),
        "failed_variants": len(failures),
        "results": results,
        "failures": failures,
        "production_status": "EXPERIMENTAL_NOT_PROMOTED",
    }
    (OUT / "experiment_summary.json").write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=safe_json,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print("\n" + "=" * 78)
    print("SOURCE-HELD-OUT EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Successful folds: {len(results)}")
    print(f"Failed folds: {len(failures)}")
    print(f"Expected folds: 32")
    print(f"Summary: {summary_path}")
    print(f"Full report: {OUT / 'experiment_summary.json'}")
    print("\nIMPORTANT: Models remain experimental and are NOT promoted.")
if __name__ == "__main__":
    main()
