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
OUT = ROOT / "training_results" / "uncertainty_calibration_v1"
MODEL_OUT = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
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
RANDOM_STATE = 42
TEST_FRACTION = 0.20
CALIBRATION_FRACTION_OF_REMAINDER = 0.25
CONFIDENCE_LEVELS = [0.90, 0.95]
def finite_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")
def grouped_split(X, y, groups):
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )
    train_cal_idx, test_idx = next(
        first.split(X, y, groups=groups)
    )
    X_train_cal = X.iloc[train_cal_idx]
    y_train_cal = y.iloc[train_cal_idx]
    groups_train_cal = groups.iloc[train_cal_idx]
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_FRACTION_OF_REMAINDER,
        random_state=RANDOM_STATE + 1,
    )
    train_rel, calibration_rel = next(
        second.split(
            X_train_cal,
            y_train_cal,
            groups=groups_train_cal,
        )
    )
    train_idx = np.asarray(train_cal_idx)[train_rel]
    calibration_idx = np.asarray(train_cal_idx)[calibration_rel]
    return train_idx, calibration_idx, np.asarray(test_idx)
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
def grade_max_scores(actual, predicted, groups):
    frame = pd.DataFrame({
        "grade_id": groups.astype(str).to_numpy(),
        "absolute_residual": np.abs(
            np.asarray(actual, dtype=float)
            - np.asarray(predicted, dtype=float)
        ),
    })
    return (
        frame.groupby("grade_id")["absolute_residual"]
        .max()
        .reset_index(name="maximum_absolute_residual")
    )
def evaluate_variant(
    df,
    target,
    variant,
    numeric_features,
    categorical_features,
    trees,
):
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target, "grade_id"]).copy()
    work["grade_id"] = work["grade_id"].astype(str)
    for col in numeric_features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    X = work[numeric_features + categorical_features].copy()
    y = work[target].astype(float)
    groups = work["grade_id"]
    train_idx, calibration_idx, test_idx = grouped_split(
        X, y, groups
    )
    X_train = X.iloc[train_idx]
    X_cal = X.iloc[calibration_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_cal = y.iloc[calibration_idx]
    y_test = y.iloc[test_idx]
    g_train = groups.iloc[train_idx]
    g_cal = groups.iloc[calibration_idx]
    g_test = groups.iloc[test_idx]
    train_grades = set(g_train)
    calibration_grades = set(g_cal)
    test_grades = set(g_test)
    if train_grades & calibration_grades:
        raise RuntimeError("Grade leakage between training and calibration.")
    if train_grades & test_grades:
        raise RuntimeError("Grade leakage between training and test.")
    if calibration_grades & test_grades:
        raise RuntimeError("Grade leakage between calibration and test.")
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
        g_cal,
    )
    quantiles = {}
    interval_results = {}
    for confidence in CONFIDENCE_LEVELS:
        q = conformal_quantile(
            calibration_scores["maximum_absolute_residual"],
            confidence,
        )
        key = f"{int(confidence * 100)}"
        quantiles[key] = q
        lower = test_predictions - q
        upper = test_predictions + q
        record_coverage = float(
            np.mean(
                (y_test.to_numpy() >= lower)
                & (y_test.to_numpy() <= upper)
            )
        )
        test_frame = pd.DataFrame({
            "grade_id": g_test.to_numpy(),
            "covered": (
                (y_test.to_numpy() >= lower)
                & (y_test.to_numpy() <= upper)
            ),
        })
        whole_grade_coverage = float(
            test_frame.groupby("grade_id")["covered"].all().mean()
        )
        interval_results[key] = {
            "record_coverage": record_coverage,
            "whole_grade_coverage": whole_grade_coverage,
            "mean_interval_width": (
                float(np.mean(upper - lower))
                if math.isfinite(q)
                else None
            ),
            "quantile": q,
        }
    model_filename = f"{target}_{variant}_grade_grouped.joblib"
    model_path = MODEL_OUT / model_filename
    joblib.dump(pipeline, model_path)
    prediction_frame = pd.DataFrame({
        "grade_id": g_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "prediction": test_predictions,
        "absolute_error": np.abs(
            y_test.to_numpy() - test_predictions
        ),
    })
    for confidence in CONFIDENCE_LEVELS:
        key = f"{int(confidence * 100)}"
        q = quantiles[key]
        prediction_frame[f"lower_{key}"] = (
            test_predictions - q
        )
        prediction_frame[f"upper_{key}"] = (
            test_predictions + q
        )
    prediction_path = OUT / (
        f"{target}_{variant}_test_intervals.csv"
    )
    prediction_frame.to_csv(prediction_path, index=False)
    calibration_scores.to_csv(
        OUT / f"{target}_{variant}_calibration_grade_scores.csv",
        index=False,
    )
    report = {
        "target": target,
        "variant": variant,
        "model": "ExtraTreesRegressor",
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "dataset": str(DATASET),
        "feature_columns": numeric_features + categorical_features,
        "train_records": len(train_idx),
        "calibration_records": len(calibration_idx),
        "test_records": len(test_idx),
        "train_grades": len(train_grades),
        "calibration_grades": len(calibration_grades),
        "test_grades": len(test_grades),
        "grade_overlap_checks": {
            "train_calibration_disjoint": True,
            "train_test_disjoint": True,
            "calibration_test_disjoint": True,
        },
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_test, test_predictions))
        ),
        "r2": float(r2_score(y_test, test_predictions)),
        "calibration_grade_score_count": len(calibration_scores),
        "intervals": interval_results,
        "uncertainty_method": (
            "Group conformal calibration using maximum absolute "
            "residual per calibration grade"
        ),
        "validation_scope": "Internal grade-held-out validation only",
        "external_validation": "NOT_PERFORMED",
        "production_status": "EXPERIMENTAL_NOT_PROMOTED",
        "limitations": [
            "Calibration and test grades are separated, but source-held-out validation is not performed.",
            "Formal group coverage depends on exchangeability of grades.",
            "Record-level coverage is descriptive; the group calibration score is the maximum residual per grade.",
            "Intervals are symmetric and use absolute residuals.",
            "Evidence-enriched models require the additional measured-property inputs at inference.",
            "No external laboratory validation has been performed.",
        ],
    }
    report_path = OUT / f"{target}_{variant}_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=finite_json, allow_nan=False),
        encoding="utf-8",
    )
    return report
def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    df = pd.read_csv(DATASET, low_memory=False)
    required_identity = ["grade_id", "heat_id"]
    missing_identity = [
        col for col in required_identity if col not in df.columns
    ]
    if missing_identity:
        raise ValueError(
            f"Dataset is missing identity columns: {missing_identity}"
        )
    print("=" * 76)
    print("MAT-X GRADE-GROUPED UNCERTAINTY CALIBRATION")
    print("=" * 76)
    print(f"Dataset: {DATASET}")
    print(f"Rows: {len(df)}")
    print(f"Grades: {df['grade_id'].nunique()}")
    print(f"Output: {OUT}")
    results = []
    failures = []
    for target in PROPERTY_TARGETS:
        if target not in df.columns:
            failures.append({
                "target": target,
                "variant": "baseline",
                "reason": "Target column absent from SteelBench dataset",
            })
            print(f"\nSKIP {target}: target column absent.")
            continue
        if target in ["tensile_strength", "yield_strength"]:
            numeric_features = [
                col for col in BASE_NUMERIC if col in df.columns
            ]
            categorical_features = [
                col for col in BASE_CATEGORICAL if col in df.columns
            ]
            clean = df.dropna(subset=[target, "grade_id"]).copy()
            if target == "yield_strength":
                clean[target] = pd.to_numeric(
                    clean[target], errors="coerce"
                )
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
                excluded_count = int(anomaly_mask.sum())
                clean = clean.loc[~anomaly_mask].copy()
                print(
                    f"\nYield anomaly records excluded: {excluded_count}"
                )
            trees = 400 if target == "tensile_strength" else 500
            try:
                report = evaluate_variant(
                    clean,
                    target,
                    "baseline",
                    numeric_features,
                    categorical_features,
                    trees,
                )
                results.append(report)
                print(
                    f"SUCCESS {target} baseline | "
                    f"MAE={report['mae']:.4f} | "
                    f"R2={report['r2']:.4f} | "
                    f"train/cal/test grades="
                    f"{report['train_grades']}/"
                    f"{report['calibration_grades']}/"
                    f"{report['test_grades']}"
                )
            except Exception as exc:
                failures.append({
                    "target": target,
                    "variant": "baseline",
                    "reason": repr(exc),
                })
                print(f"FAILED {target} baseline: {exc}")
            continue
        numeric_features = [
            col for col in BASE_NUMERIC if col in df.columns
        ]
        categorical_features = [
            col for col in BASE_CATEGORICAL if col in df.columns
        ]
        try:
            baseline_report = evaluate_variant(
                df,
                target,
                "baseline",
                numeric_features,
                categorical_features,
                500,
            )
            results.append(baseline_report)
            print(
                f"SUCCESS {target} baseline | "
                f"MAE={baseline_report['mae']:.4f} | "
                f"R2={baseline_report['r2']:.4f}"
            )
        except Exception as exc:
            failures.append({
                "target": target,
                "variant": "baseline",
                "reason": repr(exc),
            })
            print(f"FAILED {target} baseline: {exc}")
        other_properties = [
            col for col in PROPERTY_TARGETS
            if col != target and col in df.columns
        ]
        enriched_numeric = numeric_features + other_properties
        try:
            enriched_report = evaluate_variant(
                df,
                target,
                "evidence_enriched",
                enriched_numeric,
                categorical_features,
                500,
            )
            results.append(enriched_report)
            print(
                f"SUCCESS {target} evidence_enriched | "
                f"MAE={enriched_report['mae']:.4f} | "
                f"R2={enriched_report['r2']:.4f}"
            )
        except Exception as exc:
            failures.append({
                "target": target,
                "variant": "evidence_enriched",
                "reason": repr(exc),
            })
            print(f"FAILED {target} evidence_enriched: {exc}")
    summary_rows = []
    for report in results:
        row = {
            "target": report["target"],
            "variant": report["variant"],
            "mae": report["mae"],
            "rmse": report["rmse"],
            "r2": report["r2"],
            "train_records": report["train_records"],
            "calibration_records": report["calibration_records"],
            "test_records": report["test_records"],
            "train_grades": report["train_grades"],
            "calibration_grades": report["calibration_grades"],
            "test_grades": report["test_grades"],
        }
        for confidence in CONFIDENCE_LEVELS:
            key = str(int(confidence * 100))
            interval = report["intervals"][key]
            row[f"q{key}"] = interval["quantile"]
            row[f"record_coverage_{key}"] = interval["record_coverage"]
            row[f"whole_grade_coverage_{key}"] = (
                interval["whole_grade_coverage"]
            )
            row[f"mean_width_{key}"] = interval["mean_interval_width"]
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            OUT / "grade_grouped_uncertainty_summary.csv",
            index=False,
        )
    final_summary = {
        "experiment": "MAT-X grade-grouped uncertainty calibration v1",
        "dataset": str(DATASET),
        "successful_variants": len(results),
        "failed_or_skipped_variants": len(failures),
        "results": results,
        "failures": failures,
        "production_integration": "NOT_AUTHORIZED",
        "interpretation": (
            "Review independent test coverage and interval widths before "
            "considering any production integration."
        ),
    }
    (OUT / "experiment_summary.json").write_text(
        json.dumps(
            final_summary,
            indent=2,
            default=finite_json,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print("\n" + "=" * 76)
    print("CALIBRATION EXPERIMENT FINISHED")
    print("=" * 76)
    print(f"Successful variants: {len(results)}")
    print(f"Failed/skipped variants: {len(failures)}")
    print(f"Results directory: {OUT}")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(
                f"- {failure['target']} / {failure['variant']}: "
                f"{failure['reason']}"
            )
    if results:
        print("\nINTERVAL PERFORMANCE:")
        for report in results:
            print(
                f"\n{report['target']} | {report['variant']}"
            )
            for confidence in CONFIDENCE_LEVELS:
                key = str(int(confidence * 100))
                item = report["intervals"][key]
                print(
                    f"  {key}%: record coverage="
                    f"{item['record_coverage']:.4f}, "
                    f"whole-grade coverage="
                    f"{item['whole_grade_coverage']:.4f}, "
                    f"mean width={item['mean_interval_width']}"
                )
    print("\nIMPORTANT: These models are experimental and are NOT promoted.")
if __name__ == "__main__":
    main()
