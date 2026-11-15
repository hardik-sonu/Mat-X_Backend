from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
NUMERIC_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL_FEATURES = ["quench_medium", "steel_family"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
COMPOSITION_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
]
PROCESS_FEATURES = [
    "austenitize_T", "temper_T", "quench_medium",
]
TARGET_CONFIG = {
    "tensile_strength": {
        "model": "models/steelbench_tensile_extratrees.joblib",
        "artifact": "training_results/regression/steelbench_tensile_candidate_test_predictions.csv",
    },
    "yield_strength": {
        "model": "models/steelbench_yield_extratrees.joblib",
        "artifact": "training_results/regression/steelbench_yield_candidate_test_predictions.csv",
    },
}
RANDOM_STATE = 42
BOOTSTRAP_REPLICATES = 5000
def build_evidence_conditions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build four evidence conditions without mutating the original frame."""
    missing = set(COMPOSITION_FEATURES + PROCESS_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required evidence features: {sorted(missing)}")
    hidden = frame.copy()
    hidden[COMPOSITION_FEATURES + PROCESS_FEATURES] = np.nan
    composition_only = hidden.copy()
    composition_only[COMPOSITION_FEATURES] = frame[COMPOSITION_FEATURES]
    process_only = hidden.copy()
    process_only[PROCESS_FEATURES] = frame[PROCESS_FEATURES]
    return {
        "hidden_baseline": hidden,
        "composition_only": composition_only,
        "process_only": process_only,
        "full_evidence": frame.copy(),
    }
def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int = RANDOM_STATE,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of paired per-record differences."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap input must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap input must contain only finite values.")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_95_lower": float(np.quantile(means, 0.025)),
        "ci_95_upper": float(np.quantile(means, 0.975)),
        "replicates": int(replicates),
    }
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Actual and predicted values must be matching one-dimensional arrays.")
    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty test set.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation values must be finite.")
    errors = actual - predicted
    absolute_errors = np.abs(errors)
    return {
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }
def evaluate_model(
    model: Any,
    conditions: dict[str, pd.DataFrame],
    actual: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics = {}
    predictions = {}
    for name, frame in conditions.items():
        predicted = np.asarray(model.predict(frame), dtype=float)
        if predicted.shape != actual.shape:
            raise ValueError(f"Unexpected prediction shape for {name}.")
        if not np.isfinite(predicted).all():
            raise ValueError(f"Non-finite predictions produced for {name}.")
        predictions[name] = predicted
        metrics[name] = calculate_metrics(actual, predicted)
    return metrics, predictions
def prepare_target_data(
    dataset: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = set(FEATURES + [target, "grade_id", "heat_id"])
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    work = dataset.dropna(subset=[target, "grade_id"]).copy()
    excluded_anomalies = np.zeros(len(work), dtype=bool)
    if target == "yield_strength":
        ratio = (
            work[target] /
            work["tensile_strength"].replace(0, np.nan)
        )
        excluded_anomalies = (
            work["grade_id"].astype(str).isin(
                ["EMK0319", "EMK0320", "EMK0321"]
            )
            & (ratio < 0.03)
        ).to_numpy()
        work = work.loc[~excluded_anomalies].copy()
    if not np.isfinite(work[target].to_numpy(dtype=float)).all():
        raise ValueError(f"{target} contains non-finite target values.")
    X = work[FEATURES].copy()
    y = work[target].to_numpy(dtype=float)
    groups = work["grade_id"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train = work.iloc[train_idx]
    test = work.iloc[test_idx]
    if set(train["grade_id"]) & set(test["grade_id"]):
        raise RuntimeError("Grade leakage detected in the reproduced split.")
    if set(train["heat_id"]) & set(test["heat_id"]):
        raise RuntimeError("Heat leakage detected in the reproduced split.")
    return work, train_idx, test_idx, y
def run_experiment(
    dataset_path: Path,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = pd.read_csv(dataset_path, low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment": "MAT-X Experiment G — Hide-and-Reveal Evidence Evaluation",
        "experiment_version": "1.0.0",
        "status": "COMPLETED",
        "method": "Grade-held-out paired hide-and-reveal prediction-error evaluation",
        "split": {
            "type": "GroupShuffleSplit",
            "group": "grade_id",
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
        },
        "evidence_groups": {
            "composition": COMPOSITION_FEATURES,
            "process_history": PROCESS_FEATURES,
            "fixed_context": ["steel_family"],
        },
        "targets": {},
        "limitations": [
            "ExtraTrees model predictions are evaluated; this is not calibrated predictive uncertainty.",
            "Prediction-error improvement is not Shannon information gain.",
            "The experiment covers only composition and process-history features.",
            "Results do not validate microstructure, phase, grain, mechanical-testing, or damage recommendations.",
            "Results are conditional on the SteelBench dataset and promoted model versions.",
            "The four conditions are model-input ablations, not a simulation of the full cost or logistics of laboratory testing.",
            "The fixed steel_family feature remains visible in every condition.",
        ],
    }
    all_rows = []
    for target, config in TARGET_CONFIG.items():
        work, train_idx, test_idx, y = prepare_target_data(dataset, target)
        X = work[FEATURES].copy()
        X_test = X.iloc[test_idx].copy()
        actual = y[test_idx]
        model_path = project_root / config["model"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = joblib.load(model_path)
        conditions = build_evidence_conditions(X_test)
        metrics, predictions = evaluate_model(model, conditions, actual)
        baseline_error = np.abs(actual - predictions["hidden_baseline"])
        composition_error = np.abs(actual - predictions["composition_only"])
        process_error = np.abs(actual - predictions["process_only"])
        full_error = np.abs(actual - predictions["full_evidence"])
        composition_improvement = baseline_error - composition_error
        process_improvement = baseline_error - process_error
        full_improvement = baseline_error - full_error
        target_report = {
            "model_path": config["model"],
            "eligible_records": int(len(work)),
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "train_unique_grades": int(work.iloc[train_idx]["grade_id"].nunique()),
            "test_unique_grades": int(work.iloc[test_idx]["grade_id"].nunique()),
            "grade_overlap": 0,
            "heat_overlap": 0,
            "metrics": metrics,
            "paired_improvement": {
                "composition_vs_hidden": paired_bootstrap_ci(composition_improvement),
                "process_vs_hidden": paired_bootstrap_ci(process_improvement, seed=43),
                "full_vs_hidden": paired_bootstrap_ci(full_improvement, seed=44),
                "composition_minus_process_improvement": paired_bootstrap_ci(
                    composition_improvement - process_improvement,
                    seed=45,
                ),
            },
            "record_counts": {
                "composition_improved": int((composition_improvement > 0).sum()),
                "composition_worsened": int((composition_improvement < 0).sum()),
                "composition_unchanged": int((composition_improvement == 0).sum()),
                "process_improved": int((process_improvement > 0).sum()),
                "process_worsened": int((process_improvement < 0).sum()),
                "full_evidence_improved": int((full_improvement > 0).sum()),
                "full_evidence_worsened": int((full_improvement < 0).sum()),
                "composition_better_than_process": int(
                    (composition_improvement > process_improvement).sum()
                ),
                "process_better_than_composition": int(
                    (process_improvement > composition_improvement).sum()
                ),
            },
        }
        artifact_path = project_root / config["artifact"]
        if artifact_path.exists():
            artifact = pd.read_csv(artifact_path)
            if len(artifact) != len(test_idx):
                raise ValueError(
                    f"{target}: candidate artifact row count does not match test split."
                )
            if "actual" not in artifact:
                raise ValueError(f"{target}: candidate artifact has no actual column.")
            max_difference = float(
                np.max(np.abs(artifact["actual"].to_numpy(dtype=float) - actual))
            )
            target_report["candidate_artifact"] = {
                "path": config["artifact"],
                "rows": int(len(artifact)),
                "max_actual_value_difference": max_difference,
                "matches_reproduced_split": bool(max_difference == 0.0),
            }
            if max_difference != 0.0:
                raise ValueError(f"{target}: candidate artifact actual values do not match.")
        else:
            target_report["candidate_artifact"] = {
                "status": "NOT_FOUND",
                "path": config["artifact"],
            }
        target_report["model_feature_count"] = int(len(FEATURES))
        report["targets"][target] = target_report
        for row_number, (actual_value, row) in enumerate(
            zip(actual, X_test.to_dict(orient="records"))
        ):
            output_row = {
                "target": target,
                "test_position": row_number,
                "actual": float(actual_value),
                "grade_id": str(work.iloc[test_idx[row_number]]["grade_id"]),
                "heat_id": str(work.iloc[test_idx[row_number]]["heat_id"]),
            }
            for condition_name, values in predictions.items():
                output_row[f"{condition_name}_prediction"] = float(values[row_number])
                output_row[f"{condition_name}_absolute_error"] = float(
                    abs(actual_value - values[row_number])
                )
            output_row["composition_error_improvement"] = float(
                composition_improvement[row_number]
            )
            output_row["process_error_improvement"] = float(
                process_improvement[row_number]
            )
            output_row["full_evidence_error_improvement"] = float(
                full_improvement[row_number]
            )
            all_rows.append(output_row)
    report["overall_interpretation"] = (
        "Empirical hide-and-reveal prediction-error results are reported for "
        "composition and process-history features only. They must not be "
        "interpreted as calibrated uncertainty or generalized to unsupported "
        "characterization categories."
    )
    csv_path = output_dir / "experiment_g_record_results.csv"
    json_path = output_dir / "experiment_g_report.json"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nCSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    return report
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_results/active_investigation"),
    )
    args = parser.parse_args()
    run_experiment(
        dataset_path=args.dataset,
        project_root=args.project_root,
        output_dir=args.project_root / args.output_dir,
    )
if __name__ == "__main__":
    main()