"""Scoped evaluation of Experiment G evidence-ablation results.
This module summarizes prediction-error changes. It does NOT estimate
Shannon information gain, measurement uncertainty reduction, or the
causal value of physically acquiring a characterization test.
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "training_results"
    / "active_investigation"
    / "experiment_g_record_results.csv"
)
REQUIRED_COLUMNS = {
    "target",
    "composition_error_improvement",
    "process_error_improvement",
}
def _finite_number(value: str, column: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value in {column}, CSV row {row_number}."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite numeric value in {column}, CSV row {row_number}."
        )
    return number
def _summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize an empty experiment group.")
    return {
        "records": len(values),
        "mean_error_improvement": mean(values),
        "median_error_improvement": median(values),
        "improved_records": sum(value > 0 for value in values),
        "worsened_records": sum(value < 0 for value in values),
        "unchanged_records": sum(value == 0 for value in values),
        "improved_fraction": sum(value > 0 for value in values) / len(values),
    }
def evaluate_experiment_g(csv_path: str | Path = DEFAULT_RESULTS_PATH) -> dict:
    """Summarize composition/process ablation by target from the source CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment G results not found: {path}")
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"composition": [], "process": []}
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Experiment G CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            target = (row.get("target") or "").strip()
            if not target:
                raise ValueError(f"Missing target in CSV row {row_number}.")
            groups[target]["composition"].append(
                _finite_number(
                    row["composition_error_improvement"],
                    "composition_error_improvement",
                    row_number,
                )
            )
            groups[target]["process"].append(
                _finite_number(
                    row["process_error_improvement"],
                    "process_error_improvement",
                    row_number,
                )
            )
    if not groups:
        raise ValueError("Experiment G CSV contains no data records.")
    target_results = {}
    for target, evidence in sorted(groups.items()):
        composition = _summarize(evidence["composition"])
        process = _summarize(evidence["process"])
        if composition["records"] != process["records"]:
            raise ValueError(f"Unequal paired record counts for target {target}.")
        descriptive_preference = (
            "COMPOSITION"
            if (
                composition["mean_error_improvement"]
                > process["mean_error_improvement"]
                and composition["improved_records"] > process["improved_records"]
            )
            else "PROCESS"
            if (
                process["mean_error_improvement"]
                > composition["mean_error_improvement"]
                and process["improved_records"] > composition["improved_records"]
            )
            else "NO_CLEAR_DIFFERENCE"
        )
        target_results[target] = {
            "composition": composition,
            "process": process,
            "descriptive_preference": descriptive_preference,
        }
    return {
        "experiment": "EXPERIMENT_G",
        "status": "LIMITED_EXPERIMENTAL_PREDICTION_ERROR_ABLATION",
        "scope": {
            "evaluation": "grade-held-out hide-and-reveal prediction-error comparison",
            "evidence_groups": ["composition", "process"],
            "targets": sorted(target_results),
            "measures_information_gain": False,
            "validates_physical_test_acquisition": False,
        },
        "targets": target_results,
        "limitations": [
            "Prediction-error ablation is not Shannon information gain.",
            "The experiment does not measure uncertainty reduction from a physical test.",
            "Results are limited to the targets and evidence groups present in this CSV.",
            "No conclusion is established for the other characterization categories.",
        ],
    }
