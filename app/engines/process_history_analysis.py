from pathlib import Path
import json
import os
import pandas as pd
from app.engines.process_history_validation import validate_process_history_frame
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.getenv(
    "MATX_PROCESS_HISTORY_OUTPUT_DIR",
    str(ROOT / "training_results" / "process_history")
))
INPUT = OUTPUT / "process_history_reconstruction.csv"
ENGINE_VERSION = "1.0.0"
PROCESS_FIELDS = ["austenitize_T", "quench_medium", "temper_T"]
REQUIRED_COLUMNS = [
    "heat_id", "grade_id", "steel_family", "source",
    "process_status", *PROCESS_FIELDS
]
def analyze_process_history(
    input_path=INPUT,
    output_dir=OUTPUT,
    verbose=False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Process history reconstruction CSV not found: {input_path}"
        )
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Process history reconstruction dataset is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    validate_process_history_frame(df, require_generated=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_df = df.copy()
    for field in PROCESS_FIELDS:
        route_df[field] = route_df[field].where(
            route_df[field].notna(), "NOT_DOCUMENTED"
        )
    route_df["observed_process_combination"] = (
        route_df[PROCESS_FIELDS].astype(str).agg(" | ".join, axis=1)
    )
    route_summary = (
        route_df.groupby(
            ["steel_family", "observed_process_combination"],
            dropna=False,
        )
        .agg(
            heat_count=("heat_id", "nunique"),
            source_count=("source", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["heat_count", "steel_family"],
            ascending=[False, True],
        )
    )
    route_summary.to_csv(
        output_dir / "process_history_route_summary.csv",
        index=False,
    )
    complete_routes = route_df[
        route_df[PROCESS_FIELDS].ne("NOT_DOCUMENTED").all(axis=1)
    ]
    complete_route_summary = (
        complete_routes.groupby(
            ["steel_family", "observed_process_combination"],
            dropna=False,
        )
        .agg(
            heat_count=("heat_id", "nunique"),
            source_count=("source", "nunique"),
        )
        .reset_index()
        .sort_values("heat_count", ascending=False)
    )
    complete_route_summary.to_csv(
        output_dir / "process_history_complete_routes.csv",
        index=False,
    )
    family_summary = (
        df.groupby("steel_family")
        .agg(
            total_heats=("heat_id", "nunique"),
            complete_records=(
                "process_status",
                lambda x: (x == "COMPLETE_DOCUMENTED_PROCESS_FIELDS").sum(),
            ),
            partial_records=(
                "process_status",
                lambda x: (x == "PARTIAL_PROCESS_HISTORY").sum(),
            ),
            no_documented_records=(
                "process_status",
                lambda x: (x == "NO_DOCUMENTED_PROCESS_HISTORY").sum(),
            ),
            austenitize_available=("austenitize_T", "count"),
            quench_available=("quench_medium", "count"),
            temper_available=("temper_T", "count"),
        )
        .reset_index()
    )
    family_summary.to_csv(
        output_dir / "process_history_family_summary.csv",
        index=False,
    )
    grade_summary = (
        df.groupby(["grade_id", "steel_family"])
        .agg(
            heat_count=("heat_id", "nunique"),
            austenitize_available=("austenitize_T", "count"),
            quench_available=("quench_medium", "count"),
            temper_available=("temper_T", "count"),
            complete_records=(
                "process_status",
                lambda x: (x == "COMPLETE_DOCUMENTED_PROCESS_FIELDS").sum(),
            ),
            partial_records=(
                "process_status",
                lambda x: (x == "PARTIAL_PROCESS_HISTORY").sum(),
            ),
            no_documented_records=(
                "process_status",
                lambda x: (x == "NO_DOCUMENTED_PROCESS_HISTORY").sum(),
            ),
        )
        .reset_index()
        .sort_values("heat_count", ascending=False)
    )
    grade_summary.to_csv(
        output_dir / "process_history_grade_summary.csv",
        index=False,
    )
    report = {
        "engine": "MAT-X Process History Pattern Analysis",
        "version": ENGINE_VERSION,
        "status": "OBSERVED_PROCESS_COMBINATIONS_ONLY",
        "total_heat_records": int(len(df)),
        "steel_families": int(df["steel_family"].nunique()),
        "unique_grades": int(df["grade_id"].nunique()),
        "unique_observed_combinations": int(
            route_summary["observed_process_combination"].nunique()
        ),
        "complete_route_combinations": int(len(complete_route_summary)),
        "complete_route_heat_records": int(len(complete_routes)),
        "methodology": {
            "uses_observed_values_only": True,
            "missing_values_imputed": False,
            "unknown_conditions_predicted": False,
            "chronological_sequence_assumed": False,
        },
        "outputs": [
            "process_history_route_summary.csv",
            "process_history_complete_routes.csv",
            "process_history_family_summary.csv",
            "process_history_grade_summary.csv",
        ],
    }
    (output_dir / "process_history_pattern_report.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )
    if verbose:
        print("MAT-X PROCESS HISTORY PATTERN ANALYSIS")
        print("Heat records:", len(df))
        print("Steel families:", df["steel_family"].nunique())
        print("Unique grades:", df["grade_id"].nunique())
        print("Observed combinations:", report["unique_observed_combinations"])
        print("Complete-route combinations:", len(complete_route_summary))
        print("Complete-route heat records:", len(complete_routes))
        print("Original dataset modified: NO")
    return report
if __name__ == "__main__":
    analyze_process_history(verbose=True)
