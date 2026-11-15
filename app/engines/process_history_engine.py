from pathlib import Path
import json
import os
import pandas as pd
from app.engines.process_history_validation import validate_process_history_frame
ROOT = Path(__file__).resolve().parents[2]
from app.core.paths import DATASET_ROOT as DATASET, STEELBENCH_PATH
DATA_FILE = STEELBENCH_PATH
OUTPUT_DIR = Path(os.getenv(
    "MATX_PROCESS_HISTORY_OUTPUT_DIR",
    str(ROOT / "training_results" / "process_history")
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_VERSION = "1.0.0"
PROCESS_FIELDS = [
    "austenitize_T",
    "quench_medium",
    "temper_T",
]
def reconstruct_process_history():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = [
        "heat_id", "grade_id", "steel_family",
        "source", *PROCESS_FIELDS
    ]
    missing_columns = [
        column for column in required
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )
    validate_process_history_frame(df)
    records = []
    for _, row in df.iterrows():
        observed = {}
        missing = []
        for field in PROCESS_FIELDS:
            value = row[field]
            if pd.isna(value):
                observed[field] = None
                missing.append(field)
            else:
                observed[field] = (
                    float(value)
                    if field != "quench_medium"
                    else str(value)
                )
        documented_count = len(PROCESS_FIELDS) - len(missing)
        if documented_count == 3:
            status = "COMPLETE_DOCUMENTED_PROCESS_FIELDS"
        elif documented_count > 0:
            status = "PARTIAL_PROCESS_HISTORY"
        else:
            status = "NO_DOCUMENTED_PROCESS_HISTORY"
        route = []
        if observed["austenitize_T"] is not None:
            route.append("AUSTENITIZING_DOCUMENTED")
        if observed["quench_medium"] is not None:
            route.append("QUENCHING_DOCUMENTED")
        if observed["temper_T"] is not None:
            route.append("TEMPERING_DOCUMENTED")
        records.append({
            "heat_id": str(row["heat_id"]),
            "grade_id": str(row["grade_id"]),
            "steel_family": str(row["steel_family"]),
            "source": str(row["source"]),
            "austenitize_T": observed["austenitize_T"],
            "quench_medium": observed["quench_medium"],
            "temper_T": observed["temper_T"],
            "documented_field_count": documented_count,
            "missing_process_fields": missing,
            "process_status": status,
            "documented_route_evidence": route,
            "provenance": {
                "source_dataset": DATA_FILE.name,
                "source_record_id": str(row["heat_id"]),
                "value_classification": "SOURCE_VALUE",
                "missing_values_imputed": False,
            },
        })
    result = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "process_history_reconstruction.csv"
    json_path = OUTPUT_DIR / "process_history_report.json"
    result.to_csv(csv_path, index=False)
    report = {
        "engine": "MAT-X Process History Reconstruction Engine",
        "version": ENGINE_VERSION,
        "status": "DOCUMENTED_PROCESS_EVIDENCE_ONLY",
        "dataset": DATA_FILE.name,
        "total_records": len(result),
        "unique_heats": int(result["heat_id"].nunique()),
        "process_fields": PROCESS_FIELDS,
        "status_distribution": {
            str(key): int(value)
            for key, value in
            result["process_status"].value_counts().items()
        },
        "field_availability": {
            field: {
                "available": int(result[field].notna().sum()),
                "missing": int(result[field].isna().sum()),
            }
            for field in PROCESS_FIELDS
        },
        "methodology": {
            "approach": "Direct extraction and evidence completeness assessment",
            "missing_values_imputed": False,
            "process_conditions_predicted": False,
            "definitive_process_reconstruction": False,
        },
        "limitations": [
            "Only three process-related fields are available.",
            "Missing conditions are treated as unknown, not absent.",
            "No complete processing sequence or time history is available.",
            "No independent validation split is supplied.",
            "Process-route evidence does not establish the full thermal history.",
        ],
        "outputs": {
            "records_csv": str(csv_path),
            "report_json": str(json_path),
        },
    }
    json_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )
    print("=" * 70)
    print("MAT-X PROCESS HISTORY ENGINE")
    print("=" * 70)
    print("Engine version:", ENGINE_VERSION)
    print("Total records:", len(result))
    print("Unique heats:", result["heat_id"].nunique())
    print("\nPROCESS HISTORY STATUS")
    print(result["process_status"].value_counts().to_string())
    print("\nOUTPUT FILES")
    print(csv_path)
    print(json_path)
    print("\nOriginal dataset modified: NO")
    return report
if __name__ == "__main__":
    reconstruct_process_history()
