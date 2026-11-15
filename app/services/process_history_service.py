import json
import os
from pathlib import Path
import pandas as pd
from app.engines.process_history_validation import validate_process_history_frame
from app.engines.process_history_engine import reconstruct_process_history
from app.engines.process_history_analysis import analyze_process_history
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(os.getenv(
    "MATX_PROCESS_HISTORY_OUTPUT_DIR",
    str(ROOT / "training_results" / "process_history")
))
def get_process_history():
    return reconstruct_process_history()
def get_process_history_analysis():
    # Rebuild source evidence and analysis outputs before serving results.
    reconstruct_process_history()
    analyze_process_history()
    report_path = OUTPUT / "process_history_pattern_report.json"
    route_path = OUTPUT / "process_history_route_summary.csv"
    complete_path = OUTPUT / "process_history_complete_routes.csv"
    family_path = OUTPUT / "process_history_family_summary.csv"
    grade_path = OUTPUT / "process_history_grade_summary.csv"
    required_files = [
        report_path, route_path, complete_path,
        family_path, grade_path
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Process history analysis outputs missing: " + ", ".join(missing)
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    routes = pd.read_csv(route_path)
    complete_routes = pd.read_csv(complete_path)
    families = pd.read_csv(family_path)
    grades = pd.read_csv(grade_path)
    return {
        **report,
        "top_observed_process_combinations": (
            routes.head(20).to_dict(orient="records")
        ),
        "complete_documented_routes": (
            complete_routes.head(50).to_dict(orient="records")
        ),
        "steel_family_summary": (
            families.to_dict(orient="records")
        ),
        "grade_summary": (
            grades.head(100).to_dict(orient="records")
        ),
        "returned_record_limits": {
            "observed_combinations": 20,
            "complete_routes": 50,
            "grade_records": 100,
            "family_records": len(families),
        },
    }
def _load_process_records():
    path = OUTPUT / "process_history_reconstruction.csv"
    if not path.exists():
        reconstruct_process_history()
    if not path.exists():
        raise FileNotFoundError(
            "Process history reconstruction CSV could not be generated."
        )
    records = pd.read_csv(path)
    return validate_process_history_frame(
        records,
        require_generated=True,
    )
def get_heat_process_history(heat_id: str):
    df = _load_process_records()
    matches = df[df["heat_id"].astype(str) == str(heat_id)]
    if matches.empty:
        return None
    records = matches.astype(object).where(
        pd.notna(matches), None
    ).to_dict(orient="records")
    return {
        "engine": "MAT-X Process History Reconstruction Engine",
        "status": "DOCUMENTED_PROCESS_EVIDENCE_ONLY",
        "lookup_type": "HEAT_ID",
        "heat_id": str(heat_id),
        "record_count": len(records),
        "records": records,
        "limitations": [
            "Only documented process fields are returned.",
            "Missing values are not imputed.",
            "This record does not establish the complete thermal history.",
        ],
    }
def get_grade_process_history(grade_id: str):
    df = _load_process_records()
    matches = df[df["grade_id"].astype(str) == str(grade_id)]
    if matches.empty:
        return None
    records = matches.astype(object).where(
        pd.notna(matches), None
    ).to_dict(orient="records")
    return {
        "engine": "MAT-X Process History Reconstruction Engine",
        "status": "DOCUMENTED_PROCESS_EVIDENCE_ONLY",
        "lookup_type": "GRADE_ID",
        "grade_id": str(grade_id),
        "record_count": len(records),
        "records": records,
        "limitations": [
            "Records are dataset observations, not universal grade recipes.",
            "Missing values are not imputed.",
            "Different heats may have different documented histories.",
        ],
    }

