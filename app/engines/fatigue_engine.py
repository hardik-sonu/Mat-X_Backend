from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
FATIGUE_DIR = ROOT / "training_results" / "fatigue"
CYCLE_FILE = FATIGUE_DIR / "fatigue_316L_cycle_features.csv"
SUMMARY_FILE = FATIGUE_DIR / "fatigue_316L_specimen_summary.csv"
ASSESSMENT_FILE = FATIGUE_DIR / "fatigue_316L_scientific_assessment.json"
def analyze_fatigue():
    required_files = [CYCLE_FILE, SUMMARY_FILE, ASSESSMENT_FILE]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError({
            "message": "Fatigue assessment files are missing.",
            "missing_files": missing
        })
    cycles = pd.read_csv(CYCLE_FILE)
    specimens = pd.read_csv(SUMMARY_FILE)
    results = []
    for specimen_id, group in cycles.groupby("specimen_id"):
        group = group.sort_values("cycle_count")
        x = group["cycle_count"].to_numpy(dtype=float)
        y = group["stress_amplitude_MPa"].to_numpy(dtype=float)
        slope = None
        if len(group) >= 2 and np.ptp(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
        first = float(y[0])
        last = float(y[-1])
        results.append({
            "specimen_id": specimen_id,
            "geometry": str(group["geometry"].iloc[0]),
            "recorded_cycles": int(len(group)),
            "first_recorded_cycle": int(x[0]),
            "last_recorded_cycle": int(x[-1]),
            "declared_cycles": int(group["declared_cycles"].iloc[0]),
            "initial_stress_amplitude_MPa": first,
            "final_stress_amplitude_MPa": last,
            "stress_amplitude_change_MPa": last - first,
            "stress_amplitude_change_percent": (
                (last - first) / first * 100 if first else None
            ),
            "stress_amplitude_trend_MPa_per_cycle": slope,
            "loop_area_status": "PROVISIONAL_DERIVED_VALUE"
        })
    return {
        "engine": "MAT-X Fatigue Response Analysis Engine",
        "engine_version": "1.0.0",
        "dataset": "316L LPBF low-cycle fatigue",
        "status": "analysis_available",
        "independent_specimens": int(cycles["specimen_id"].nunique()),
        "cycle_feature_records": int(len(cycles)),
        "analysis_type": "DESCRIPTIVE_EXPERIMENTAL_RESPONSE",
        "fatigue_life_prediction": "NOT_SUPPORTED",
        "specimen_results": results,
        "scientific_limitations": [
            "Only seven independent specimens are available.",
            "Recorded cycles are sparsely sampled.",
            "Trend slopes describe recorded observations only.",
            "Hysteresis-loop area remains provisional.",
            "Results are specific to the supplied 316L LPBF dataset."
        ],
        "provenance": {
            "stress_amplitude": "DERIVED_FROM_MEASURED_FORCE_AND_AREA",
            "loop_area": "PROVISIONAL_DERIVED_VALUE",
            "cycle_measurements": "EXPERIMENTAL_SOURCE_DATA"
        }
    }
