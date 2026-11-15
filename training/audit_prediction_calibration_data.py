from pathlib import Path
import pandas as pd
import json
ROOT = Path.cwd()
REG = ROOT / "training_results" / "regression"
print("=" * 80)
print("MAT-X PREDICTION AND CALIBRATION DATA AUDIT")
print("=" * 80)
files = [
    "steelbench_tensile_test_predictions.csv",
    "steelbench_yield_test_predictions.csv",
    "matx_elongation_test_predictions.csv",
    "matx_reduction_area_test_predictions.csv",
    "matx_impact_J_avg_test_predictions.csv",
    "matx_elongation_evidence_enriched_test_predictions.csv",
    "matx_reduction_area_evidence_enriched_test_predictions.csv",
    "matx_impact_J_avg_evidence_enriched_test_predictions.csv",
]
for name in files:
    path = REG / name
    if not path.exists():
        print(f"\nNOT FOUND: {name}")
        continue
    print(f"\nFILE: {name}")
    try:
        df = pd.read_csv(path)
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("First 2 records:")
        print(df.head(2).to_string(index=False))
    except Exception as exc:
        print("ERROR:", repr(exc))
print("\n" + "=" * 80)
print("MODEL REPORT STRUCTURES")
print("=" * 80)
for path in sorted(REG.glob("*report.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n{path.name}")
        print("Top-level keys:", list(data.keys()))
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                print(f"  {key}: {value}")
    except Exception as exc:
        print(path.name, "ERROR:", repr(exc))
print("\nAUDIT COMPLETE")
