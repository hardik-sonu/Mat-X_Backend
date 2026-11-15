from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
SCRIPT = ROOT / "training" / "regression" / "run_grade_grouped_uncertainty_calibration.py"
CSV = DATA / "SteelBench v1.0.csv"
df = pd.read_csv(CSV, low_memory=False)
print("=" * 75)
print("SOURCE-STRATIFIED VALIDATION: PREPARATION AUDIT")
print("=" * 75)
print("\nDataset columns:")
for column in df.columns:
    print(f"  {column}")
print("\nPotential source / grade / heat identifiers:")
for column in df.columns:
    if any(term in column.lower() for term in
           ["source", "dataset", "origin", "grade", "heat", "steel_id"]):
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(15).to_string())
print("\nCalibration script locations:")
lines = SCRIPT.read_text(encoding="utf-8").splitlines()
terms = [
    "prediction_path", "test_indices", "test_idx",
    "test_df", "test_records", "to_csv"
]
for i, line in enumerate(lines):
    if any(term in line for term in terms):
        start = max(0, i - 3)
        end = min(len(lines), i + 5)
        print(f"\n--- Lines {start + 1}-{end} ---")
        for j in range(start, end):
            print(f"{j + 1}: {lines[j]}")
print("\nNo dataset or model artifacts were modified.")
