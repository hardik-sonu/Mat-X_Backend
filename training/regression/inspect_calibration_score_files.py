from pathlib import Path
import pandas as pd
root = Path("training_results/uncertainty_calibration_v1")
files = [
    root / "tensile_strength_baseline_calibration_grade_scores.csv",
    root / "yield_strength_baseline_calibration_grade_scores.csv",
]
for path in files:
    print("\n" + "=" * 90)
    print("FILE:", path.name)
    print("=" * 90)
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nNumeric column summaries:")
    print(df.describe(include="all").to_string())
print("\nCALIBRATION SCORE FILE INSPECTION COMPLETED")
