from pathlib import Path
import pandas as pd
root = Path("training_results/uncertainty_calibration_v1")
files = [
    root / "tensile_strength_baseline_calibration_grade_scores.csv",
    root / "yield_strength_baseline_calibration_grade_scores.csv",
]
for path in files:
    print("\n" + "=" * 75)
    print(path.name)
    print("=" * 75)
    df = pd.read_csv(path)
    col = "maximum_absolute_residual"
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col, ascending=False)
    print("Calibration grades:", len(df))
    print("\nTOP 15 HIGHEST GRADE-LEVEL RESIDUALS:")
    print(df[["grade_id", col]].head(15).to_string(index=False))
    print("\nQUANTILE-NEIGHBORHOOD:")
    n = len(df)
    for confidence in (0.90, 0.95):
        rank = int(__import__("math").ceil((n + 1) * confidence))
        ascending = df.sort_values(col).reset_index(drop=True)
        print(f"\n{confidence:.0%} rank from ascending order: {rank}")
        print(
            ascending.iloc[max(0, rank - 3):min(n, rank + 2)]
            [["grade_id", col]]
            .to_string(index=False)
        )
print("\nINFLUENTIAL GRADE AUDIT COMPLETED")
