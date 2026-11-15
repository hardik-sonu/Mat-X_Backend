from pathlib import Path
import pandas as pd
path = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
df = pd.read_csv(path, low_memory=False, dtype={"grade_id": str})
grades = ["1044", "9310", "1038", "202"]
columns = [
    "grade_id", "source", "steel_family",
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T", "quench_medium",
    "tensile_strength", "yield_strength"
]
available = [c for c in columns if c in df.columns]
for grade in grades:
    subset = df[df["grade_id"].astype(str) == grade].copy()
    print("\n" + "=" * 85)
    print("GRADE:", grade, "| RECORDS:", len(subset))
    print("=" * 85)
    if subset.empty:
        print("NOT FOUND")
        continue
    print(subset[available].to_string(index=False))
    feature_cols = [
        c for c in [
            "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
            "austenitize_T", "temper_T", "quench_medium"
        ] if c in subset.columns
    ]
    print("\nFEATURE COMPLETENESS:")
    print("Available values:", int(subset[feature_cols].notna().sum().sum()))
    print("Possible values:", len(subset) * len(feature_cols))
    print("Missing values:", int(subset[feature_cols].isna().sum().sum()))
print("\nRECORD-LEVEL FEATURE AUDIT COMPLETED")
