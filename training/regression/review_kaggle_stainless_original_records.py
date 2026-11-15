from pathlib import Path
import pandas as pd
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\regression")
df = pd.read_csv(DATA)
subset = df[df["source"] == "Kaggle_Stainless_Steel"].copy()
columns = [
    "heat_id", "grade_id", "source", "steel_family",
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "tensile_strength", "yield_strength", "elongation",
    "reduction_area", "impact_J_avg", "hardness",
    "austenitize_T", "quench_medium", "temper_T", "condition"
]
columns = [c for c in columns if c in subset.columns]
print("=" * 100)
print("ORIGINAL KAGGLE STAINLESS STEEL RECORDS: RAW DATA REVIEW")
print("=" * 100)
print("\n1. RECORD COUNT:", len(subset))
print("2. UNIQUE GRADES:", subset["grade_id"].nunique())
print("3. UNIQUE HEAT IDS:", subset["heat_id"].nunique())
print("\n4. ALL 59 ORIGINAL RECORDS")
print(subset[columns].to_string(index=False))
print("\n5. DUPLICATE RECORD CHECK")
print("Duplicate complete rows:", int(subset.duplicated().sum()))
print("Duplicate heat IDs:", int(subset["heat_id"].duplicated().sum()))
print("\n6. ELONGATION FREQUENCY")
print(subset["elongation"].value_counts(dropna=False).sort_index().to_string())
print("\n7. MISSINGNESS")
print(subset[columns].isna().sum().to_string())
path = OUT / "matx_kaggle_stainless_original_records_review.csv"
subset[columns].to_csv(path, index=False)
print("\nORIGINAL RECORD EXPORT:", path)
print("\nREVIEW COMPLETED. NO ORIGINAL DATA MODIFIED.")
