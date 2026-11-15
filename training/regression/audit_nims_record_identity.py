from pathlib import Path
import pandas as pd
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
df = pd.read_csv(DATA)
g = df[df["grade_id"].astype(str) == "NIMS_low_alloy_other"].copy()
chemistry = [
    c for c in [
        "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"
    ] if c in g.columns
]
print("=" * 90)
print("NIMS RECORD IDENTITY AND REPEATED-COMPOSITION AUDIT")
print("=" * 90)
print("\nALL DATASET COLUMNS:")
print(df.columns.tolist())
print("\nNIMS NON-CHEMISTRY METADATA:")
metadata = [c for c in g.columns if c not in chemistry]
for col in metadata:
    print(f"{col}: {g[col].nunique(dropna=False)} distinct values")
    if g[col].nunique(dropna=False) <= 15:
        print(g[col].value_counts(dropna=False).to_string())
print("\nEXACT CHEMISTRY PROFILE GROUPS:")
profiles = (
    g.groupby(chemistry, dropna=False)
    .agg(
        record_count=("tensile_strength", "size"),
        unique_tensile_values=("tensile_strength", "nunique"),
        tensile_min=("tensile_strength", "min"),
        tensile_max=("tensile_strength", "max"),
        tensile_mean=("tensile_strength", "mean"),
        tensile_std=("tensile_strength", "std"),
    )
    .reset_index()
)
repeated = profiles[profiles["record_count"] > 1].copy()
print("Total chemistry profiles:", len(profiles))
print("Repeated chemistry profiles:", len(repeated))
print("Records belonging to repeated profiles:",
      repeated["record_count"].sum())
print("\nREPEATED PROFILES WITH LARGEST TENSILE RANGE:")
if len(repeated):
    repeated["tensile_range"] = (
        repeated["tensile_max"] - repeated["tensile_min"]
    )
    print(
        repeated.sort_values(
            ["tensile_range", "record_count"],
            ascending=False
        ).head(20).to_string(index=False)
    )
print("\nPOSSIBLE RECORD IDENTIFIERS:")
for col in g.columns:
    lower = col.lower()
    if any(term in lower for term in [
        "id", "heat", "sample", "specimen", "condition",
        "treatment", "process", "source", "reference"
    ]):
        print(f"\n{col}:")
        print(g[col].head(20).to_string(index=False))
print("\nEXACT DUPLICATE ROWS:")
print("Duplicate rows excluding dataframe index:",
      int(g.duplicated().sum()))
print("\nAUDIT COMPLETE — SOURCE DATA UNCHANGED.")
