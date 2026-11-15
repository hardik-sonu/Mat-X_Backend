from pathlib import Path
import pandas as pd
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
df = pd.read_csv(DATA)
nims = df[
    df["grade_id"].astype(str) == "NIMS_low_alloy_other"
].copy()
print("=" * 90)
print("NIMS HEAT METADATA INVESTIGATION")
print("=" * 90)
print("\nALL DATASET COLUMNS:")
for col in df.columns:
    print(col)
print("\nNIMS RECORD COUNT:", len(nims))
print("UNIQUE HEAT IDS:", nims["heat_id"].nunique())
keywords = [
    "heat", "sample", "specimen", "condition",
    "treatment", "process", "temper", "quench",
    "austen", "cool", "test", "method",
    "temperature", "time", "source", "reference"
]
metadata = [
    col for col in df.columns
    if any(k in col.lower() for k in keywords)
]
print("\nHEAT / PROCESS / TEST RELATED FIELDS:")
for col in metadata:
    values = nims[col].dropna()
    print(f"\n{col}")
    print("  non-null:", len(values))
    print("  unique:", values.nunique())
    print("  examples:", values.unique()[:15].tolist())
print("\nALL NON-CHEMISTRY FIELDS WITH VARIATION:")
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
for col in nims.columns:
    if col in chemistry:
        continue
    unique = nims[col].nunique(dropna=False)
    if unique > 1:
        print(f"\n{col}: {unique} distinct values")
        print(nims[col].value_counts(dropna=False).head(15).to_string())
print("\nINVESTIGATION COMPLETE — NO DATA MODIFIED.")
