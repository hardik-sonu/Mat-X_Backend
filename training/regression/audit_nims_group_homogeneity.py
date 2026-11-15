from pathlib import Path
import pandas as pd
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
df = pd.read_csv(DATA)
group = df[df["grade_id"].astype(str) == "NIMS_low_alloy_other"].copy()
print("=" * 85)
print("NIMS_LOW_ALLOY_OTHER — MATERIAL GROUP HOMOGENEITY AUDIT")
print("=" * 85)
print("\nRECORD COUNT:", len(group))
for col in ["source", "steel_family", "grade_id"]:
    if col in group.columns:
        print(f"\n{col.upper()} DISTRIBUTION:")
        print(group[col].value_counts(dropna=False).to_string())
numeric = [
    c for c in [
        "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
        "austenitize_T", "temper_T",
        "tensile_strength", "yield_strength"
    ]
    if c in group.columns
]
print("\nNUMERIC FEATURE SUMMARY:")
print(
    group[numeric]
    .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
    .T.to_string()
)
print("\nMISSINGNESS:")
print(
    group[numeric].isna().mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .to_string()
)
print("\nUNIQUE VALUES PER FEATURE:")
for col in numeric:
    print(f"{col}: {group[col].nunique(dropna=True)}")
print("\nSAMPLE RECORDS:")
print(group[numeric].head(15).to_string(index=False))
print("\nAUDIT COMPLETE — NO SOURCE DATA MODIFIED.")
