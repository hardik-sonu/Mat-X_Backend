from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = ROOT / "training_results" / "regression"
pred = pd.read_csv(
    OUT / "matx_elongation_feature_ablation_predictions.csv"
)
raw = pd.read_csv(DATA)
keys = ["heat_id", "grade_id", "source"]
features = [
    "C", "Mn", "V",
    "tensile_strength", "yield_strength",
]
required = keys + ["elongation", "steel_family"] + features
missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")
audit = pred[keys].merge(
    raw[required],
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True,
)
if not (audit["_merge"] == "both").all():
    raise ValueError("Some prediction records did not match original dataset.")
audit.drop(columns="_merge", inplace=True)
audit["yield_tensile_ratio"] = (
    audit["yield_strength"] / audit["tensile_strength"].replace(0, np.nan)
)
print("=" * 90)
print("MAT-X SOURCE DISTRIBUTION AND DOMAIN SHIFT AUDIT")
print("=" * 90)
print("\n1. ELONGATION TARGET DISTRIBUTION BY SOURCE")
print(
    audit.groupby("source")["elongation"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3).to_string()
)
print("\n2. ELONGATION TARGET DISTRIBUTION BY SOURCE AND FAMILY")
print(
    audit.groupby(["source", "steel_family"], dropna=False)["elongation"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(3).to_string()
)
print("\n3. COMMON FEATURE DISTRIBUTION BY SOURCE")
for source, group in audit.groupby("source"):
    print(f"\nSOURCE: {source} | N={len(group)}")
    print(
        group[features + ["yield_tensile_ratio"]]
        .describe(percentiles=[.10, .25, .50, .75, .90])
        .round(3).to_string()
    )
print("\n4. FEATURE MISSINGNESS BY SOURCE (%)")
missingness = (
    audit.groupby("source")[features + ["yield_tensile_ratio"]]
    .apply(lambda x: x.isna().mean() * 100, include_groups=False)
)
print(missingness.round(2).to_string())
print("\n5. SOURCE AND FAMILY RECORD COUNTS")
print(
    pd.crosstab(
        audit["source"],
        audit["steel_family"],
        margins=True,
    ).to_string()
)
print("\n6. HELD-OUT STAINLESS RECORDS WITH HIGHEST ELONGATION")
stainless = audit[
    audit["source"] == "Kaggle_Stainless_Steel"
].sort_values("elongation", ascending=False)
print(
    stainless[
        keys + ["steel_family", "elongation"] + features
    ].head(20).to_string(index=False)
)
output = OUT / "matx_elongation_source_domain_shift_audit.csv"
audit.to_csv(output, index=False)
print(f"\nAUDIT DATA SAVED: {output}")
print("\nSOURCE DOMAIN SHIFT AUDIT COMPLETED.")
