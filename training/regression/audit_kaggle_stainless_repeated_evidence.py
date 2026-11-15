from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(r"C:\Users\Ahmadcomputer\MAT-X")
OUT = ROOT / "training_results" / "regression"
df = pd.read_csv(
    OUT / "matx_elongation_source_domain_shift_audit.csv"
)
df = df[df["source"] == "Kaggle_Stainless_Steel"].copy()
features = [
    "C", "Mn", "V",
    "tensile_strength", "yield_strength",
]
df["yield_tensile_ratio"] = (
    df["yield_strength"] / df["tensile_strength"].replace(0, np.nan)
)
print("=" * 90)
print("MAT-X KAGGLE STAINLESS: REPEATED EVIDENCE AUDIT")
print("=" * 90)
# 1. Repeated exact feature combinations
print("\n1. EXACT COMMON-FEATURE COMBINATIONS")
groups = (
    df.groupby(features, dropna=False)
    .agg(
        records=("elongation", "size"),
        distinct_targets=("elongation", "nunique"),
        elongation_min=("elongation", "min"),
        elongation_max=("elongation", "max"),
        elongation_mean=("elongation", "mean"),
        elongation_std=("elongation", "std"),
        grades=("grade_id", "nunique"),
    )
    .reset_index()
)
repeated = groups[groups["records"] > 1].copy()
repeated["target_range"] = (
    repeated["elongation_max"] - repeated["elongation_min"]
)
print("Total records:", len(df))
print("Unique feature combinations:", len(groups))
print("Repeated feature combinations:", len(repeated))
print("Records in repeated combinations:", repeated["records"].sum())
print(
    "Repeated combinations with varying elongation:",
    int((repeated["distinct_targets"] > 1).sum())
)
print("\nRepeated combinations sorted by target range:")
print(
    repeated.sort_values("target_range", ascending=False)
    .to_string(index=False)
)
# 2. Within-grade variation
print("\n2. WITHIN-GRADE ELONGATION VARIATION")
grade_stats = (
    df.groupby("grade_id")
    .agg(
        records=("elongation", "size"),
        distinct_targets=("elongation", "nunique"),
        elongation_min=("elongation", "min"),
        elongation_max=("elongation", "max"),
        elongation_mean=("elongation", "mean"),
        tensile_min=("tensile_strength", "min"),
        tensile_max=("tensile_strength", "max"),
        yield_min=("yield_strength", "min"),
        yield_max=("yield_strength", "max"),
    )
    .reset_index()
)
grade_stats["elongation_range"] = (
    grade_stats["elongation_max"] - grade_stats["elongation_min"]
)
print(grade_stats.sort_values(
    "elongation_range", ascending=False
).to_string(index=False))
# 3. Correlation diagnostic
print("\n3. FEATURE-ELONGATION CORRELATIONS")
print(
    df[features + ["yield_tensile_ratio", "elongation"]]
    .corr(numeric_only=True)["elongation"]
    .sort_values(ascending=False)
    .to_string()
)
# 4. High and low elongation groups
print("\n4. LOW VS HIGH ELONGATION FEATURE DISTRIBUTIONS")
df["elongation_group"] = np.where(
    df["elongation"] >= 35,
    "HIGH_GE_35",
    "LOW_LT_35"
)
print(
    df.groupby("elongation_group")[
        features + ["yield_tensile_ratio", "elongation"]
    ].agg(["count", "mean", "median", "std"]).round(3).to_string()
)
# 5. Export detailed audit
repeated_path = OUT / "matx_kaggle_stainless_repeated_feature_groups.csv"
grade_path = OUT / "matx_kaggle_stainless_grade_variation.csv"
repeated.to_csv(repeated_path, index=False)
grade_stats.to_csv(grade_path, index=False)
print("\nSAVED:")
print(repeated_path)
print(grade_path)
print("\nREPEATED EVIDENCE AUDIT COMPLETED.")
