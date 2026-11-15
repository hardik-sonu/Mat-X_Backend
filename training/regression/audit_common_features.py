from pathlib import Path
import pandas as pd
import numpy as np
import json
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = Path.cwd() / "training_results" / "common_feature_audit_v1"
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
FEATURES = ["C", "Mn", "V"]
TARGETS = ["tensile_strength", "yield_strength", "elongation"]
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id"]:
    df[col] = df[col].astype("string").str.strip()
for col in FEATURES + TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
OUT.mkdir(parents=True, exist_ok=True)
rows = []
correlation_rows = []
for source in SOURCES:
    part = df.loc[df["source"] == source]
    for feature in FEATURES:
        values = part[feature].dropna()
        rows.append({
            "source": source,
            "feature": feature,
            "valid_records": len(values),
            "missing_records": int(part[feature].isna().sum()),
            "unique_values": int(values.nunique()),
            "zero_records": int((values == 0).sum()),
            "nonzero_records": int((values != 0).sum()),
            "minimum": values.min() if len(values) else None,
            "median": values.median() if len(values) else None,
            "maximum": values.max() if len(values) else None,
            "standard_deviation": (
                values.std() if len(values) > 1 else None
            ),
            "grade_count": part.loc[
                part[feature].notna(), "grade_id"
            ].nunique(),
        })
    for target in TARGETS:
        valid = part[[*FEATURES, target]].dropna()
        for feature in FEATURES:
            pair = part[[feature, target]].dropna()
            correlation = (
                pair[feature].corr(pair[target])
                if len(pair) >= 3
                and pair[feature].nunique() > 1
                and pair[target].nunique() > 1
                else None
            )
            correlation_rows.append({
                "source": source,
                "feature": feature,
                "target": target,
                "paired_records": len(pair),
                "feature_unique_values": pair[feature].nunique(),
                "target_unique_values": pair[target].nunique(),
                "pearson_correlation": correlation,
            })
summary = pd.DataFrame(rows)
correlations = pd.DataFrame(correlation_rows)
summary.to_csv(OUT / "common_feature_variation.csv", index=False)
correlations.to_csv(
    OUT / "common_feature_target_correlations.csv",
    index=False,
)
print("=" * 80)
print("COMMON FEATURE VARIATION")
print("=" * 80)
print(summary.to_string(index=False))
print("\n" + "=" * 80)
print("FEATURE-TARGET CORRELATIONS (DESCRIPTIVE ONLY)")
print("=" * 80)
print(correlations.to_string(index=False))
print("\n" + "=" * 80)
print("PRELIMINARY FEATURE ELIGIBILITY")
print("=" * 80)
for feature in FEATURES:
    subset = summary[summary["feature"] == feature]
    eligible = (
        len(subset) == len(SOURCES)
        and (subset["missing_records"] == 0).all()
        and (subset["unique_values"] > 1).all()
        and (subset["standard_deviation"].fillna(0) > 0).all()
    )
    print(f"{feature}: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}")
print("\nOutputs:")
for path in sorted(OUT.iterdir()):
    print(path)
print("\nRead-only audit. No dataset or model artifacts modified.")
