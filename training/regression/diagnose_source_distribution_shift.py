from pathlib import Path
import pandas as pd
import numpy as np
import json
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
OUT = (
    Path.cwd()
    / "training_results"
    / "source_distribution_diagnostic_v1"
)
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
NUMERIC_FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T",
]
CATEGORICAL_FEATURES = [
    "quench_medium", "steel_family",
]
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for col in NUMERIC_FEATURES + TARGETS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
OUT.mkdir(parents=True, exist_ok=True)
numeric_rows = []
categorical_rows = []
target_rows = []
missing_rows = []
print("=" * 88)
print("MAT-X SOURCE DISTRIBUTION AND DOMAIN-SHIFT DIAGNOSTIC")
print("=" * 88)
print("\nDATASET COLUMNS:")
print(df.columns.tolist())
for source in SOURCES:
    part = df.loc[df["source"] == source].copy()
    print(
        f"\n{source}: records={len(part)}, "
        f"grades={part['grade_id'].nunique()}, "
        f"heats={part['heat_id'].nunique()}"
    )
    for col in NUMERIC_FEATURES + TARGETS:
        if col not in part.columns:
            continue
        values = pd.to_numeric(part[col], errors="coerce")
        valid = values.dropna()
        row = {
            "source": source,
            "feature": col,
            "records": len(part),
            "valid_count": len(valid),
            "missing_count": int(values.isna().sum()),
            "missing_fraction": float(values.isna().mean()),
            "minimum": valid.min() if len(valid) else None,
            "q05": valid.quantile(0.05) if len(valid) else None,
            "median": valid.median() if len(valid) else None,
            "q95": valid.quantile(0.95) if len(valid) else None,
            "maximum": valid.max() if len(valid) else None,
            "mean": valid.mean() if len(valid) else None,
            "std": valid.std() if len(valid) > 1 else None,
        }
        if col in TARGETS:
            target_rows.append(row)
        else:
            numeric_rows.append(row)
    for col in CATEGORICAL_FEATURES:
        if col not in part.columns:
            continue
        values = part[col].astype("string").str.strip()
        counts = values.value_counts(dropna=False)
        categorical_rows.append({
            "source": source,
            "feature": col,
            "valid_count": int(values.notna().sum()),
            "missing_count": int(values.isna().sum()),
            "unique_values": int(values.nunique(dropna=True)),
            "top_values": json.dumps(
                {
                    str(k): int(v)
                    for k, v in counts.head(12).items()
                },
                ensure_ascii=False,
            ),
        })
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS:
        if col not in part.columns:
            continue
        missing_rows.append({
            "source": source,
            "feature": col,
            "missing_count": int(part[col].isna().sum()),
            "missing_fraction": float(part[col].isna().mean()),
        })
numeric_df = pd.DataFrame(numeric_rows)
categorical_df = pd.DataFrame(categorical_rows)
target_df = pd.DataFrame(target_rows)
missing_df = pd.DataFrame(missing_rows)
numeric_df.to_csv(
    OUT / "numeric_feature_distributions.csv",
    index=False,
)
categorical_df.to_csv(
    OUT / "categorical_feature_distributions.csv",
    index=False,
)
target_df.to_csv(
    OUT / "target_distributions.csv",
    index=False,
)
missing_df.to_csv(
    OUT / "source_feature_missingness.csv",
    index=False,
)
# Compare source medians and ranges without combining source populations.
comparison_rows = []
for feature in NUMERIC_FEATURES + TARGETS:
    if feature not in df.columns:
        continue
    row = {"feature": feature}
    for source in SOURCES:
        values = pd.to_numeric(
            df.loc[df["source"] == source, feature],
            errors="coerce",
        ).dropna()
        row[f"{source}_n"] = len(values)
        row[f"{source}_median"] = (
            float(values.median()) if len(values) else None
        )
        row[f"{source}_q05"] = (
            float(values.quantile(0.05)) if len(values) else None
        )
        row[f"{source}_q95"] = (
            float(values.quantile(0.95)) if len(values) else None
        )
    comparison_rows.append(row)
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(
    OUT / "cross_source_numeric_comparison.csv",
    index=False,
)
print("\n" + "=" * 88)
print("TARGET DISTRIBUTIONS")
print("=" * 88)
if not target_df.empty:
    print(
        target_df[
            [
                "source", "feature", "valid_count",
                "missing_fraction", "minimum", "q05",
                "median", "q95", "maximum",
            ]
        ].to_string(index=False)
    )
print("\n" + "=" * 88)
print("NUMERIC INPUT FEATURE MEDIANS BY SOURCE")
print("=" * 88)
if not numeric_df.empty:
    print(
        numeric_df[
            ["source", "feature", "valid_count",
             "missing_fraction", "median"]
        ].to_string(index=False)
    )
print("\n" + "=" * 88)
print("CATEGORICAL FEATURE DISTRIBUTIONS")
print("=" * 88)
if not categorical_df.empty:
    print(categorical_df.to_string(index=False))
print("\n" + "=" * 88)
print("OUTPUT FILES")
print("=" * 88)
for path in sorted(OUT.iterdir()):
    print(path)
print("\nREAD-ONLY DIAGNOSTIC COMPLETE.")
print("No source datasets or existing models were modified.")
