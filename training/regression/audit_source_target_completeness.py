from pathlib import Path
import pandas as pd
import numpy as np
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
    "reduction_area",
    "impact_J_avg",
]
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for target in TARGETS:
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
print("=" * 90)
print("MAT-X TARGET COMPLETENESS BY SOURCE")
print("=" * 90)
for target in TARGETS:
    if target not in df.columns:
        print(f"\nTARGET ABSENT: {target}")
        continue
    work = df.dropna(
        subset=[target, "source", "grade_id", "heat_id"]
    ).copy()
    if target == "yield_strength":
        work["tensile_strength"] = pd.to_numeric(
            work["tensile_strength"], errors="coerce"
        )
        ratio = (
            work[target]
            / work["tensile_strength"].replace(0, np.nan)
        )
        anomaly = (
            work["grade_id"].astype(str).isin(
                ["EMK0319", "EMK0320", "EMK0321"]
            )
            & (ratio < 0.03)
        )
        print(
            f"\nYield anomaly exclusions: {int(anomaly.sum())}"
        )
        work = work.loc[~anomaly].copy()
    print(f"\n{'=' * 90}")
    print(f"TARGET: {target}")
    print("=" * 90)
    rows = []
    for source in SOURCES:
        part = work.loc[work["source"] == source]
        rows.append({
            "source": source,
            "valid_records": len(part),
            "valid_grades": part["grade_id"].nunique(),
            "valid_heats": part["heat_id"].nunique(),
            "target_min": part[target].min(),
            "target_median": part[target].median(),
            "target_max": part[target].max(),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPossible held-out folds:")
    for held_out in SOURCES:
        test = work.loc[work["source"] == held_out]
        remaining = work.loc[work["source"] != held_out]
        print(
            f"  Hold out {held_out:25s} "
            f"| test records={len(test):4d} "
            f"| test grades={test['grade_id'].nunique():4d} "
            f"| remaining records={len(remaining):4d} "
            f"| remaining grades={remaining['grade_id'].nunique():4d} "
            f"| test available={'YES' if len(test) else 'NO'}"
        )
print("\n" + "=" * 90)
print("AUDIT COMPLETE — NO DATA OR MODEL FILES MODIFIED")
print("=" * 90)
