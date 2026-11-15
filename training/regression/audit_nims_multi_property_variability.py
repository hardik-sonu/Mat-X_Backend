from pathlib import Path
import pandas as pd
DATA = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\SteelBench v1.0.csv"
)
df = pd.read_csv(DATA)
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
properties = [
    "tensile_strength",
    "yield_strength",
    "elongation",
    "reduction_area",
    "impact_J_avg",
    "hardness",
]
nims = df[
    df["grade_id"].astype(str) == "NIMS_low_alloy_other"
].copy()
if len(nims) != 81:
    raise SystemExit(f"Expected 81 records, found {len(nims)}")
profiles = []
for profile, group in nims.groupby(chemistry, dropna=False):
    row = dict(zip(chemistry, profile))
    row["heat_ids"] = ", ".join(group["heat_id"].astype(str))
    for prop in properties:
        values = pd.to_numeric(group[prop], errors="coerce").dropna()
        row[f"{prop}_count"] = len(values)
        row[f"{prop}_range"] = (
            float(values.max() - values.min())
            if len(values) > 1 else float("nan")
        )
    profiles.append(row)
result = pd.DataFrame(profiles)
print("=" * 95)
print("NIMS MULTI-PROPERTY VARIABILITY — CORRECTED REPORT")
print("=" * 95)
print("\nPROPERTY AVAILABILITY:")
for prop in properties:
    values = pd.to_numeric(nims[prop], errors="coerce")
    print(
        f"{prop}: available={values.notna().sum()}, "
        f"missing={values.isna().sum()}, "
        f"unique={values.nunique()}"
    )
print("\nWITHIN-PROFILE RANGE SUMMARY:")
for prop in properties:
    ranges = pd.to_numeric(
        result[f"{prop}_range"], errors="coerce"
    ).dropna()
    if len(ranges):
        print(
            f"{prop}: profiles={len(ranges)}, "
            f"median_range={ranges.median():.4f}, "
            f"max_range={ranges.max():.4f}"
        )
    else:
        print(f"{prop}: NO WITHIN-PROFILE RANGE AVAILABLE")
print("\nTOP TENSILE VARIABILITY PROFILES:")
print(
    result.nlargest(10, "tensile_strength_range")[
        ["heat_ids", "tensile_strength_count", "tensile_strength_range"]
    ].to_string(index=False)
)
print("\nOTHER MECHANICAL PROPERTIES:")
for prop in properties[1:]:
    available = result[
        pd.to_numeric(result[f"{prop}_range"], errors="coerce").notna()
    ]
    print(f"\n{prop}:")
    if available.empty:
        print("  No profile has enough measured values to calculate a range.")
    else:
        print(
            available.nlargest(10, f"{prop}_range")[
                ["heat_ids", f"{prop}_count", f"{prop}_range"]
            ].to_string(index=False)
        )
print("\nAUDIT COMPLETE — NO SOURCE DATA OR MODELS MODIFIED.")
