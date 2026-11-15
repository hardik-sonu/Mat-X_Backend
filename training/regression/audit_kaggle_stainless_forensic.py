from pathlib import Path
import json
import numpy as np
import pandas as pd
DATA = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
OUT = Path(r"C:\Users\Ahmadcomputer\MAT-X\training_results\regression")
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA)
raw = df.loc[df["source"].eq("Kaggle_Stainless_Steel")].copy()
if len(raw) != 59:
    raise RuntimeError(f"Expected 59 Kaggle stainless records; found {len(raw)}")
numeric_cols = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "tensile_strength", "yield_strength", "elongation",
    "reduction_area", "impact_J_avg", "austenitize_T", "temper_T"
]
for col in numeric_cols:
    if col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
raw["yield_tensile_ratio"] = (
    raw["yield_strength"] / raw["tensile_strength"].replace(0, np.nan)
)
raw["strength_difference_UTS_minus_YS"] = (
    raw["tensile_strength"] - raw["yield_strength"]
)
# Keep hardness as raw evidence. Numeric conversion is exploratory only.
raw["hardness_raw"] = raw["hardness"].astype("string")
raw["hardness_numeric_exploratory"] = pd.to_numeric(
    raw["hardness"], errors="coerce"
)
chemistry = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
mechanical = [
    "tensile_strength", "yield_strength", "elongation",
    "yield_tensile_ratio", "hardness_numeric_exploratory"
]
# 1. Row-level physical and data-quality checks.
row_flags = []
for _, r in raw.iterrows():
    flags = []
    uts = r.get("tensile_strength")
    ys = r.get("yield_strength")
    elong = r.get("elongation")
    ratio = r.get("yield_tensile_ratio")
    if pd.isna(uts) or pd.isna(ys) or pd.isna(elong):
        flags.append("MISSING_CORE_MECHANICAL_PROPERTY")
    if pd.notna(uts) and uts <= 0:
        flags.append("NONPOSITIVE_TENSILE_STRENGTH")
    if pd.notna(ys) and ys <= 0:
        flags.append("NONPOSITIVE_YIELD_STRENGTH")
    if pd.notna(uts) and pd.notna(ys) and ys > uts:
        flags.append("YIELD_STRENGTH_EXCEEDS_TENSILE_STRENGTH_REVIEW")
    if pd.notna(elong) and (elong <= 0 or elong > 100):
        flags.append("ELONGATION_OUTSIDE_0_TO_100_PERCENT")
    if pd.notna(ratio) and (ratio <= 0 or ratio > 1):
        flags.append("YIELD_TENSILE_RATIO_OUTSIDE_0_TO_1_REVIEW")
    if pd.notna(ratio) and ratio >= 0.95:
        flags.append("VERY_HIGH_YIELD_TENSILE_RATIO_REVIEW")
    if pd.notna(uts) and uts >= 1500:
        flags.append("VERY_HIGH_TENSILE_STRENGTH_SOURCE_CHECK")
    if pd.notna(ys) and ys >= 1400:
        flags.append("VERY_HIGH_YIELD_STRENGTH_SOURCE_CHECK")
    if pd.notna(elong) and elong <= 2:
        flags.append("VERY_LOW_ELONGATION_SOURCE_CHECK")
    if pd.isna(r.get("Mo")):
        flags.append("MISSING_MOLYBDENUM")
    if pd.isna(r.get("Ni")):
        flags.append("MISSING_NICKEL")
    if pd.isna(r.get("hardness_numeric_exploratory")):
        flags.append("HARDNESS_MISSING_OR_NONNUMERIC")
    row_flags.append({
        "heat_id": r["heat_id"],
        "grade_id": r["grade_id"],
        "elongation": elong,
        "tensile_strength": uts,
        "yield_strength": ys,
        "yield_tensile_ratio": ratio,
        "flags": ";".join(flags) if flags else "NO_AUTOMATED_FLAG"
    })
row_flags_df = pd.DataFrame(row_flags)
# 2. Grade-level chemistry and mechanical variability.
grade_rows = []
for grade, g in raw.groupby("grade_id", dropna=False):
    item = {
        "grade_id": grade,
        "record_count": len(g),
        "unique_chemistry_signatures": g[chemistry].drop_duplicates().shape[0],
    }
    for col in chemistry + mechanical:
        values = g[col].dropna() if col in g.columns else pd.Series(dtype=float)
        item[f"{col}_n"] = int(values.count())
        item[f"{col}_min"] = float(values.min()) if len(values) else None
        item[f"{col}_max"] = float(values.max()) if len(values) else None
        item[f"{col}_range"] = (
            float(values.max() - values.min()) if len(values) else None
        )
    if len(g) >= 2:
        item["elongation_range"] = float(g["elongation"].max() - g["elongation"].min())
        item["uts_range"] = float(g["tensile_strength"].max() - g["tensile_strength"].min())
        item["ys_range"] = float(g["yield_strength"].max() - g["yield_strength"].min())
        item["chemistry_identical_across_records"] = (
            g[chemistry].nunique(dropna=False).max() == 1
        )
    else:
        item["elongation_range"] = 0.0
        item["uts_range"] = 0.0
        item["ys_range"] = 0.0
        item["chemistry_identical_across_records"] = True
    grade_rows.append(item)
grade_summary = pd.DataFrame(grade_rows).sort_values(
    ["record_count", "elongation_range"], ascending=[False, False]
)
# 3. Pairwise within-grade comparisons.
pair_rows = []
for grade, g in raw.groupby("grade_id"):
    if len(g) < 2:
        continue
    records = g.sort_values("heat_id").to_dict("records")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            pair = {
                "grade_id": grade,
                "heat_id_A": a["heat_id"],
                "heat_id_B": b["heat_id"],
            }
            for col in chemistry + [
                "tensile_strength", "yield_strength", "elongation",
                "yield_tensile_ratio", "hardness_numeric_exploratory"
            ]:
                av, bv = a.get(col), b.get(col)
                pair[f"{col}_A"] = av
                pair[f"{col}_B"] = bv
                pair[f"delta_{col}_B_minus_A"] = (
                    float(bv - av)
                    if pd.notna(av) and pd.notna(bv) else None
                )
            # Directional signature only; it cannot establish cold work.
            d_uts = pair["delta_tensile_strength_B_minus_A"]
            d_ys = pair["delta_yield_strength_B_minus_A"]
            d_el = pair["delta_elongation_B_minus_A"]
            if d_uts is not None and d_ys is not None and d_el is not None:
                if d_uts > 0 and d_ys > 0 and d_el < 0:
                    pair["directional_signature"] = (
                        "STRENGTH_UP_DUCTILITY_DOWN_POSSIBLE_PROCESSING_EFFECT"
                    )
                elif d_uts < 0 and d_ys < 0 and d_el > 0:
                    pair["directional_signature"] = (
                        "STRENGTH_DOWN_DUCTILITY_UP_POSSIBLE_PROCESSING_EFFECT"
                    )
                else:
                    pair["directional_signature"] = "NO_SIMPLE_DIRECTIONAL_PATTERN"
            else:
                pair["directional_signature"] = "INSUFFICIENT_PAIRED_VALUES"
            pair_rows.append(pair)
pair_df = pd.DataFrame(pair_rows)
# 4. Correlation analysis: overall and repeated grades only.
correlation_features = [
    "C", "Mn", "Cr", "Ni", "Si",
    "tensile_strength", "yield_strength",
    "yield_tensile_ratio", "hardness_numeric_exploratory", "elongation"
]
overall_corr = raw[correlation_features].corr(method="spearman")
within_grade_corr = []
for grade, g in raw.groupby("grade_id"):
    if len(g) < 3:
        continue
    for feature in [
        "tensile_strength", "yield_strength",
        "yield_tensile_ratio", "hardness_numeric_exploratory"
    ]:
        valid = g[[feature, "elongation"]].dropna()
        if len(valid) >= 3 and valid[feature].nunique() > 1 and valid["elongation"].nunique() > 1:
            within_grade_corr.append({
                "grade_id": grade,
                "n": len(valid),
                "feature": feature,
                "spearman_rho": float(
                    valid[feature].corr(valid["elongation"], method="spearman")
                )
            })
within_corr_df = pd.DataFrame(within_grade_corr)
# 5. Chemistry consistency flags.
# Thresholds are screening thresholds, not material specification limits.
chem_thresholds = {
    "C": 0.10, "Mn": 0.50, "Cr": 2.00, "Mo": 1.00,
    "Ni": 2.00, "Si": 1.00, "V": 0.10, "Cu": 0.50, "Al": 0.50
}
chem_flags = []
for grade, g in raw.groupby("grade_id"):
    for element, threshold in chem_thresholds.items():
        vals = g[element].dropna()
        if len(vals) >= 2:
            spread = float(vals.max() - vals.min())
            if spread > threshold:
                chem_flags.append({
                    "grade_id": grade,
                    "element": element,
                    "n_measured": len(vals),
                    "minimum": float(vals.min()),
                    "maximum": float(vals.max()),
                    "range": spread,
                    "screening_threshold": threshold,
                    "assessment": "WITHIN_GRADE_CHEMISTRY_SPREAD_REVIEW"
                })
chem_flags_df = pd.DataFrame(chem_flags)
# 6. Data-level summary and cautious interpretations.
summary = {
    "audit": "Kaggle Stainless Steel focused forensic audit",
    "record_count": int(len(raw)),
    "unique_grades": int(raw["grade_id"].nunique()),
    "unique_heat_ids": int(raw["heat_id"].nunique()),
    "duplicate_complete_rows": int(raw.duplicated().sum()),
    "duplicate_heat_ids": int(raw["heat_id"].duplicated().sum()),
    "missing_counts": {
        col: int(raw[col].isna().sum())
        for col in chemistry + [
            "tensile_strength", "yield_strength", "elongation",
            "hardness", "austenitize_T", "quench_medium", "temper_T", "condition"
        ] if col in raw.columns
    },
    "physical_screening": {
        "yield_strength_exceeds_tensile_strength": int(
            (raw["yield_strength"] > raw["tensile_strength"]).sum()
        ),
        "nonpositive_strength_records": int(
            ((raw["tensile_strength"] <= 0) | (raw["yield_strength"] <= 0)).sum()
        ),
        "elongation_outside_0_to_100": int(
            ((raw["elongation"] <= 0) | (raw["elongation"] > 100)).sum()
        ),
        "yield_tensile_ratio_above_0_95": int(
            (raw["yield_tensile_ratio"] >= 0.95).sum()
        ),
        "very_high_tensile_strength_ge_1500": int(
            (raw["tensile_strength"] >= 1500).sum()
        ),
        "very_low_elongation_le_2": int(
            (raw["elongation"] <= 2).sum()
        )
    },
    "correlation_spearman": overall_corr.where(
        pd.notna(overall_corr), None
    ).to_dict(),
    "processing_history": {
        "austenitize_T_recorded": int(raw["austenitize_T"].notna().sum()),
        "quench_medium_recorded": int(raw["quench_medium"].notna().sum()),
        "temper_T_recorded": int(raw["temper_T"].notna().sum()),
        "condition_recorded": int(raw["condition"].notna().sum()),
        "interpretation": (
            "No processing history is recorded in these 59 rows. "
            "Cold work or heat treatment cannot be confirmed from this dataset."
        )
    },
    "hardness_caution": (
        "Hardness is retained as raw source data and used only exploratorily. "
        "Its scale/unit is not established; do not compare or convert it as a verified common scale."
    ),
    "source_documentation_status": (
        "This CSV alone does not establish original test standard, specimen orientation, "
        "product form, heat-treatment condition, hardness scale, or unit provenance. "
        "These require the source dataset documentation or original publication."
    ),
    "decision": (
        "No records deleted, corrected, imputed, or reclassified. "
        "All automated flags are review prompts, not confirmed errors."
    )
}
# Save artifacts.
raw_export = raw.copy()
raw_export.to_csv(OUT / "matx_kaggle_stainless_forensic_records.csv", index=False)
row_flags_df.to_csv(OUT / "matx_kaggle_stainless_forensic_row_flags.csv", index=False)
grade_summary.to_csv(OUT / "matx_kaggle_stainless_forensic_grade_summary.csv", index=False)
pair_df.to_csv(OUT / "matx_kaggle_stainless_forensic_pair_comparisons.csv", index=False)
chem_flags_df.to_csv(OUT / "matx_kaggle_stainless_forensic_chemistry_flags.csv", index=False)
within_corr_df.to_csv(OUT / "matx_kaggle_stainless_forensic_within_grade_correlations.csv", index=False)
overall_corr.to_csv(OUT / "matx_kaggle_stainless_forensic_overall_correlations.csv")
(OUT / "matx_kaggle_stainless_forensic_report.json").write_text(
    json.dumps(summary, indent=2, default=str), encoding="utf-8"
)
print("=" * 90)
print("MAT-X: KAGGLE STAINLESS STEEL FORENSIC AUDIT")
print("=" * 90)
print("\n1. DATASET INTEGRITY")
print("Records:", len(raw))
print("Grades:", raw["grade_id"].nunique())
print("Duplicate complete rows:", raw.duplicated().sum())
print("Duplicate heat IDs:", raw["heat_id"].duplicated().sum())
print("\n2. PHYSICAL SCREENING")
for key, value in summary["physical_screening"].items():
    print(f"{key}: {value}")
print("\n3. OVERALL SPEARMAN CORRELATIONS WITH ELONGATION")
print(overall_corr["elongation"].sort_values().to_string())
print("\n4. REPEATED-GRADE VARIATION")
display_cols = [
    "grade_id", "record_count", "unique_chemistry_signatures",
    "C_range", "Mn_range", "Cr_range", "Ni_range",
    "tensile_strength_range", "yield_strength_range",
    "elongation_range", "hardness_numeric_exploratory_range",
    "chemistry_identical_across_records"
]
display_cols = [c for c in display_cols if c in grade_summary.columns]
print(grade_summary[display_cols].to_string(index=False))
print("\n5. WITHIN-GRADE CHEMISTRY SPREAD FLAGS")
if len(chem_flags_df):
    print(chem_flags_df.to_string(index=False))
else:
    print("No chemistry spread exceeded the screening thresholds.")
print("\n6. WITHIN-GRADE STRENGTH-UP / DUCTILITY-DOWN PAIRS")
if len(pair_df):
    flagged = pair_df[
        pair_df["directional_signature"].str.startswith("STRENGTH_UP")
    ]
    print("Directional pairs:", len(flagged))
    if len(flagged):
        print(flagged[[
            "grade_id", "heat_id_A", "heat_id_B",
            "delta_tensile_strength_B_minus_A",
            "delta_yield_strength_B_minus_A",
            "delta_elongation_B_minus_A",
            "directional_signature"
        ]].to_string(index=False))
else:
    print("No repeated-grade pairs available.")
print("\n7. WITHIN-GRADE CORRELATIONS")
if len(within_corr_df):
    print(within_corr_df.to_string(index=False))
else:
    print("Insufficient repeated-grade observations for within-grade correlations.")
print("\n8. PROCESS HISTORY AVAILABILITY")
print(summary["processing_history"])
print("\n9. HARDNESS CAUTION")
print(summary["hardness_caution"])
print("\n10. SOURCE DOCUMENTATION LIMITATION")
print(summary["source_documentation_status"])
print("\n11. REPORTS SAVED")
for name in [
    "matx_kaggle_stainless_forensic_records.csv",
    "matx_kaggle_stainless_forensic_row_flags.csv",
    "matx_kaggle_stainless_forensic_grade_summary.csv",
    "matx_kaggle_stainless_forensic_pair_comparisons.csv",
    "matx_kaggle_stainless_forensic_chemistry_flags.csv",
    "matx_kaggle_stainless_forensic_within_grade_correlations.csv",
    "matx_kaggle_stainless_forensic_overall_correlations.csv",
    "matx_kaggle_stainless_forensic_report.json"
]:
    print(OUT / name)
print("\nAUDIT COMPLETE. ORIGINAL DATASET AND MODELS WERE NOT MODIFIED.")
