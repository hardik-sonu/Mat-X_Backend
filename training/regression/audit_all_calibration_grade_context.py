from pathlib import Path
import pandas as pd
import numpy as np
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
ROOT = Path("training_results/uncertainty_calibration_v1")
FEATURES = [
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "temper_T", "quench_medium"
]
AUDITS = [
    ("tensile_strength",
     ROOT / "tensile_strength_baseline_calibration_grade_scores.csv"),
    ("yield_strength",
     ROOT / "yield_strength_baseline_calibration_grade_scores.csv"),
]
df = pd.read_csv(DATASET, low_memory=False, dtype={"grade_id": str})
df["grade_id"] = df["grade_id"].astype("string").str.strip()
for target, score_path in AUDITS:
    print("\n" + "=" * 95)
    print("TARGET:", target)
    print("=" * 95)
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=["grade_id", target]).copy()
    if target == "yield_strength":
        uts = pd.to_numeric(work["tensile_strength"], errors="coerce")
        ratio = work[target] / uts.replace(0, np.nan)
        anomaly = (
            work["grade_id"].isin(["EMK0319", "EMK0320", "EMK0321"])
            & (ratio < 0.03)
        )
        work = work.loc[~anomaly].copy()
    features = [c for c in FEATURES if c in work.columns]
    for col in features:
        if col != "quench_medium":
            work[col] = pd.to_numeric(work[col], errors="coerce")
    grouped = work.groupby("grade_id", dropna=False)
    stats = grouped.agg(
        record_count=(target, "size"),
        source_count=("source", "nunique"),
        source_names=("source", lambda x: " ; ".join(
            sorted(x.dropna().astype(str).unique())
        )),
        target_min=(target, "min"),
        target_max=(target, "max"),
        target_std=(target, "std"),
    ).reset_index()
    completeness = grouped[features].apply(
        lambda x: x.notna().sum().sum() / (len(x) * len(features))
        if len(x) and len(features) else np.nan
    ).rename("feature_completeness").reset_index()
    stats = stats.merge(completeness, on="grade_id", how="left")
    scores = pd.read_csv(score_path, dtype={"grade_id": str})
    scores["grade_id"] = scores["grade_id"].astype(str).str.strip()
    scores["maximum_absolute_residual"] = pd.to_numeric(
        scores["maximum_absolute_residual"], errors="coerce"
    )
    audit = scores.merge(stats, on="grade_id", how="left")
    audit["records_per_grade"] = audit["record_count"].fillna(0).astype(int)
    audit = audit.sort_values(
        "maximum_absolute_residual", ascending=False
    )
    print("Calibration grade scores:", len(scores))
    print("Matched dataset grades:", int(audit["record_count"].notna().sum()))
    print("Unmatched calibration grades:", int(audit["record_count"].isna().sum()))
    print("Singleton grades:", int((audit["records_per_grade"] == 1).sum()))
    print("Grades with <= 3 records:", int((audit["records_per_grade"] <= 3).sum()))
    print("Grades spanning multiple sources:", int((audit["source_count"] > 1).sum()))
    print("\nCALIBRATION SCORE DISTRIBUTION BY GROUP SIZE:")
    audit["group_size_band"] = pd.cut(
        audit["records_per_grade"],
        bins=[-1, 1, 3, 10, float("inf")],
        labels=["1 record", "2-3 records", "4-10 records", ">10 records"]
    )
    print(
        audit.groupby("group_size_band", observed=False)
        ["maximum_absolute_residual"]
        .agg(["count", "median", "mean", "max"])
        .to_string()
    )
    print("\nTOP 20 CALIBRATION SCORES WITH DATASET CONTEXT:")
    columns = [
        "grade_id", "maximum_absolute_residual", "record_count",
        "source_count", "source_names", "feature_completeness",
        "target_min", "target_max", "target_std"
    ]
    print(audit[columns].head(20).to_string(index=False))
    print("\nSOURCE DISTRIBUTION OF CALIBRATION GRADES:")
    print(
        audit["source_names"].fillna("UNMATCHED")
        .value_counts().head(15).to_string()
    )
print("\nFULL CALIBRATION-GRADE CONTEXT AUDIT COMPLETED")
