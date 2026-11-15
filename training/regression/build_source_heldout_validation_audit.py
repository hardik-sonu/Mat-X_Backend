from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT = Path.cwd()
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
EXPERIMENT = (
    ROOT / "training_results" / "source_heldout_validation_v1"
)
SUMMARY = EXPERIMENT / "experiment_summary.json"
OUT = (
    ROOT / "training_results" / "source_heldout_validation_audit_v1"
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
    "reduction_area",
    "impact_J_avg",
]
VARIANTS = {
    "tensile_strength": ["baseline"],
    "yield_strength": ["baseline"],
    "elongation": ["baseline", "evidence_enriched"],
    "reduction_area": ["baseline", "evidence_enriched"],
    "impact_J_avg": ["baseline", "evidence_enriched"],
}
df = pd.read_csv(DATASET, low_memory=False)
for col in ["source", "grade_id", "heat_id"]:
    df[col] = df[col].astype("string").str.strip()
for target in TARGETS:
    if target in df.columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
experiment = json.loads(SUMMARY.read_text(encoding="utf-8"))
successful = {
    (
        r["target"],
        r["variant"],
        r["held_out_source"],
    ): r
    for r in experiment["results"]
}
failed = {
    (
        r["target"],
        r["variant"],
        r["held_out_source"],
    ): r
    for r in experiment["failures"]
}
OUT.mkdir(parents=True, exist_ok=True)
availability_rows = []
matrix_rows = []
for target in TARGETS:
    if target not in df.columns:
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
        work = work.loc[~anomaly].copy()
    for source in SOURCES:
        part = work.loc[work["source"] == source]
        availability_rows.append({
            "target": target,
            "source": source,
            "valid_records": len(part),
            "valid_grades": part["grade_id"].nunique(),
            "valid_heats": part["heat_id"].nunique(),
        })
    for held_out in SOURCES:
        test = work.loc[work["source"] == held_out]
        remaining = work.loc[work["source"] != held_out]
        if len(test) == 0:
            structural_status = "NOT_EVALUABLE_NO_TEST_LABELS"
        elif len(remaining) == 0:
            structural_status = "NOT_EVALUABLE_NO_TRAINING_LABELS"
        elif remaining["grade_id"].nunique() < 5:
            structural_status = "NOT_EVALUABLE_INSUFFICIENT_GROUPS"
        else:
            structural_status = "DATA_AVAILABLE"
        for variant in VARIANTS[target]:
            key = (target, variant, held_out)
            if key in successful:
                report = successful[key]
                status = "COMPLETED"
                reason = "Source-held-out evaluation completed."
            elif structural_status != "DATA_AVAILABLE":
                status = structural_status
                reason = (
                    f"Valid test records={len(test)}; "
                    f"remaining training records={len(remaining)}; "
                    f"remaining grades="
                    f"{remaining['grade_id'].nunique()}."
                )
            elif key in failed:
                status = "EXECUTION_FAILED_REQUIRES_REVIEW"
                reason = failed[key].get("reason", "Unknown failure.")
            else:
                status = "NOT_RUN"
                reason = "No matching result or failure entry."
            result = successful.get(key, {})
            row = {
                "target": target,
                "variant": variant,
                "held_out_source": held_out,
                "status": status,
                "test_records_available": len(test),
                "test_grades_available": test["grade_id"].nunique(),
                "remaining_training_records": len(remaining),
                "remaining_training_grades": (
                    remaining["grade_id"].nunique()
                ),
                "completed_test_records": result.get("test_records"),
                "completed_test_grades": result.get("test_grades"),
                "mae": result.get("mae"),
                "rmse": result.get("rmse"),
                "r2": result.get("r2"),
                "reason": reason,
            }
            if result:
                for level in ["90", "95"]:
                    interval = result["intervals"][level]
                    row[f"record_coverage_{level}"] = (
                        interval["record_coverage"]
                    )
                    row[f"whole_grade_coverage_{level}"] = (
                        interval["whole_grade_coverage"]
                    )
                    row[f"mean_width_{level}"] = (
                        interval["mean_interval_width"]
                    )
            matrix_rows.append(row)
availability = pd.DataFrame(availability_rows)
matrix = pd.DataFrame(matrix_rows)
availability.to_csv(
    OUT / "target_source_availability.csv",
    index=False,
)
matrix.to_csv(
    OUT / "source_heldout_evaluation_matrix.csv",
    index=False,
)
status_counts = (
    matrix.groupby(["target", "status"])
    .size()
    .reset_index(name="fold_count")
)
status_counts.to_csv(
    OUT / "evaluation_status_counts.csv",
    index=False,
)
summary = {
    "experiment": "MAT-X source-held-out validation feasibility audit",
    "original_experiment_directory": str(EXPERIMENT),
    "successful_folds": int((matrix["status"] == "COMPLETED").sum()),
    "not_evaluable_folds": int(
        matrix["status"].str.startswith("NOT_EVALUABLE").sum()
    ),
    "execution_failures": int(
        (
            matrix["status"]
            == "EXECUTION_FAILED_REQUIRES_REVIEW"
        ).sum()
    ),
    "not_run_folds": int((matrix["status"] == "NOT_RUN").sum()),
    "interpretation": [
        "Only folds with target labels in the held-out source and "
        "remaining training sources are evaluable.",
        "Reduction area and impact energy are available only in EMK_spec.",
        "NIMS has no usable yield-strength or elongation labels.",
        "NIMS has only four independent grade groups for tensile strength.",
        "Source-held-out results indicate empirical cross-source transfer; "
        "they do not establish guaranteed uncertainty coverage.",
        "All models remain experimental and unpromoted.",
    ],
    "output_directory": str(OUT),
}
(OUT / "audit_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("=" * 78)
print("MAT-X SOURCE-HELD-OUT FEASIBILITY AUDIT")
print("=" * 78)
print(json.dumps(summary, indent=2))
print("\nTARGET-SOURCE AVAILABILITY")
print(availability.to_string(index=False))
print("\nEVALUATION STATUS COUNTS")
print(status_counts.to_string(index=False))
print("\nOUTPUT FILES")
for path in sorted(OUT.iterdir()):
    print(path)
print("\nExisting experiment outputs were not modified.")
