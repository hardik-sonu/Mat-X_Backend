from pathlib import Path
import pandas as pd
import json
ROOT = Path.cwd()
INPUT = (
    ROOT
    / "training_results"
    / "applicability_prediction_error_validation_v1"
    / "matched_applicability_predictions.csv"
)
OUT = (
    ROOT
    / "training_results"
    / "applicability_prediction_error_validation_v1"
    / "grade_support_diagnostic"
)
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT)
required = {
    "target",
    "held_out_source",
    "variant",
    "grade_id",
    "applicability_status",
    "calculated_absolute_error",
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")
df["grade_id"] = df["grade_id"].astype(str)
keys = ["target", "held_out_source", "variant"]
summary_rows = []
shared_rows = []
for group_key, group in df.groupby(keys, dropna=False):
    target, source, variant = group_key
    grade_status = (
        group.groupby(
            ["grade_id", "applicability_status"]
        )
        .agg(
            records=("calculated_absolute_error", "size"),
            grade_mae=("calculated_absolute_error", "mean"),
        )
        .reset_index()
    )
    grade_counts = (
        grade_status.groupby("applicability_status")
        .agg(
            distinct_grades=("grade_id", "nunique"),
            records=("records", "sum"),
        )
        .reset_index()
    )
    in_domain_grades = set(
        grade_status.loc[
            grade_status["applicability_status"] == "IN_DOMAIN",
            "grade_id",
        ]
    )
    for status in [
        "IN_DOMAIN",
        "BORDERLINE",
        "OUT_OF_DOMAIN",
        "INSUFFICIENT_EVIDENCE",
    ]:
        status_grades = set(
            grade_status.loc[
                grade_status["applicability_status"] == status,
                "grade_id",
            ]
        )
        shared = (
            len(in_domain_grades)
            if status == "IN_DOMAIN"
            else len(in_domain_grades.intersection(status_grades))
        )
        row = {
            "target": target,
            "held_out_source": source,
            "variant": variant,
            "applicability_status": status,
            "distinct_grades": len(status_grades),
            "in_domain_grades": len(in_domain_grades),
            "grades_shared_with_in_domain": shared,
            "records": int(
                grade_counts.loc[
                    grade_counts["applicability_status"] == status,
                    "records",
                ].sum()
            ) if status in set(grade_counts["applicability_status"]) else 0,
        }
        summary_rows.append(row)
        shared_rows.append({
            **row,
            "shared_grade_ids": ";".join(
                sorted(in_domain_grades.intersection(status_grades))
            ),
        })
summary = pd.DataFrame(summary_rows)
shared = pd.DataFrame(shared_rows)
summary.to_csv(
    OUT / "grade_support_summary.csv",
    index=False,
)
shared.to_csv(
    OUT / "shared_grade_ids.csv",
    index=False,
)
print("=== GRADE SUPPORT BY STATUS ===")
print(summary.to_string(index=False))
print("\n=== PAIRED-GRADE SUPPORT ===")
print(
    summary[
        [
            "target",
            "held_out_source",
            "variant",
            "applicability_status",
            "grades_shared_with_in_domain",
        ]
    ].to_string(index=False)
)
print("\n=== SUPPORT COUNTS ===")
print(
    summary.groupby("applicability_status")[
        ["distinct_grades", "grades_shared_with_in_domain"]
    ].agg(["count", "sum", "max"]).to_string()
)
result = {
    "input_records": len(df),
    "fold_groups": len(summary[keys].drop_duplicates()),
    "minimum_required_paired_grades": 5,
    "contrasts_with_at_least_five_shared_grades": int(
        summary.loc[
            summary["applicability_status"] != "IN_DOMAIN",
            "grades_shared_with_in_domain",
        ].ge(5).sum()
    ),
    "outputs": [
        "grade_support_summary.csv",
        "shared_grade_ids.csv",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)
print("\n=== SUMMARY ===")
print(json.dumps(result, indent=2))
print("\nOutput:", OUT.resolve())
