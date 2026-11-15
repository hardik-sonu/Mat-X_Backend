from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path.cwd()
BASE = (
    ROOT / "training_results"
    / "applicability_prediction_error_validation_v1"
)
INPUT = BASE / "matched_applicability_predictions.csv"
OUT = BASE / "independent_grade_bootstrap_v1"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 5000
MIN_GRADES = 10
SEED = 42
rng = np.random.default_rng(SEED)
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
df["calculated_absolute_error"] = pd.to_numeric(
    df["calculated_absolute_error"],
    errors="coerce",
)
if not np.isfinite(
    df["calculated_absolute_error"].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite absolute errors found.")
fold_keys = ["target", "held_out_source", "variant"]
grade_data = (
    df.groupby(
        fold_keys + ["applicability_status", "grade_id"],
        as_index=False,
    )
    .agg(
        records=("calculated_absolute_error", "size"),
        grade_mae=("calculated_absolute_error", "mean"),
    )
)
status_order = [
    "IN_DOMAIN",
    "BORDERLINE",
    "OUT_OF_DOMAIN",
    "INSUFFICIENT_EVIDENCE",
]
results = []
distribution_rows = []
for fold_key, fold in grade_data.groupby(
    fold_keys, sort=True
):
    target, source, variant = fold_key
    groups = {}
    for status in status_order:
        values = fold.loc[
            fold["applicability_status"] == status,
            "grade_mae",
        ].to_numpy(dtype=float)
        groups[status] = values
        if len(values):
            distribution_rows.append({
                "target": target,
                "held_out_source": source,
                "variant": variant,
                "applicability_status": status,
                "grade_count": len(values),
                "mean_grade_mae": float(values.mean()),
                "median_grade_mae": float(np.median(values)),
                "q90_grade_mae": float(np.quantile(values, 0.90)),
                "q95_grade_mae": float(np.quantile(values, 0.95)),
            })
    reference = groups["IN_DOMAIN"]
    for status in [
        "BORDERLINE",
        "OUT_OF_DOMAIN",
        "INSUFFICIENT_EVIDENCE",
    ]:
        comparison = groups[status]
        row = {
            "target": target,
            "held_out_source": source,
            "variant": variant,
            "comparison_status": status,
            "in_domain_grades": len(reference),
            "comparison_grades": len(comparison),
            "in_domain_mean_grade_mae": None,
            "comparison_mean_grade_mae": None,
            "mean_mae_difference": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_repetitions": 0,
            "interpretation": "INSUFFICIENT_GRADE_SUPPORT",
        }
        if len(reference) and len(comparison):
            ref_mean = float(reference.mean())
            comp_mean = float(comparison.mean())
            row["in_domain_mean_grade_mae"] = ref_mean
            row["comparison_mean_grade_mae"] = comp_mean
            row["mean_mae_difference"] = comp_mean - ref_mean
        if (
            len(reference) >= MIN_GRADES
            and len(comparison) >= MIN_GRADES
        ):
            ref_boot = np.empty(REPS)
            comp_boot = np.empty(REPS)
            for b in range(REPS):
                ref_sample = rng.choice(
                    reference,
                    size=len(reference),
                    replace=True,
                )
                comp_sample = rng.choice(
                    comparison,
                    size=len(comparison),
                    replace=True,
                )
                ref_boot[b] = ref_sample.mean()
                comp_boot[b] = comp_sample.mean()
            differences = comp_boot - ref_boot
            lower, upper = np.quantile(
                differences,
                [0.025, 0.975],
            )
            row["ci95_lower"] = float(lower)
            row["ci95_upper"] = float(upper)
            row["bootstrap_repetitions"] = REPS
            if lower > 0:
                row["interpretation"] = (
                    "HIGHER_ERROR_THAN_IN_DOMAIN"
                )
            elif upper < 0:
                row["interpretation"] = (
                    "LOWER_ERROR_THAN_IN_DOMAIN"
                )
            else:
                row["interpretation"] = (
                    "NO_CLEAR_DIFFERENCE"
                )
        results.append(row)
contrasts = pd.DataFrame(results)
distributions = pd.DataFrame(distribution_rows)
contrasts.to_csv(
    OUT / "independent_grade_bootstrap_contrasts.csv",
    index=False,
)
distributions.to_csv(
    OUT / "grade_error_distributions.csv",
    index=False,
)
summary = {
    "input_records": int(len(df)),
    "fold_groups": int(df.groupby(fold_keys).ngroups),
    "grade_status_rows": int(len(grade_data)),
    "contrast_rows": int(len(contrasts)),
    "bootstrap_repetitions": REPS,
    "minimum_grades_per_category": MIN_GRADES,
    "higher_error_contrasts": int(
        (
            contrasts["interpretation"]
            == "HIGHER_ERROR_THAN_IN_DOMAIN"
        ).sum()
    ),
    "lower_error_contrasts": int(
        (
            contrasts["interpretation"]
            == "LOWER_ERROR_THAN_IN_DOMAIN"
        ).sum()
    ),
    "no_clear_difference": int(
        (
            contrasts["interpretation"]
            == "NO_CLEAR_DIFFERENCE"
        ).sum()
    ),
    "insufficient_support": int(
        (
            contrasts["interpretation"]
            == "INSUFFICIENT_GRADE_SUPPORT"
        ).sum()
    ),
    "outputs": [
        "independent_grade_bootstrap_contrasts.csv",
        "grade_error_distributions.csv",
    ],
    "limitations": [
        "Grades are independently resampled within each status.",
        "This is not a paired comparison.",
        "Results describe association, not causation.",
        "Comparisons are specific to source, target and variant.",
        "Intervals do not establish external validity.",
        "Small categories are not assigned confidence intervals.",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("=== INDEPENDENT GRADE-LEVEL BOOTSTRAP ===")
print(contrasts.to_string(index=False))
print("\n=== GRADE ERROR DISTRIBUTIONS ===")
print(distributions.to_string(index=False))
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nOutput:", OUT.resolve())
print("\n=== EXPERIMENT COMPLETE ===")
