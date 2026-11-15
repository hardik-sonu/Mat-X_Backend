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
BOOTSTRAP_DIR = (
    BASE / "independent_grade_bootstrap_v1"
)
ORIGINAL = (
    BOOTSTRAP_DIR
    / "independent_grade_bootstrap_contrasts.csv"
)
OUT = BASE / "statistical_robustness_v1"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 5000
MIN_GRADES = 10
SEED = 42
FAMILY_ALPHA = 0.05
rng = np.random.default_rng(SEED)
df = pd.read_csv(INPUT)
original = pd.read_csv(ORIGINAL)
fold_keys = ["target", "held_out_source", "variant"]
required = {
    *fold_keys,
    "grade_id",
    "applicability_status",
    "calculated_absolute_error",
}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(
        f"Missing input columns: {sorted(missing)}"
    )
df["grade_id"] = df["grade_id"].astype(str)
df["calculated_absolute_error"] = pd.to_numeric(
    df["calculated_absolute_error"],
    errors="coerce",
)
if not np.isfinite(
    df["calculated_absolute_error"].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite errors found.")
grade_data = (
    df.groupby(
        fold_keys + [
            "applicability_status",
            "grade_id",
        ],
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
# Reproduce the original bootstrap sequence exactly.
results = []
for fold_key, fold in grade_data.groupby(
    fold_keys,
    sort=True,
):
    target, source, variant = fold_key
    groups = {}
    for status in status_order:
        groups[status] = fold.loc[
            fold["applicability_status"] == status,
            "grade_mae",
        ].to_numpy(dtype=float)
    reference = groups["IN_DOMAIN"]
    for status in [
        "BORDERLINE",
        "OUT_OF_DOMAIN",
        "INSUFFICIENT_EVIDENCE",
    ]:
        comparison = groups[status]
        if (
            len(reference) < MIN_GRADES
            or len(comparison) < MIN_GRADES
        ):
            continue
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
        nominal_lower, nominal_upper = np.quantile(
            differences,
            [0.025, 0.975],
        )
        results.append({
            "target": target,
            "held_out_source": source,
            "variant": variant,
            "comparison_status": status,
            "in_domain_grades": len(reference),
            "comparison_grades": len(comparison),
            "mean_mae_difference": float(
                comparison.mean() - reference.mean()
            ),
            "nominal_ci95_lower": float(nominal_lower),
            "nominal_ci95_upper": float(nominal_upper),
            "bootstrap_repetitions": REPS,
            "_bootstrap_differences": differences,
        })
if len(results) != 19:
    raise RuntimeError(
        f"Expected 19 eligible comparisons, found {len(results)}."
    )
# Verify reproducibility against the original results.
original["comparison_key"] = (
    original[fold_keys + ["comparison_status"]]
    .astype(str)
    .agg("|".join, axis=1)
)
reproduced = pd.DataFrame([
    {
        **{
            key: row[key]
            for key in fold_keys + ["comparison_status"]
        },
        "mean_mae_difference": row["mean_mae_difference"],
        "ci95_lower": row["nominal_ci95_lower"],
        "ci95_upper": row["nominal_ci95_upper"],
    }
    for row in results
])
reproduced["comparison_key"] = (
    reproduced[fold_keys + ["comparison_status"]]
    .astype(str)
    .agg("|".join, axis=1)
)
original_eligible = original[
    original["bootstrap_repetitions"] > 0
].copy()
if len(original_eligible) != 19:
    raise RuntimeError(
        "Original eligible comparison count is not 19."
    )
check = reproduced.merge(
    original_eligible[
        [
            "comparison_key",
            "mean_mae_difference",
            "ci95_lower",
            "ci95_upper",
        ]
    ],
    on="comparison_key",
    how="outer",
    suffixes=("_new", "_original"),
    indicator=True,
)
if not (check["_merge"] == "both").all():
    raise RuntimeError(
        "Original and reproduced comparison keys differ."
    )
numeric_checks = [
    ("mean_mae_difference_new", "mean_mae_difference_original"),
    ("ci95_lower_new", "ci95_lower_original"),
    ("ci95_upper_new", "ci95_upper_original"),
]
max_discrepancy = 0.0
for new_col, old_col in numeric_checks:
    discrepancy = np.max(
        np.abs(
            check[new_col].to_numpy(dtype=float)
            - check[old_col].to_numpy(dtype=float)
        )
    )
    max_discrepancy = max(
        max_discrepancy,
        float(discrepancy),
    )
if max_discrepancy > 1e-8:
    raise RuntimeError(
        "Bootstrap reproduction differs from original results: "
        f"{max_discrepancy}"
    )
# Bonferroni family-wise confidence interval.
number_of_comparisons = len(results)
adjusted_tail_probability = (
    FAMILY_ALPHA / (2 * number_of_comparisons)
)
for row in results:
    differences = row.pop("_bootstrap_differences")
    adjusted_lower, adjusted_upper = np.quantile(
        differences,
        [
            adjusted_tail_probability,
            1 - adjusted_tail_probability,
        ],
    )
    row["bonferroni_ci95_lower"] = float(adjusted_lower)
    row["bonferroni_ci95_upper"] = float(adjusted_upper)
    if adjusted_lower > 0:
        interpretation = "HIGHER_ERROR_AFTER_ADJUSTMENT"
    elif adjusted_upper < 0:
        interpretation = "LOWER_ERROR_AFTER_ADJUSTMENT"
    else:
        interpretation = "NOT_SIGNIFICANT_AFTER_ADJUSTMENT"
    row["bonferroni_interpretation"] = interpretation
    row["nominal_interpretation"] = (
        "HIGHER_ERROR"
        if row["nominal_ci95_lower"] > 0
        else (
            "LOWER_ERROR"
            if row["nominal_ci95_upper"] < 0
            else "NO_CLEAR_DIFFERENCE"
        )
    )
output = pd.DataFrame(results)
output.to_csv(
    OUT / "multiplicity_adjusted_contrasts.csv",
    index=False,
)
# Verify interval integrity.
interval_valid = (
    output["nominal_ci95_lower"]
    <= output["nominal_ci95_upper"]
) & (
    output["bonferroni_ci95_lower"]
    <= output["bonferroni_ci95_upper"]
)
if not interval_valid.all():
    raise RuntimeError("Invalid confidence interval detected.")
nominal_significant = (
    output["nominal_interpretation"].isin([
        "HIGHER_ERROR",
        "LOWER_ERROR",
    ])
)
adjusted_significant = (
    output["bonferroni_interpretation"].isin([
        "HIGHER_ERROR_AFTER_ADJUSTMENT",
        "LOWER_ERROR_AFTER_ADJUSTMENT",
    ])
)
summary = {
    "input_records": int(len(df)),
    "fold_groups": int(df.groupby(fold_keys).ngroups),
    "grade_status_rows": int(len(grade_data)),
    "eligible_comparisons": int(len(output)),
    "bootstrap_repetitions": REPS,
    "minimum_grades_per_category": MIN_GRADES,
    "family_alpha": FAMILY_ALPHA,
    "bonferroni_comparisons": number_of_comparisons,
    "adjusted_tail_probability": adjusted_tail_probability,
    "original_bootstrap_reproduced": True,
    "maximum_reproduction_discrepancy": max_discrepancy,
    "nominal_significant_comparisons": int(
        nominal_significant.sum()
    ),
    "bonferroni_significant_comparisons": int(
        adjusted_significant.sum()
    ),
    "higher_error_after_adjustment": int(
        (
            output["bonferroni_interpretation"]
            == "HIGHER_ERROR_AFTER_ADJUSTMENT"
        ).sum()
    ),
    "lower_error_after_adjustment": int(
        (
            output["bonferroni_interpretation"]
            == "LOWER_ERROR_AFTER_ADJUSTMENT"
        ).sum()
    ),
    "not_significant_after_adjustment": int(
        (
            output["bonferroni_interpretation"]
            == "NOT_SIGNIFICANT_AFTER_ADJUSTMENT"
        ).sum()
    ),
    "all_intervals_valid": bool(interval_valid.all()),
    "limitations": [
        "Bonferroni adjustment controls family-wise error "
        "for the 19 tested comparisons under the stated procedure.",
        "Bootstrap intervals are percentile intervals.",
        "Independent grade resampling is not paired.",
        "Comparisons are source-, target- and variant-specific.",
        "Statistical significance does not establish causality.",
        "This analysis does not establish external validity.",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("=== MULTIPLICITY-ADJUSTED RESULTS ===")
columns = [
    "target",
    "held_out_source",
    "variant",
    "comparison_status",
    "in_domain_grades",
    "comparison_grades",
    "mean_mae_difference",
    "nominal_ci95_lower",
    "nominal_ci95_upper",
    "bonferroni_ci95_lower",
    "bonferroni_ci95_upper",
    "bonferroni_interpretation",
]
print(
    output[columns].to_string(index=False)
)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nOutput:", OUT.resolve())
print("\n=== ROBUSTNESS ASSESSMENT COMPLETE ===")
