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
PREVIOUS = (
    BASE / "statistical_robustness_v1"
    / "multiplicity_adjusted_contrasts.csv"
)
OUT = BASE / "statistical_robustness_v2"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 100_000
BATCH_SIZE = 2_000
MIN_GRADES = 10
SEED = 20261114
ALPHA = 0.05
rng = np.random.default_rng(SEED)
df = pd.read_csv(INPUT)
previous = pd.read_csv(PREVIOUS)
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
        f"Missing columns: {sorted(missing)}"
    )
df["grade_id"] = df["grade_id"].astype(str)
df["calculated_absolute_error"] = pd.to_numeric(
    df["calculated_absolute_error"],
    errors="coerce",
)
if not np.isfinite(
    df["calculated_absolute_error"].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite errors detected.")
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
        differences = np.empty(REPS, dtype=float)
        # Generate independent grade-level bootstrap samples
        # in batches to control memory usage.
        for start in range(0, REPS, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, REPS)
            batch_n = stop - start
            ref_indices = rng.integers(
                0,
                len(reference),
                size=(batch_n, len(reference)),
            )
            comp_indices = rng.integers(
                0,
                len(comparison),
                size=(batch_n, len(comparison)),
            )
            ref_means = reference[ref_indices].mean(axis=1)
            comp_means = comparison[comp_indices].mean(axis=1)
            differences[start:stop] = (
                comp_means - ref_means
            )
        nominal_lower, nominal_upper = np.quantile(
            differences,
            [0.025, 0.975],
        )
        adjusted_tail = ALPHA / (
            2 * 19
        )
        adjusted_lower, adjusted_upper = np.quantile(
            differences,
            [adjusted_tail, 1 - adjusted_tail],
        )
        # Split-half estimates provide a diagnostic of
        # extreme-tail Monte Carlo stability.
        first_half = differences[:REPS // 2]
        second_half = differences[REPS // 2:]
        first_lower, first_upper = np.quantile(
            first_half,
            [adjusted_tail, 1 - adjusted_tail],
        )
        second_lower, second_upper = np.quantile(
            second_half,
            [adjusted_tail, 1 - adjusted_tail],
        )
        if adjusted_lower > 0:
            interpretation = "HIGHER_ERROR_AFTER_ADJUSTMENT"
        elif adjusted_upper < 0:
            interpretation = "LOWER_ERROR_AFTER_ADJUSTMENT"
        else:
            interpretation = "NOT_SIGNIFICANT_AFTER_ADJUSTMENT"
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
            "bonferroni_ci95_lower": float(adjusted_lower),
            "bonferroni_ci95_upper": float(adjusted_upper),
            "split_half_lower_1": float(first_lower),
            "split_half_upper_1": float(first_upper),
            "split_half_lower_2": float(second_lower),
            "split_half_upper_2": float(second_upper),
            "split_half_lower_difference": float(
                abs(first_lower - second_lower)
            ),
            "split_half_upper_difference": float(
                abs(first_upper - second_upper)
            ),
            "adjusted_tail_expected_observations": float(
                REPS * adjusted_tail
            ),
            "bootstrap_repetitions": REPS,
            "interpretation": interpretation,
        })
output = pd.DataFrame(results)
if len(output) != 19:
    raise RuntimeError(
        f"Expected 19 comparisons, found {len(output)}."
    )
interval_valid = (
    (output["nominal_ci95_lower"] <= output["nominal_ci95_upper"])
    & (
        output["bonferroni_ci95_lower"]
        <= output["bonferroni_ci95_upper"]
    )
)
if not interval_valid.all():
    raise RuntimeError("Invalid confidence intervals.")
# Compare against the previous 5,000-repetition assessment.
key_columns = fold_keys + ["comparison_status"]
old = previous[key_columns + [
    "bonferroni_ci95_lower",
    "bonferroni_ci95_upper",
    "bonferroni_interpretation",
]].copy()
merged = output.merge(
    old,
    on=key_columns,
    how="outer",
    suffixes=("_v2", "_v1"),
    indicator=True,
)
if not (merged["_merge"] == "both").all():
    raise RuntimeError(
        "Comparison keys differ from v1."
    )
merged["lower_bound_change"] = (
    merged["bonferroni_ci95_lower_v2"]
    - merged["bonferroni_ci95_lower_v1"]
)
merged["upper_bound_change"] = (
    merged["bonferroni_ci95_upper_v2"]
    - merged["bonferroni_ci95_upper_v1"]
)
merged["classification_changed"] = (
    merged["interpretation"]
    != merged["bonferroni_interpretation"]
)
comparison_output = merged[
    key_columns + [
        "bonferroni_ci95_lower_v1",
        "bonferroni_ci95_upper_v1",
        "bonferroni_ci95_lower_v2",
        "bonferroni_ci95_upper_v2",
        "lower_bound_change",
        "upper_bound_change",
        "bonferroni_interpretation",
        "interpretation",
        "classification_changed",
    ]
]
output.to_csv(
    OUT / "multiplicity_adjusted_contrasts_100k.csv",
    index=False,
)
comparison_output.to_csv(
    OUT / "comparison_against_5k_bootstrap.csv",
    index=False,
)
higher = int(
    (
        output["interpretation"]
        == "HIGHER_ERROR_AFTER_ADJUSTMENT"
    ).sum()
)
lower = int(
    (
        output["interpretation"]
        == "LOWER_ERROR_AFTER_ADJUSTMENT"
    ).sum()
)
not_significant = int(
    (
        output["interpretation"]
        == "NOT_SIGNIFICANT_AFTER_ADJUSTMENT"
    ).sum()
)
summary = {
    "input_records": int(len(df)),
    "fold_groups": int(df.groupby(fold_keys).ngroups),
    "grade_status_rows": int(len(grade_data)),
    "eligible_comparisons": int(len(output)),
    "bootstrap_repetitions": REPS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "family_alpha": ALPHA,
    "bonferroni_comparisons": 19,
    "adjusted_tail_probability": ALPHA / 38,
    "expected_observations_per_adjusted_tail": (
        REPS * ALPHA / 38
    ),
    "higher_error_after_adjustment": higher,
    "lower_error_after_adjustment": lower,
    "not_significant_after_adjustment": not_significant,
    "classification_changes_vs_5k": int(
        merged["classification_changed"].sum()
    ),
    "maximum_split_half_lower_difference": float(
        output["split_half_lower_difference"].max()
    ),
    "maximum_split_half_upper_difference": float(
        output["split_half_upper_difference"].max()
    ),
    "all_intervals_valid": bool(interval_valid.all()),
    "all_comparison_keys_matched": True,
    "limitations": [
        "Independent grade-level resampling.",
        "Bonferroni adjustment across 19 eligible comparisons.",
        "Split-half comparison is a diagnostic, not a formal error bound.",
        "Percentile bootstrap intervals may retain finite-sample limitations.",
        "Results are source-, target- and variant-specific.",
        "No causal or external-validity claim.",
    ],
    "outputs": [
        "multiplicity_adjusted_contrasts_100k.csv",
        "comparison_against_5k_bootstrap.csv",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("=== 100,000-REPETITION ADJUSTED RESULTS ===")
columns = [
    "target",
    "held_out_source",
    "variant",
    "comparison_status",
    "in_domain_grades",
    "comparison_grades",
    "mean_mae_difference",
    "bonferroni_ci95_lower",
    "bonferroni_ci95_upper",
    "interpretation",
]
print(output[columns].to_string(index=False))
print("\n=== COMPARISON AGAINST 5,000 REPS ===")
print(comparison_output.to_string(index=False))
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nOutput:", OUT.resolve())
print("\n=== 100K ROBUSTNESS ASSESSMENT COMPLETE ===")



