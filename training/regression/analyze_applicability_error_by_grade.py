from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path("training_results")
INPUT = (
    ROOT / "applicability_prediction_error_validation_v1"
    / "matched_applicability_predictions.csv"
)
OUT = ROOT / "applicability_prediction_error_validation_v1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BOOTSTRAP_REPS = 2000
MIN_PAIRED_GRADES = 5
STATUS_ORDER = [
    "IN_DOMAIN",
    "BORDERLINE",
    "OUT_OF_DOMAIN",
    "INSUFFICIENT_EVIDENCE",
]
FOLD_KEYS = ["target", "held_out_source", "variant"]
df = pd.read_csv(INPUT)
required = set(
    FOLD_KEYS + [
        "source", "grade_id", "heat_id",
        "applicability_status", "calculated_absolute_error",
    ]
)
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {sorted(missing)}")
if df[list(required)].isna().any().any():
    raise RuntimeError("Null values found in required analysis fields.")
df["calculated_absolute_error"] = pd.to_numeric(
    df["calculated_absolute_error"], errors="coerce"
)
if not np.isfinite(
    df["calculated_absolute_error"].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite absolute errors found.")
unknown = set(df["applicability_status"].unique()) - set(STATUS_ORDER)
if unknown:
    raise RuntimeError(f"Unexpected statuses: {sorted(unknown)}")
status_rows = []
contrast_rows = []
rng = np.random.default_rng(SEED)
print("Input records:", len(df))
print("Fold groups:", df.groupby(FOLD_KEYS).ngroups)
print("Bootstrap repetitions:", BOOTSTRAP_REPS)
for fold_number, (fold_key, fold) in enumerate(
    df.groupby(FOLD_KEYS, sort=True)
):
    target, held_out, variant = fold_key
    print(
        f"\nFOLD {fold_number + 1}: "
        f"{target} | {held_out} | {variant}"
    )
    # Each grade contributes one mean error per status.
    grade_status = (
        fold.groupby(
            ["grade_id", "applicability_status"],
            as_index=False,
        )
        .agg(
            records=("calculated_absolute_error", "size"),
            grade_mae=("calculated_absolute_error", "mean"),
        )
    )
    for status in STATUS_ORDER:
        subset = fold.loc[
            fold["applicability_status"] == status
        ]
        if subset.empty:
            continue
        grade_values = grade_status.loc[
            grade_status["applicability_status"] == status,
            "grade_mae",
        ]
        status_rows.append({
            "target": target,
            "held_out_source": held_out,
            "variant": variant,
            "applicability_status": status,
            "records": int(len(subset)),
            "grades": int(subset["grade_id"].nunique()),
            "record_mae": float(
                subset["calculated_absolute_error"].mean()
            ),
            "record_median_absolute_error": float(
                subset["calculated_absolute_error"].median()
            ),
            "grade_macro_mae": float(grade_values.mean()),
            "grade_median_mae": float(grade_values.median()),
        })
    # Compare each non-reference status with IN_DOMAIN
    # using only grades represented in both statuses.
    reference = grade_status.loc[
        grade_status["applicability_status"] == "IN_DOMAIN",
        ["grade_id", "grade_mae"],
    ].rename(columns={"grade_mae": "in_domain_grade_mae"})
    for status in [
        "BORDERLINE",
        "OUT_OF_DOMAIN",
        "INSUFFICIENT_EVIDENCE",
    ]:
        comparison = grade_status.loc[
            grade_status["applicability_status"] == status,
            ["grade_id", "grade_mae"],
        ].rename(columns={"grade_mae": "comparison_grade_mae"})
        paired = reference.merge(
            comparison,
            on="grade_id",
            how="inner",
            validate="one_to_one",
        )
        differences = (
            paired["comparison_grade_mae"].to_numpy(dtype=float)
            - paired["in_domain_grade_mae"].to_numpy(dtype=float)
        )
        n_paired = len(differences)
        row = {
            "target": target,
            "held_out_source": held_out,
            "variant": variant,
            "comparison_status": status,
            "paired_grades": n_paired,
            "mean_paired_grade_mae_difference": None,
            "median_paired_grade_mae_difference": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "bootstrap_repetitions": 0,
            "interpretation": "INSUFFICIENT_PAIRED_GRADES",
        }
        if n_paired:
            observed = float(np.mean(differences))
            row["mean_paired_grade_mae_difference"] = observed
            row["median_paired_grade_mae_difference"] = float(
                np.median(differences)
            )
        if n_paired >= MIN_PAIRED_GRADES:
            boot_means = np.empty(BOOTSTRAP_REPS)
            for b in range(BOOTSTRAP_REPS):
                sampled = rng.integers(
                    0, n_paired, size=n_paired
                )
                boot_means[b] = differences[sampled].mean()
            lower, upper = np.quantile(
                boot_means, [0.025, 0.975]
            )
            row["ci95_lower"] = float(lower)
            row["ci95_upper"] = float(upper)
            row["bootstrap_repetitions"] = BOOTSTRAP_REPS
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
                    "NO_CLEAR_PAIRED_DIFFERENCE"
                )
        contrast_rows.append(row)
        print(
            f"  {status}: paired grades={n_paired}, "
            f"difference={row['mean_paired_grade_mae_difference']}, "
            f"result={row['interpretation']}"
        )
status_table = pd.DataFrame(status_rows)
contrast_table = pd.DataFrame(contrast_rows)
status_path = OUT / "grade_level_error_by_status.csv"
contrast_path = OUT / "paired_grade_bootstrap_contrasts.csv"
status_table.to_csv(status_path, index=False)
contrast_table.to_csv(contrast_path, index=False)
summary = {
    "input_records": int(len(df)),
    "fold_groups": int(df.groupby(FOLD_KEYS).ngroups),
    "status_summary_rows": int(len(status_table)),
    "paired_contrast_rows": int(len(contrast_table)),
    "bootstrap_repetitions": BOOTSTRAP_REPS,
    "minimum_paired_grades_for_interval": MIN_PAIRED_GRADES,
    "positive_ci_contrasts": int(
        (
            contrast_table["interpretation"]
            == "HIGHER_ERROR_THAN_IN_DOMAIN"
        ).sum()
    ),
    "negative_ci_contrasts": int(
        (
            contrast_table["interpretation"]
            == "LOWER_ERROR_THAN_IN_DOMAIN"
        ).sum()
    ),
    "overlapping_ci_contrasts": int(
        (
            contrast_table["interpretation"]
            == "NO_CLEAR_PAIRED_DIFFERENCE"
        ).sum()
    ),
    "insufficient_paired_grade_contrasts": int(
        (
            contrast_table["interpretation"]
            == "INSUFFICIENT_PAIRED_GRADES"
        ).sum()
    ),
    "outputs": [
        status_path.name,
        contrast_path.name,
    ],
    "limitations": [
        "Bootstrap resamples whole grades, not individual heats.",
        "Contrasts use only grades represented in both statuses.",
        "Results are descriptive validation, not causal effects.",
        "Small paired-grade counts produce no confidence interval.",
        "IN_DOMAIN is the reference status.",
        "INSUFFICIENT_EVIDENCE is not an ordinal distance category.",
        "Results remain source- and feature-variant-specific.",
    ],
}
(OUT / "grade_level_validation_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\n=== GRADE-LEVEL STATUS SUMMARY ===")
print(status_table.to_string(index=False))
print("\n=== PAIRED GRADE BOOTSTRAP CONTRASTS ===")
print(contrast_table.to_string(index=False))
print("\n=== VALIDATION SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nOutput directory:", OUT.resolve())
print("\n=== GRADE-LEVEL ERROR VALIDATION COMPLETE ===")
