from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "training_results"
    / "controlled_feature_ablation_v1"
    / "predictions"
)
OUT = (
    ROOT
    / "training_results"
    / "grade_cluster_bootstrap_v1"
)
TARGETS = [
    "tensile_strength",
    "yield_strength",
    "elongation",
]
SOURCES = [
    "EMK_spec",
    "NIMS",
    "Kaggle_Carbon_Steel",
    "Kaggle_Stainless_Steel",
]
VARIANTS = [
    "full_13",
    "chemistry_2",
    "chemistry_3",
]
COMPARISONS = [
    ("chemistry_2", "full_13"),
    ("chemistry_3", "chemistry_2"),
]
BOOTSTRAPS = 2000
RANDOM_SEED = 20261113
if not SOURCE_DIR.exists():
    raise FileNotFoundError(SOURCE_DIR)
if OUT.exists():
    raise FileExistsError(
        f"Output already exists; refusing to overwrite: {OUT}"
    )
OUT.mkdir(parents=True)
rng = np.random.default_rng(RANDOM_SEED)
def load_fold(target, source):
    frames = {}
    for variant in VARIANTS:
        path = (
            SOURCE_DIR
            / target
            / source
            / f"{variant}.csv"
        )
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        required = {
            "source",
            "grade_id",
            "heat_id",
            "actual",
            "prediction",
        }
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{path}: missing columns {sorted(missing)}"
            )
        if frame.duplicated(
            ["source", "grade_id", "heat_id"]
        ).any():
            raise RuntimeError(
                f"Duplicate identifiers in {path}"
            )
        frames[variant] = frame
    keys = ["source", "grade_id", "heat_id"]
    base = frames["full_13"][
        keys + ["actual"]
    ].copy()
    for variant in ["chemistry_2", "chemistry_3"]:
        other = frames[variant][
            keys + ["actual"]
        ].copy()
        merged = base.merge(
            other,
            on=keys,
            how="outer",
            suffixes=("_base", "_other"),
            indicator=True,
            validate="one_to_one",
        )
        if not (merged["_merge"] == "both").all():
            raise RuntimeError(
                f"Identifier mismatch: {target}, {source}, {variant}"
            )
        if not np.allclose(
            merged["actual_base"].to_numpy(dtype=float),
            merged["actual_other"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"Actual values differ: {target}, {source}, {variant}"
            )
    result = base.copy()
    for variant in VARIANTS:
        pred = frames[variant][
            keys + ["prediction"]
        ].copy()
        pred = pred.rename(
            columns={"prediction": f"prediction_{variant}"}
        )
        result = result.merge(
            pred,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    if len(result) != len(base):
        raise RuntimeError(
            f"Unexpected row loss: {target}, {source}"
        )
    numeric_columns = [
        "actual",
        *[f"prediction_{v}" for v in VARIANTS],
    ]
    if not np.isfinite(
        result[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"Non-finite values: {target}, {source}"
        )
    result["held_out_source"] = source
    return result
def cluster_bootstrap(
    frame,
    prediction_b,
    prediction_a,
    group_columns,
):
    """
    Difference = MAE(B) - MAE(A).
    Negative values favour B.
    Resampling unit is the complete grade cluster.
    """
    actual = frame["actual"].to_numpy(dtype=float)
    pred_a = frame[prediction_a].to_numpy(dtype=float)
    pred_b = frame[prediction_b].to_numpy(dtype=float)
    point_difference = (
        mean_absolute_error(actual, pred_b)
        - mean_absolute_error(actual, pred_a)
    )
    grouped = frame.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ).indices
    clusters = list(grouped.values())
    if len(clusters) < 2:
        return {
            "point_difference": float(point_difference),
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_probability_b_better": None,
            "bootstrap_repetitions": 0,
            "status": "INSUFFICIENT_CLUSTERS",
        }
    distributions = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        selected = rng.integers(
            0,
            len(clusters),
            size=len(clusters),
        )
        sampled_indices = np.concatenate(
            [clusters[i] for i in selected]
        )
        sampled_actual = actual[sampled_indices]
        sampled_a = pred_a[sampled_indices]
        sampled_b = pred_b[sampled_indices]
        distributions[b] = (
            np.mean(np.abs(sampled_actual - sampled_b))
            - np.mean(np.abs(sampled_actual - sampled_a))
        )
    lower, upper = np.percentile(
        distributions,
        [2.5, 97.5],
    )
    return {
        "point_difference": float(point_difference),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_probability_b_better": float(
            np.mean(distributions < 0)
        ),
        "bootstrap_repetitions": BOOTSTRAPS,
        "status": "COMPLETED",
    }
fold_frames = []
fold_results = []
integrity_rows = []
for target in TARGETS:
    for source in SOURCES:
        frame = load_fold(target, source)
        if frame is None:
            continue
        fold_frames.append((target, source, frame))
        integrity_rows.append({
            "target": target,
            "held_out_source": source,
            "matched_records": len(frame),
            "matched_grades": frame["grade_id"].nunique(),
            "actual_values_verified": True,
            "predictions_finite": True,
        })
        for variant_b, variant_a in COMPARISONS:
            result = cluster_bootstrap(
                frame,
                f"prediction_{variant_b}",
                f"prediction_{variant_a}",
                ["grade_id"],
            )
            fold_results.append({
                "scope": "SOURCE_FOLD",
                "target": target,
                "held_out_source": source,
                "comparison": f"{variant_b}_minus_{variant_a}",
                "records": len(frame),
                "grades": frame["grade_id"].nunique(),
                **result,
            })
# Pooled comparison: independently resample grades within each
# held-out source, preserving the source composition of the test set.
for target in TARGETS:
    pieces = [
        frame
        for t, _, frame in fold_frames
        if t == target
    ]
    if not pieces:
        continue
    pooled = pd.concat(pieces, ignore_index=True)
    for variant_b, variant_a in COMPARISONS:
        result = cluster_bootstrap(
            pooled,
            f"prediction_{variant_b}",
            f"prediction_{variant_a}",
            ["held_out_source", "grade_id"],
        )
        fold_results.append({
            "scope": "POOLED_SOURCE_STRATIFIED",
            "target": target,
            "held_out_source": "ALL_AVAILABLE_SOURCES",
            "comparison": f"{variant_b}_minus_{variant_a}",
            "records": len(pooled),
            "grades": pooled[
                ["held_out_source", "grade_id"]
            ].drop_duplicates().shape[0],
            **result,
        })
results = pd.DataFrame(fold_results)
integrity = pd.DataFrame(integrity_rows)
results.to_csv(
    OUT / "grade_cluster_bootstrap_results.csv",
    index=False,
)
integrity.to_csv(
    OUT / "bootstrap_pairing_integrity.csv",
    index=False,
)
summary = {
    "experiment": "MAT-X grade-cluster bootstrap v1",
    "source_directory": str(SOURCE_DIR),
    "bootstrap_repetitions": BOOTSTRAPS,
    "random_seed": RANDOM_SEED,
    "confidence_interval": "Percentile 95% CI",
    "resampling_unit": "Entire grade cluster",
    "pooled_resampling": (
        "Grades independently resampled within each held-out source"
    ),
    "difference_definition": "MAE(B) - MAE(A); negative favours B",
    "successful_comparisons": len(results),
    "paired_folds": len(integrity),
    "all_actual_values_verified": bool(
        not integrity.empty
        and integrity["actual_values_verified"].all()
    ),
    "all_predictions_finite": bool(
        not integrity.empty
        and integrity["predictions_finite"].all()
    ),
    "outputs": str(OUT),
    "limitations": [
        "Bootstrap intervals quantify test-grade sampling uncertainty, not external validity.",
        "Only available source-held-out folds are included.",
        "The pooled result is stratified by held-out source.",
        "A confidence interval excluding zero is evidence of a difference under this resampling design, not proof of universal superiority.",
    ],
}
(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print("\n" + "=" * 100)
print("GRADE-CLUSTER BOOTSTRAP RESULTS")
print("=" * 100)
if not results.empty:
    print(
        results[
            [
                "scope",
                "target",
                "held_out_source",
                "comparison",
                "records",
                "grades",
                "point_difference",
                "ci_lower",
                "ci_upper",
                "bootstrap_probability_b_better",
                "status",
            ]
        ].to_string(index=False)
    )
print("\n" + "=" * 100)
print("PAIRING INTEGRITY")
print("=" * 100)
print(integrity.to_string(index=False))
print("\nSUMMARY")
print(json.dumps(summary, indent=2))
print("\nNo models retrained. No original artifacts modified.")
