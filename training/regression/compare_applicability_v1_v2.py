from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path("training_results")
V1 = ROOT / "applicability_domain_v1"
V2 = ROOT / "applicability_domain_v2"
OUT = ROOT / "applicability_domain_comparison_v1"
if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")
OUT.mkdir(parents=True)
a = pd.read_csv(V1 / "record_applicability.csv")
b = pd.read_csv(V2 / "record_applicability.csv")
keys = [
    "target", "held_out_source", "variant",
    "source", "grade_id", "heat_id"
]
for name, frame in [("v1", a), ("v2", b)]:
    if frame.duplicated(keys).any():
        raise RuntimeError(f"{name}: duplicate record keys")
    if not np.isfinite(
        frame["nearest_training_distance"].to_numpy()
    ).all():
        raise RuntimeError(f"{name}: non-finite distances")
merged = a.merge(
    b,
    on=keys,
    how="outer",
    suffixes=("_v1", "_v2"),
    indicator=True,
    validate="one_to_one",
)
if not (merged["_merge"] == "both").all():
    raise RuntimeError("v1/v2 record populations do not match")
if len(merged) != 9024:
    raise RuntimeError(f"Unexpected matched row count: {len(merged)}")
merged["status_changed"] = (
    merged["applicability_status_v1"]
    != merged["applicability_status_v2"]
)
merged["distance_change"] = (
    merged["nearest_training_distance_v2"]
    - merged["nearest_training_distance_v1"]
)
merged["threshold_q95_change"] = (
    merged["training_distance_q95_v2"]
    - merged["training_distance_q95_v1"]
)
merged["threshold_q99_change"] = (
    merged["training_distance_q99_v2"]
    - merged["training_distance_q99_v1"]
)
status_transition = (
    merged.groupby(
        ["applicability_status_v1", "applicability_status_v2"],
        dropna=False,
    )
    .size()
    .rename("records")
    .reset_index()
)
fold_changes = (
    merged.groupby(
        ["target", "held_out_source", "variant"],
        dropna=False,
    )
    .agg(
        records=("heat_id", "size"),
        changed_records=("status_changed", "sum"),
        v1_median_distance=(
            "nearest_training_distance_v1", "median"
        ),
        v2_median_distance=(
            "nearest_training_distance_v2", "median"
        ),
        v1_q95=("training_distance_q95_v1", "first"),
        v2_q95=("training_distance_q95_v2", "first"),
        v1_q99=("training_distance_q99_v1", "first"),
        v2_q99=("training_distance_q99_v2", "first"),
    )
    .reset_index()
)
fold_changes["changed_fraction"] = (
    fold_changes["changed_records"] / fold_changes["records"]
)
source_changes = (
    merged.groupby(
        ["target", "held_out_source", "variant"],
        dropna=False,
    )
    .apply(
        lambda x: pd.Series({
            "records": len(x),
            "v1_in_domain": int(
                (x["applicability_status_v1"] == "IN_DOMAIN").sum()
            ),
            "v2_in_domain": int(
                (x["applicability_status_v2"] == "IN_DOMAIN").sum()
            ),
            "v1_borderline": int(
                (x["applicability_status_v1"] == "BORDERLINE").sum()
            ),
            "v2_borderline": int(
                (x["applicability_status_v2"] == "BORDERLINE").sum()
            ),
            "v1_out_of_domain": int(
                (x["applicability_status_v1"] == "OUT_OF_DOMAIN").sum()
            ),
            "v2_out_of_domain": int(
                (x["applicability_status_v2"] == "OUT_OF_DOMAIN").sum()
            ),
            "v1_insufficient": int(
                (x["applicability_status_v1"] == "INSUFFICIENT_EVIDENCE").sum()
            ),
            "v2_insufficient": int(
                (x["applicability_status_v2"] == "INSUFFICIENT_EVIDENCE").sum()
            ),
            "changed_records": int(x["status_changed"].sum()),
        })
    )
    .reset_index()
)
split = pd.read_csv(V2 / "split_integrity_audit.csv")
overlap_columns = [
    "train_calibration_overlap",
    "train_test_overlap",
    "calibration_test_overlap",
]
if not (split[overlap_columns].to_numpy() == 0).all():
    raise RuntimeError("Grade overlap found in v2 split audit")
transitions_path = OUT / "status_transition_matrix.csv"
fold_path = OUT / "fold_comparison.csv"
source_path = OUT / "source_status_comparison.csv"
changed_path = OUT / "changed_records.csv"
status_transition.to_csv(transitions_path, index=False)
fold_changes.to_csv(fold_path, index=False)
source_changes.to_csv(source_path, index=False)
merged.loc[merged["status_changed"]].to_csv(
    changed_path, index=False
)
print("\n=== V1/V2 RECORD MATCHING ===")
print("V1 rows:", len(a))
print("V2 rows:", len(b))
print("Matched rows:", len(merged))
print("Changed classifications:", int(merged["status_changed"].sum()))
print(
    "Unchanged classifications:",
    int((~merged["status_changed"]).sum())
)
print("\n=== STATUS TRANSITIONS ===")
print(status_transition.to_string(index=False))
print("\n=== CLASSIFICATION CHANGES BY FOLD ===")
print(
    fold_changes.sort_values(
        ["target", "held_out_source", "variant"]
    ).to_string(index=False)
)
print("\n=== SOURCE-LEVEL STATUS COMPARISON ===")
print(source_changes.to_string(index=False))
print("\n=== V2 SPLIT INTEGRITY ===")
print("Folds:", len(split))
print("All grade-overlap checks zero:", bool(
    (split[overlap_columns].to_numpy() == 0).all()
))
print("\n=== V2 REPORT METADATA ===")
summary = json.loads(
    (V2 / "summary.json").read_text(encoding="utf-8")
)
print("Experiment label:", summary.get("experiment"))
print("Threshold description:", summary["method"]["thresholds"])
print("Reference distance:", summary["method"]["reference_distance"])
print("\nOutputs:")
for path in [
    transitions_path,
    fold_path,
    source_path,
    changed_path,
]:
    print(path.resolve())
