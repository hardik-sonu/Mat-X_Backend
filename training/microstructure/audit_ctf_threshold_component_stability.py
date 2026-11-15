import json
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
thresholds = [2, 5, 10, 15]
maps = {}
# Load and validate original segmentation outputs.
for threshold in thresholds:
    path = root / f"ctf_grain_pixels_{threshold}deg.csv"
    df = pd.read_csv(path)
    required = {"X", "Y", "Phase", "Error", "grain_id"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Invalid columns in {path.name}")
    if df.duplicated(["X", "Y"]).any():
        raise RuntimeError(f"Duplicate coordinates in {path.name}")
    df = df.sort_values(["Y", "X"]).reset_index(drop=True)
    valid = (df["Phase"] == 1) & (df["Error"] == 0)
    if int(valid.sum()) != 73163:
        raise RuntimeError(
            f"Unexpected valid-pixel count in {path.name}"
        )
    if (df.loc[valid, "grain_id"] <= 0).any():
        raise RuntimeError(f"Invalid grain labels in {path.name}")
    if (df.loc[~valid, "grain_id"] != 0).any():
        raise RuntimeError(
            f"Nonzero labels found outside valid pixels in {path.name}"
        )
    maps[threshold] = df
    print(
        f"{threshold}°: "
        f"{df.loc[valid, 'grain_id'].nunique()} components, "
        f"{int(valid.sum())} indexed pixels"
    )
# Verify exact pixel-grid alignment.
reference = maps[2][["X", "Y", "Phase", "Error"]]
for threshold in thresholds[1:]:
    current = maps[threshold][["X", "Y", "Phase", "Error"]]
    if not reference.equals(current):
        raise RuntimeError(
            f"Pixel alignment mismatch at {threshold} degrees"
        )
merge_records = []
transition_records = []
for lower, upper in [(2, 5), (5, 10), (10, 15)]:
    low = maps[lower]
    high = maps[upper]
    valid = (low["Phase"] == 1) & (low["Error"] == 0)
    crosswalk = pd.DataFrame({
        "lower_component": low.loc[valid, "grain_id"].to_numpy(),
        "upper_component": high.loc[valid, "grain_id"].to_numpy()
    })
    # A lower-threshold component must not split into multiple
    # higher-threshold components.
    mapping_counts = crosswalk.groupby(
        "lower_component"
    )["upper_component"].nunique()
    if (mapping_counts != 1).any():
        raise RuntimeError(
            f"Unexpected component splitting: {lower}° to {upper}°"
        )
    # Count how many lower-threshold components form each
    # higher-threshold component.
    groups = (
        crosswalk.groupby("upper_component")
        .agg(
            lower_components=("lower_component", "nunique"),
            pixels=("lower_component", "size")
        )
        .reset_index()
    )
    merged = groups[groups["lower_components"] > 1]
    for _, row in merged.iterrows():
        upper_id = int(row["upper_component"])
        members = sorted(
            crosswalk.loc[
                crosswalk["upper_component"] == upper_id,
                "lower_component"
            ].unique().tolist()
        )
        merge_records.append({
            "lower_threshold": lower,
            "upper_threshold": upper,
            "upper_component": upper_id,
            "lower_components_merged": int(row["lower_components"]),
            "pixels_in_merged_component": int(row["pixels"]),
            "lower_component_ids": ",".join(
                str(int(v)) for v in members
            )
        })
    lower_count = int(crosswalk["lower_component"].nunique())
    upper_count = int(crosswalk["upper_component"].nunique())
    transition_records.append({
        "from_degrees": lower,
        "to_degrees": upper,
        "components_before": lower_count,
        "components_after": upper_count,
        "component_reduction": lower_count - upper_count,
        "merged_upper_components": int(len(merged)),
        "lower_components_involved": int(
            merged["lower_components"].sum()
        ) if len(merged) else 0,
        "pixels_in_merged_components": int(
            merged["pixels"].sum()
        ) if len(merged) else 0
    })
    print("\n" + "=" * 65)
    print(f"COMPONENT MERGES: {lower}° TO {upper}°")
    print("=" * 65)
    print(transition_records[-1])
    if len(merged):
        print("\nLargest merged components:")
        print(
            merged.sort_values("pixels", ascending=False)
            .head(10)
            .to_string(index=False)
        )
merge_df = pd.DataFrame(
    merge_records,
    columns=[
        "lower_threshold",
        "upper_threshold",
        "upper_component",
        "lower_components_merged",
        "pixels_in_merged_component",
        "lower_component_ids"
    ]
)
transition_df = pd.DataFrame(transition_records)
merge_df.to_csv(
    root / "ctf_threshold_component_merge_audit.csv",
    index=False
)
transition_df.to_csv(
    root / "ctf_threshold_component_transitions.csv",
    index=False
)
summary = {
    "valid_pixels": 73163,
    "threshold_component_counts": {
        str(t): int(
            maps[t].loc[
                (maps[t]["Phase"] == 1) &
                (maps[t]["Error"] == 0),
                "grain_id"
            ].nunique()
        )
        for t in thresholds
    },
    "transitions": transition_records,
    "merge_records": len(merge_records),
    "interpretation": (
        "Nested connected-component comparison using original "
        "pixel-level grain labels. Higher thresholds may merge "
        "lower-threshold components but must not split them."
    )
}
(root / "ctf_threshold_component_stability.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("\n" + "=" * 65)
print("CORRECTED THRESHOLD STABILITY SUMMARY")
print("=" * 65)
print(json.dumps(summary, indent=2))
print("\nSaved:")
print(root / "ctf_threshold_component_merge_audit.csv")
print(root / "ctf_threshold_component_transitions.csv")
print(root / "ctf_threshold_component_stability.json")
print("\nOriginal CTF and segmentation files unchanged.")
