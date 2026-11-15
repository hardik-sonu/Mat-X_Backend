from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT = Path("training_results/microstructure/abaqus_deep_audit")
source = OUT / "ctf_5deg_grain_level_features_v2.csv"
output_csv = OUT / "ctf_5deg_grain_level_features_v3.csv"
output_json = OUT / "ctf_5deg_grain_level_features_v3_report.json"
df = pd.read_csv(source)
if len(df) != 652 or df["grain_id"].nunique() != 652:
    raise RuntimeError("Unexpected grain count or duplicate grain IDs.")
df["is_very_small_region"] = df["pixel_count"] < 5
df["has_high_map_edge_exposure"] = (
    df["map_edge_exposure_fraction"] >= 0.25
)
df["has_high_unindexed_exposure"] = (
    df["unindexed_exposure_fraction"] >= 0.25
)
flag_columns = [
    "is_very_small_region",
    "has_high_map_edge_exposure",
    "has_high_unindexed_exposure"
]
df["quality_flag_count"] = df[flag_columns].sum(axis=1)
df["boundary_quality_flags"] = df.apply(
    lambda row: ";".join(
        name for name, column in [
            ("VERY_SMALL_REGION", "is_very_small_region"),
            ("HIGH_MAP_EDGE_EXPOSURE", "has_high_map_edge_exposure"),
            ("HIGH_UNINDEXED_EXPOSURE", "has_high_unindexed_exposure")
        ]
        if bool(row[column])
    ) or "NO_AUTOMATIC_FLAG",
    axis=1
)
overlap = {
    "small_and_map_edge": int(
        (df["is_very_small_region"] &
         df["has_high_map_edge_exposure"]).sum()
    ),
    "small_and_unindexed": int(
        (df["is_very_small_region"] &
         df["has_high_unindexed_exposure"]).sum()
    ),
    "map_edge_and_unindexed": int(
        (df["has_high_map_edge_exposure"] &
         df["has_high_unindexed_exposure"]).sum()
    ),
    "all_three": int(
        (df[flag_columns].sum(axis=1) == 3).sum()
    )
}
report = {
    "grain_count": int(len(df)),
    "feature_count": int(len(df.columns)),
    "independent_flag_counts": {
        column: int(df[column].sum())
        for column in flag_columns
    },
    "grains_with_any_flag": int(
        (df["quality_flag_count"] > 0).sum()
    ),
    "grains_with_no_flags": int(
        (df["quality_flag_count"] == 0).sum()
    ),
    "flag_overlap_counts": overlap,
    "flag_combinations": {
        str(k): int(v)
        for k, v in df["boundary_quality_flags"].value_counts().items()
    },
    "grain_exclusion_count": 0,
    "interpretation": [
        "Flags are independent and may overlap.",
        "No grains are removed or automatically rejected.",
        "Thresholds are screening rules, not validated metallographic criteria.",
        "All geometry remains provisional."
    ]
}
df.to_csv(output_csv, index=False)
output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("=" * 72)
print("GRAIN FEATURE TABLE V3 - INDEPENDENT QUALITY FLAGS")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nFlagged grains:")
print(
    df.loc[df["quality_flag_count"] > 0, [
        "grain_id",
        "pixel_count",
        "map_edge_exposure_fraction",
        "unindexed_exposure_fraction",
        "boundary_quality_flags"
    ]].to_string(index=False)
)
print("\nSaved:")
print(output_csv.resolve())
print(output_json.resolve())
