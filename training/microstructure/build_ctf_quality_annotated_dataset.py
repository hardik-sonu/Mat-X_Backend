import json
import numpy as np
import pandas as pd
from pathlib import Path
root = Path("training_results/microstructure/abaqus_deep_audit")
features = pd.read_csv(root / "ctf_5deg_grain_level_features.csv")
quality = pd.read_csv(root / "ctf_5deg_grain_geometry_quality_flags.csv")
# Join without losing or duplicating grains.
df = features.merge(
    quality[
        [
            "grain_id",
            "map_edge_length",
            "unindexed_neighbour_edge_length",
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction"
        ]
    ],
    on="grain_id",
    how="left",
    validate="one_to_one"
)
if len(df) != len(features):
    raise RuntimeError("Grain count changed during quality merge.")
if df.isna().any().any():
    missing = df.columns[df.isna().any()].tolist()
    raise RuntimeError(f"Missing values after merge: {missing}")
# ---------------------------------------------------------
# Independent quality flags
# ---------------------------------------------------------
df["quality_low_pixel_count"] = df["pixel_count"] <= 4
df["quality_map_edge_exposed"] = (
    df["map_edge_exposure_fraction"] > 0.25
)
df["quality_unindexed_exposed"] = (
    df["unindexed_exposure_fraction"] > 0.25
)
df["geometry_screen_pass"] = (
    (df["pixel_count"] >= 5)
    & (~df["quality_map_edge_exposed"])
    & (~df["quality_unindexed_exposed"])
)
df["boundary_screen_pass"] = (
    (df["pixel_count"] >= 5)
    & (~df["quality_unindexed_exposed"])
)
df["grain_size_screen_pass"] = (
    (df["pixel_count"] >= 5)
    & (~df["quality_map_edge_exposed"])
)
def quality_reasons(row):
    reasons = []
    if row["quality_low_pixel_count"]:
        reasons.append("LOW_PIXEL_COUNT")
    if row["quality_map_edge_exposed"]:
        reasons.append("MAP_EDGE_EXPOSED")
    if row["quality_unindexed_exposed"]:
        reasons.append("UNINDEXED_EXPOSED")
    return "|".join(reasons) if reasons else "NO_SCREENING_FLAG"
df["quality_flags"] = df.apply(quality_reasons, axis=1)
df["geometry_quality_status"] = np.where(
    df["geometry_screen_pass"],
    "SCREEN_PASS_PROVISIONAL",
    "SCREEN_REVIEW_REQUIRED"
)
# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
if df["grain_id"].duplicated().any():
    raise RuntimeError("Duplicate grain IDs.")
if not (
    df["geometry_screen_pass"]
    == (
        df["grain_size_screen_pass"]
        & df["boundary_screen_pass"]
    )
).all():
    raise RuntimeError("Quality-screen logic inconsistency.")
if not np.allclose(
    df["map_edge_length"]
    + df["unindexed_neighbour_edge_length"],
    df["unindexed_or_map_edge_length_ctf_units"]
):
    raise RuntimeError("Exposure decomposition mismatch.")
# ---------------------------------------------------------
# Save consolidated dataset
# ---------------------------------------------------------
output_csv = root / "ctf_5deg_matx_grain_features_quality_annotated.csv"
output_json = root / "ctf_5deg_matx_grain_features_quality_report.json"
df.to_csv(output_csv, index=False)
summary = {
    "dataset": "EBSD.ctf",
    "segmentation_threshold_degrees": 5,
    "grain_count": int(len(df)),
    "feature_count": int(len(df.columns)),
    "quality_screening_rules": {
        "small_grain_pixels_max": 4,
        "map_edge_exposure_fraction_review_above": 0.25,
        "unindexed_exposure_fraction_review_above": 0.25
    },
    "quality_counts": {
        "low_pixel_count": int(df["quality_low_pixel_count"].sum()),
        "map_edge_exposed": int(df["quality_map_edge_exposed"].sum()),
        "unindexed_exposed": int(df["quality_unindexed_exposed"].sum()),
        "geometry_screen_pass": int(df["geometry_screen_pass"].sum()),
        "geometry_screen_review": int((~df["geometry_screen_pass"]).sum()),
        "grain_size_screen_pass": int(df["grain_size_screen_pass"].sum()),
        "boundary_screen_pass": int(df["boundary_screen_pass"].sum())
    },
    "overlapping_quality_flags": True,
    "records_removed": 0,
    "source_dataset_modified": False,
    "interpretation": (
        "All grain records are preserved. Quality flags identify "
        "provisional limitations on geometry and boundary descriptors."
    ),
    "limitations": [
        "5-degree grain segmentation remains provisional.",
        "Quality thresholds are screening choices, not validated standards.",
        "CTF coordinate units are not confirmed as micrometres.",
        "Orientation convention remains subject to source verification.",
        "Unindexed exposure does not establish a physical grain boundary."
    ]
}
output_json.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)
print("=" * 70)
print("MAT-X QUALITY-ANNOTATED GRAIN DATASET")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nQuality status distribution:")
print(df["geometry_quality_status"].value_counts().to_string())
print("\nSample records:")
print(
    df[
        [
            "grain_id",
            "pixel_count",
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction",
            "geometry_screen_pass",
            "boundary_screen_pass",
            "quality_flags"
        ]
    ].head(15).to_string(index=False)
)
print("\nSaved:")
print(output_csv)
print(output_json)
