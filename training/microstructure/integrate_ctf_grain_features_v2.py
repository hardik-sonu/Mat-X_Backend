from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT = Path(
    r"C:\Users\Ahmadcomputer\MAT-X\training_results"
) / "microstructure" / "abaqus_deep_audit"
features_path = OUT / "ctf_5deg_grain_level_features.csv"
exposure_path = OUT / "ctf_5deg_separated_boundary_exposure.csv"
features = pd.read_csv(features_path)
exposure = pd.read_csv(exposure_path)
if features["grain_id"].duplicated().any():
    raise RuntimeError("Duplicate grain IDs in feature table.")
if exposure["grain_id"].duplicated().any():
    raise RuntimeError("Duplicate grain IDs in exposure table.")
merged = features.merge(
    exposure,
    on="grain_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=("", "_separated")
)
if not (merged["_merge"] == "both").all():
    raise RuntimeError(
        "Feature/exposure grain IDs do not match:\n"
        + merged["_merge"].value_counts().to_string()
    )
merged = merged.drop(columns="_merge")
if len(merged) != 652:
    raise RuntimeError(f"Expected 652 grains, got {len(merged)}.")
# Check the old combined exposure feature against the new
# separated map-edge + unindexed exposure measurements.
if "boundary_exposure_fraction" in merged.columns:
    merged["separated_external_exposure_fraction"] = (
        merged["map_edge_exposure_fraction"]
        + merged["unindexed_exposure_fraction"]
    )
    merged["external_exposure_difference"] = (
        merged["boundary_exposure_fraction"]
        - merged["separated_external_exposure_fraction"]
    )
    max_difference = float(
        merged["external_exposure_difference"].abs().max()
    )
else:
    max_difference = None
# Check all three fractions sum to one.
fraction_sum = (
    merged["map_edge_exposure_fraction"]
    + merged["unindexed_exposure_fraction"]
    + merged["internal_boundary_fraction"]
)
fraction_error = float((fraction_sum - 1.0).abs().max())
if fraction_error > 1e-12:
    raise RuntimeError(
        f"Exposure fraction identity failed: {fraction_error}"
    )
# Mark grains needing additional caution.
merged["boundary_quality_flag"] = np.select(
    [
        merged["pixel_count"] < 5,
        merged["map_edge_exposure_fraction"] >= 0.25,
        merged["unindexed_exposure_fraction"] >= 0.25
    ],
    [
        "VERY_SMALL_REGION",
        "HIGH_MAP_EDGE_EXPOSURE",
        "HIGH_UNINDEXED_EXPOSURE"
    ],
    default="NO_AUTOMATIC_FLAG"
)
output_csv = OUT / "ctf_5deg_grain_level_features_v2.csv"
output_json = OUT / "ctf_5deg_grain_level_features_v2_report.json"
merged.to_csv(output_csv, index=False)
report = {
    "grain_count": int(len(merged)),
    "feature_count": int(len(merged.columns)),
    "fraction_identity_max_error": fraction_error,
    "old_vs_separated_external_exposure_max_difference": max_difference,
    "flag_counts": {
        str(k): int(v)
        for k, v in merged["boundary_quality_flag"].value_counts().items()
    },
    "mean_map_edge_exposure_fraction": float(
        merged["map_edge_exposure_fraction"].mean()
    ),
    "mean_unindexed_exposure_fraction": float(
        merged["unindexed_exposure_fraction"].mean()
    ),
    "mean_internal_boundary_fraction": float(
        merged["internal_boundary_fraction"].mean()
    ),
    "interpretation": [
        "Original feature table is preserved.",
        "Separated map-edge and unindexed-neighbour exposure are added.",
        "Automatic flags are screening indicators, not grain rejection decisions.",
        "All geometric measurements remain provisional."
    ]
}
output_json.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)
print("=" * 72)
print("GRAIN FEATURE TABLE V2 INTEGRATION")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nFlagged grains:")
print(
    merged.loc[
        merged["boundary_quality_flag"] != "NO_AUTOMATIC_FLAG",
        [
            "grain_id",
            "pixel_count",
            "map_edge_exposure_fraction",
            "unindexed_exposure_fraction",
            "boundary_quality_flag"
        ]
    ].to_string(index=False)
)
print("\nSaved:")
print(output_csv)
print(output_json)
