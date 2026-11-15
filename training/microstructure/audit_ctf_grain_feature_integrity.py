from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT = Path("training_results/microstructure/abaqus_deep_audit")
SOURCE = OUT / "ctf_5deg_grain_level_features_v3.csv"
df = pd.read_csv(SOURCE)
required = [
    "grain_id", "pixel_count",
    "area_ctf_coordinate_units_squared",
    "equivalent_diameter_ctf_coordinate_units",
    "perimeter_ctf_coordinate_units",
    "circularity", "aspect_ratio_bbox",
    "neighbour_count",
    "mean_boundary_misorientation",
    "high_angle_boundary_fraction",
    "map_edge_exposure_fraction",
    "unindexed_exposure_fraction",
    "internal_boundary_fraction",
    "is_very_small_region",
    "has_high_map_edge_exposure",
    "has_high_unindexed_exposure"
]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise RuntimeError(f"Missing required columns: {missing_columns}")
if len(df) != 652:
    raise RuntimeError(f"Expected 652 grains, found {len(df)}.")
if df["grain_id"].nunique() != 652:
    raise RuntimeError("Duplicate grain IDs detected.")
numeric = df.select_dtypes(include=[np.number])
infinite_counts = {
    c: int(np.isinf(numeric[c]).sum())
    for c in numeric.columns
}
missing_counts = {
    c: int(df[c].isna().sum())
    for c in df.columns
}
# Geometric and physical-domain checks.
checks = {
    "positive_pixel_counts": bool((df["pixel_count"] > 0).all()),
    "positive_areas": bool(
        (df["area_ctf_coordinate_units_squared"] > 0).all()
    ),
    "positive_equivalent_diameters": bool(
        (df["equivalent_diameter_ctf_coordinate_units"] > 0).all()
    ),
    "positive_perimeters": bool(
        (df["perimeter_ctf_coordinate_units"] > 0).all()
    ),
    "circularity_in_unit_interval": bool(
        df["circularity"].between(0, 1.000001).all()
    ),
    "positive_aspect_ratios": bool(
        (df["aspect_ratio_bbox"] > 0).all()
    ),
    "nonnegative_neighbour_counts": bool(
        (df["neighbour_count"] >= 0).all()
    ),
    "misorientation_within_0_90": bool(
        df["mean_boundary_misorientation"].between(0, 90).all()
    ),
    "high_angle_fraction_in_unit_interval": bool(
        df["high_angle_boundary_fraction"].between(0, 1).all()
    ),
    "exposure_fractions_in_unit_interval": bool(
        df[
            [
                "map_edge_exposure_fraction",
                "unindexed_exposure_fraction",
                "internal_boundary_fraction"
            ]
        ].apply(lambda s: s.between(0, 1).all()).all()
    )
}
fraction_sum = (
    df["map_edge_exposure_fraction"]
    + df["unindexed_exposure_fraction"]
    + df["internal_boundary_fraction"]
)
fraction_error = float((fraction_sum - 1).abs().max())
checks["exposure_fraction_identity"] = fraction_error < 1e-12
checks["all_numeric_values_finite"] = all(
    value == 0 for value in infinite_counts.values()
)
# Verify independent flags against their thresholds.
checks["small_flag_consistency"] = bool(
    (
        df["is_very_small_region"].astype(bool)
        == (df["pixel_count"] < 5)
    ).all()
)
checks["map_edge_flag_consistency"] = bool(
    (
        df["has_high_map_edge_exposure"].astype(bool)
        == (df["map_edge_exposure_fraction"] >= 0.25)
    ).all()
)
checks["unindexed_flag_consistency"] = bool(
    (
        df["has_high_unindexed_exposure"].astype(bool)
        == (df["unindexed_exposure_fraction"] >= 0.25)
    ).all()
)
# Conservation and distribution summaries.
pixel_total = int(df["pixel_count"].sum())
distribution_columns = [
    "pixel_count",
    "area_ctf_coordinate_units_squared",
    "equivalent_diameter_ctf_coordinate_units",
    "perimeter_ctf_coordinate_units",
    "circularity",
    "aspect_ratio_bbox",
    "neighbour_count",
    "mean_boundary_misorientation",
    "high_angle_boundary_fraction",
    "map_edge_exposure_fraction",
    "unindexed_exposure_fraction",
    "internal_boundary_fraction"
]
distribution = {}
for col in distribution_columns:
    s = df[col].dropna()
    distribution[col] = {
        "count": int(s.count()),
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max())
    }
size_bins = {
    "1_pixel": int((df["pixel_count"] == 1).sum()),
    "2_to_4_pixels": int(df["pixel_count"].between(2, 4).sum()),
    "5_to_10_pixels": int(df["pixel_count"].between(5, 10).sum()),
    "11_to_25_pixels": int(df["pixel_count"].between(11, 25).sum()),
    "26_to_100_pixels": int(df["pixel_count"].between(26, 100).sum()),
    "over_100_pixels": int((df["pixel_count"] > 100).sum())
}
report = {
    "grain_count": int(len(df)),
    "unique_grain_ids": int(df["grain_id"].nunique()),
    "pixel_count_total": pixel_total,
    "expected_valid_pixel_count": 73163,
    "pixel_conservation_pass": pixel_total == 73163,
    "missing_values_by_column": missing_counts,
    "infinite_values_by_numeric_column": infinite_counts,
    "integrity_checks": checks,
    "all_integrity_checks_pass": all(checks.values()),
    "fraction_identity_max_error": fraction_error,
    "grain_size_bins": size_bins,
    "feature_distributions": distribution,
    "limitations": [
        "Grains are provisional 5-degree connected components.",
        "Euler-angle export convention remains subject to source verification.",
        "CTF coordinate units are not confirmed as micrometres.",
        "Map-edge and unindexed exposure are quality indicators, not rejection rules.",
        "No model suitability or predictive validity is established by this audit."
    ]
}
report_path = OUT / "ctf_5deg_grain_feature_integrity_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("=" * 72)
print("FINAL GRAIN FEATURE INTEGRITY AUDIT")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nFeature distribution:")
print(
    pd.DataFrame(distribution).T.to_string()
)
print("\nIntegrity result:")
print("PASS" if report["all_integrity_checks_pass"] else "REVIEW REQUIRED")
print("\nSaved:")
print(report_path.resolve())
if not report["all_integrity_checks_pass"]:
    raise RuntimeError("One or more integrity checks failed.")
