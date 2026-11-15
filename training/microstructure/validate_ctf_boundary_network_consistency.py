from pathlib import Path
import json
import pandas as pd
import numpy as np
OUT = Path("training_results/microstructure/abaqus_deep_audit")
files = {
    "features": OUT / "ctf_5deg_grain_level_features_v3.csv",
    "segments": OUT / "ctf_5deg_grain_boundary_segments.csv",
    "pairs": OUT / "ctf_5deg_grain_neighbour_pairs.csv",
    "exposure": OUT / "ctf_5deg_separated_boundary_exposure.csv"
}
for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(path)
features = pd.read_csv(files["features"])
segments = pd.read_csv(files["segments"])
pairs = pd.read_csv(files["pairs"])
exposure = pd.read_csv(files["exposure"])
print("INPUT COLUMNS")
for name, df in [
    ("features", features),
    ("segments", segments),
    ("pairs", pairs),
    ("exposure", exposure)
]:
    print(f"{name}: {list(df.columns)}")
checks = {}
checks["652_feature_grains"] = len(features) == 652
checks["652_exposure_grains"] = len(exposure) == 652
checks["unique_feature_grain_ids"] = features.grain_id.nunique() == 652
checks["unique_exposure_grain_ids"] = exposure.grain_id.nunique() == 652
segment_count = len(segments)
pair_count = len(pairs)
checks["expected_boundary_segment_count"] = segment_count == 14248
checks["expected_unique_neighbour_pair_count"] = pair_count == 1752
internal_sides = int(exposure["internal_grain_boundary_sides"].sum())
checks["internal_sides_equal_twice_segments"] = (
    internal_sides == 2 * segment_count
)
checks["feature_segment_count_matches_network"] = (
    int(features["boundary_segment_count"].sum())
    == 2 * segment_count
)
checks["feature_neighbour_count_matches_pairs"] = (
    int(features["neighbour_count"].sum())
    == 2 * pair_count
)
checks["feature_internal_sides_match_exposure"] = (
    int(features["internal_boundary_length_ctf_units"].sum() / 1.1)
    == internal_sides
)
# Verify the feature and exposure grain IDs are identical.
checks["feature_exposure_ids_match"] = set(
    features.grain_id
) == set(exposure.grain_id)
# Verify grain-level high/low angle counts reconcile.
if {
    "high_angle_boundary_count",
    "low_angle_boundary_count",
    "boundary_segment_count"
}.issubset(features.columns):
    checks["high_low_counts_sum_to_segments"] = bool(
        (
            features["high_angle_boundary_count"]
            + features["low_angle_boundary_count"]
            == features["boundary_segment_count"]
        ).all()
    )
# Check that every grain has a nonnegative number of neighbours.
checks["nonnegative_neighbour_counts"] = bool(
    (features["neighbour_count"] >= 0).all()
)
# Check network graph degree sum.
degree_sum = int(features["neighbour_count"].sum())
report = {
    "grain_count": int(len(features)),
    "boundary_segment_count": int(segment_count),
    "unique_neighbour_pairs": int(pair_count),
    "total_internal_perimeter_sides": internal_sides,
    "expected_internal_perimeter_sides": 2 * segment_count,
    "grain_network_degree_sum": degree_sum,
    "expected_degree_sum": 2 * pair_count,
    "integrity_checks": checks,
    "all_checks_pass": all(checks.values()),
    "interpretation": [
        "Boundary segments represent pixel-grid interfaces between adjacent grain labels.",
        "Each internal interface contributes one side to each adjoining grain.",
        "Neighbour counts describe the unique grain adjacency graph.",
        "These checks validate internal data consistency, not physical grain-boundary truth."
    ]
}
report_path = OUT / "ctf_5deg_boundary_network_cross_validation.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n" + "=" * 72)
print("BOUNDARY NETWORK CROSS-VALIDATION")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nFeature-level totals:")
for col in [
    "boundary_segment_count",
    "neighbour_count",
    "high_angle_boundary_count",
    "low_angle_boundary_count",
    "internal_boundary_length_ctf_units"
]:
    if col in features.columns:
        print(f"{col}: {features[col].sum()}")
print("\nSaved:")
print(report_path.resolve())
if not report["all_checks_pass"]:
    raise RuntimeError("Boundary network cross-validation requires review.")
