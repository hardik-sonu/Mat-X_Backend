from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT = Path("training_results/microstructure/abaqus_deep_audit")
STEP = 1.1
features = pd.read_csv(OUT / "ctf_5deg_grain_level_features_v3.csv")
segments = pd.read_csv(OUT / "ctf_5deg_grain_boundary_segments.csv")
pairs = pd.read_csv(OUT / "ctf_5deg_grain_neighbour_pairs.csv")
exposure = pd.read_csv(OUT / "ctf_5deg_separated_boundary_exposure.csv")
merged = features.merge(
    exposure,
    on="grain_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=("_feature", "_exposure")
)
ids_match = bool((merged["_merge"] == "both").all())
merged = merged.drop(columns="_merge")
feature_sides = merged[
    "internal_grain_boundary_sides_feature"
].to_numpy(dtype=int)
exposure_sides = merged[
    "internal_grain_boundary_sides_exposure"
].to_numpy(dtype=int)
segment_counts = merged[
    "boundary_segment_count"
].to_numpy(dtype=int)
length_actual = merged[
    "internal_boundary_length_ctf_units"
].to_numpy(dtype=float)
length_expected = exposure_sides.astype(float) * STEP
checks = {
    "652_grains": len(features) == 652,
    "unique_grain_ids": features["grain_id"].nunique() == 652,
    "feature_exposure_ids_match": ids_match,
    "14248_unique_boundary_segments": len(segments) == 14248,
    "1752_unique_neighbour_pairs": len(pairs) == 1752,
    "internal_sides_equal_twice_segments": (
        int(exposure_sides.sum()) == 2 * len(segments)
    ),
    "feature_and_exposure_sides_match_per_grain": bool(
        np.array_equal(feature_sides, exposure_sides)
    ),
    "segment_counts_match_grain_side_counts": bool(
        np.array_equal(segment_counts, exposure_sides)
    ),
    "neighbour_degree_sum_matches": (
        int(features["neighbour_count"].sum()) == 2 * len(pairs)
    ),
    "boundary_lengths_match_per_grain": bool(
        np.allclose(length_actual, length_expected, rtol=1e-10, atol=1e-9)
    ),
    "high_low_counts_reconcile": bool((
        features["high_angle_boundary_count"]
        + features["low_angle_boundary_count"]
        == features["boundary_segment_count"]
    ).all()),
    "pixel_count_conserved": int(features["pixel_count"].sum()) == 73163
}
report = {
    "grain_count": int(len(features)),
    "unique_boundary_segments": int(len(segments)),
    "unique_neighbour_pairs": int(len(pairs)),
    "internal_grain_side_counts": int(exposure_sides.sum()),
    "expected_grain_side_counts": int(2 * len(segments)),
    "feature_side_count_sum": int(feature_sides.sum()),
    "exposure_side_count_sum": int(exposure_sides.sum()),
    "neighbour_degree_sum": int(features["neighbour_count"].sum()),
    "expected_neighbour_degree_sum": int(2 * len(pairs)),
    "high_angle_side_count": int(features["high_angle_boundary_count"].sum()),
    "low_angle_side_count": int(features["low_angle_boundary_count"].sum()),
    "internal_boundary_length_sum": float(length_actual.sum()),
    "expected_internal_boundary_length": float(exposure_sides.sum() * STEP),
    "maximum_per_grain_length_error": float(
        np.max(np.abs(length_actual - length_expected))
    ),
    "integrity_checks": checks,
    "all_checks_pass": all(checks.values()),
    "limitations": [
        "These checks establish internal numerical consistency only.",
        "Grain segmentation remains provisional at a 5-degree threshold.",
        "CTF coordinate units remain physically unconfirmed.",
        "Physical grain-boundary validity is not established."
    ]
}
path = OUT / "ctf_5deg_boundary_network_cross_validation_v3.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("=" * 72)
print("BOUNDARY NETWORK CROSS-VALIDATION V3")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nIndividual checks:")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print("\nSaved:")
print(path.resolve())
if not all(checks.values()):
    raise RuntimeError("Boundary network cross-validation requires review.")
