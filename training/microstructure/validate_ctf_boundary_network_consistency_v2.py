from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT = Path("training_results/microstructure/abaqus_deep_audit")
features = pd.read_csv(OUT / "ctf_5deg_grain_level_features_v3.csv")
segments = pd.read_csv(OUT / "ctf_5deg_grain_boundary_segments.csv")
pairs = pd.read_csv(OUT / "ctf_5deg_grain_neighbour_pairs.csv")
exposure = pd.read_csv(OUT / "ctf_5deg_separated_boundary_exposure.csv")
STEP = 1.1
EXPECTED_GRAINS = 652
EXPECTED_SEGMENTS = 14248
EXPECTED_PAIRS = 1752
# One-to-one alignment of grain records.
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
# Reconcile counts per grain, not just aggregate totals.
segment_count_match = bool(np.array_equal(
    merged["boundary_segment_count"].to_numpy(dtype=int),
    merged["internal_grain_boundary_sides"].to_numpy(dtype=int)
))
neighbour_count_match = (
    int(features["neighbour_count"].sum()) == 2 * len(pairs)
)
length_expected = (
    merged["internal_grain_boundary_sides"].to_numpy(dtype=float)
    * STEP
)
length_actual = merged[
    "internal_boundary_length_ctf_units"
].to_numpy(dtype=float)
length_match = bool(np.allclose(
    length_actual,
    length_expected,
    rtol=1e-10,
    atol=1e-9
))
internal_sides = int(
    exposure["internal_grain_boundary_sides"].sum()
)
high_low_match = bool((
    features["high_angle_boundary_count"]
    + features["low_angle_boundary_count"]
    == features["boundary_segment_count"]
).all())
checks = {
    "expected_grain_count": len(features) == EXPECTED_GRAINS,
    "unique_grain_ids": features["grain_id"].nunique() == EXPECTED_GRAINS,
    "feature_exposure_ids_match": ids_match,
    "expected_boundary_segment_count": len(segments) == EXPECTED_SEGMENTS,
    "expected_neighbour_pair_count": len(pairs) == EXPECTED_PAIRS,
    "internal_sides_equal_twice_segments": (
        internal_sides == 2 * len(segments)
    ),
    "grain_side_segment_counts_match_per_grain": segment_count_match,
    "neighbour_degree_sum_equals_twice_pairs": neighbour_count_match,
    "boundary_lengths_match_per_grain": length_match,
    "high_low_counts_reconcile_per_grain": high_low_match,
    "pixel_count_conserved": int(features["pixel_count"].sum()) == 73163
}
report = {
    "grain_count": int(len(features)),
    "boundary_segments": int(len(segments)),
    "unique_neighbour_pairs": int(len(pairs)),
    "internal_boundary_sides": internal_sides,
    "expected_internal_boundary_sides": int(2 * len(segments)),
    "feature_boundary_segment_count_sum": int(
        features["boundary_segment_count"].sum()
    ),
    "feature_neighbour_count_sum": int(
        features["neighbour_count"].sum()
    ),
    "expected_neighbour_degree_sum": int(2 * len(pairs)),
    "high_angle_count_sum": int(
        features["high_angle_boundary_count"].sum()
    ),
    "low_angle_count_sum": int(
        features["low_angle_boundary_count"].sum()
    ),
    "internal_boundary_length_sum": float(
        features["internal_boundary_length_ctf_units"].sum()
    ),
    "expected_internal_boundary_length_sum": float(
        internal_sides * STEP
    ),
    "maximum_per_grain_boundary_length_error": float(
        np.max(np.abs(length_actual - length_expected))
    ),
    "integrity_checks": checks,
    "all_checks_pass": all(checks.values()),
    "interpretation": [
        "Floating-point comparisons use numerical tolerance.",
        "Internal interfaces contribute one side to each adjoining grain.",
        "Boundary lengths are checked per grain against pixel-side counts.",
        "Consistency does not independently validate physical grain boundaries."
    ]
}
path = OUT / "ctf_5deg_boundary_network_cross_validation_v2.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("=" * 72)
print("BOUNDARY NETWORK CROSS-VALIDATION V2")
print("=" * 72)
print(json.dumps(report, indent=2))
print("\nIndividual checks:")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print("\nSaved:")
print(path.resolve())
if not all(checks.values()):
    raise RuntimeError("Boundary network cross-validation still requires review.")
