from pathlib import Path
import pandas as pd
ROOT = Path("training_results")
CASES = {
    "TENSILE": (
        ROOT / "regression" /
        "steelbench_tensile_candidate_test_predictions.csv",
        ROOT / "uncertainty_calibration_v1" /
        "tensile_strength_baseline_test_intervals.csv",
    ),
    "YIELD": (
        ROOT / "regression" /
        "steelbench_yield_candidate_test_predictions.csv",
        ROOT / "uncertainty_calibration_v1" /
        "yield_strength_baseline_test_intervals.csv",
    ),
}
for name, (promoted_path, calibration_path) in CASES.items():
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    promoted = pd.read_csv(promoted_path)
    calibrated = pd.read_csv(calibration_path)
    print("Promoted columns:", promoted.columns.tolist())
    print("Calibration columns:", calibrated.columns.tolist())
    print("Promoted rows:", len(promoted))
    print("Calibration rows:", len(calibrated))
    keys = ["grade_id", "actual"]
    for frame in (promoted, calibrated):
        frame["grade_id"] = frame["grade_id"].astype(str)
        frame["actual"] = pd.to_numeric(
            frame["actual"], errors="coerce"
        ).round(10)
    promoted_keys = set(map(tuple, promoted[keys].to_numpy()))
    calibration_keys = set(map(tuple, calibrated[keys].to_numpy()))
    common = promoted_keys & calibration_keys
    print("Unique promoted keys:", len(promoted_keys))
    print("Unique calibration keys:", len(calibration_keys))
    print("Common test keys:", len(common))
    print(
        "Same test-key sets:",
        promoted_keys == calibration_keys
    )
    if len(promoted_keys) != len(promoted):
        print("WARNING: Duplicate promoted keys detected.")
    if len(calibration_keys) != len(calibrated):
        print("WARNING: Duplicate calibration keys detected.")
    print("Promoted sample:")
    print(promoted.head(3).to_string(index=False))
    print("Calibration sample:")
    print(calibrated.head(3).to_string(index=False))
print("\nREAD-ONLY TEST-SET COMPATIBILITY AUDIT COMPLETED")
