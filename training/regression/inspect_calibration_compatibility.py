import json
from pathlib import Path
ROOT = Path("training_results")
FILES = {
    "PROMOTED_TENSILE": ROOT / "regression" / "steelbench_tensile_candidate_training_report.json",
    "PROMOTED_YIELD": ROOT / "regression" / "steelbench_yield_candidate_training_report.json",
    "CALIBRATED_TENSILE": ROOT / "uncertainty_calibration_v1" / "tensile_strength_baseline_report.json",
    "CALIBRATED_YIELD": ROOT / "uncertainty_calibration_v1" / "yield_strength_baseline_report.json",
}
FIELDS = [
    "target", "algorithm", "model", "validation",
    "training_records", "train_records", "calibration_records",
    "test_records", "training_grades", "train_grades",
    "calibration_grades", "test_grades",
    "MAE", "RMSE", "R2", "mae", "rmse", "r2",
    "random_state", "features", "feature_columns",
    "uncertainty_method", "validation_scope", "production_status",
]
for label, path in FILES.items():
    print("\n" + "=" * 75)
    print(label)
    print("FILE:", path)
    print("=" * 75)
    if not path.exists():
        print("MISSING REPORT")
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    for field in FIELDS:
        if field in report:
            print(f"{field}: {report[field]}")
    if "intervals" in report:
        print("INTERVALS:")
        print(json.dumps(report["intervals"], indent=2))
