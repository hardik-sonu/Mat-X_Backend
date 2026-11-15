from pathlib import Path
import json
import math
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training_results" / "uncertainty_calibration_v1"
summary = json.loads(
    (OUT / "experiment_summary.json").read_text(encoding="utf-8")
)
print("=" * 76)
print("MAT-X CORRECTED CALIBRATION INTEGRITY AUDIT")
print("=" * 76)
checks = {}
for report in summary["results"]:
    target = report["target"]
    variant = report["variant"]
    name = f"{target} | {variant}"
    model_path = Path(report["model_path"])
    prediction_path = Path(report["prediction_path"])
    model_exists = model_path.is_file()
    prediction_exists = prediction_path.is_file()
    model_loads = False
    if model_exists:
        try:
            joblib.load(model_path)
            model_loads = True
        except Exception as exc:
            print(f"MODEL LOAD ERROR: {name}: {exc}")
    row_count_matches = False
    intervals_finite = False
    intervals_nested = False
    coverage_reproduces = False
    widths_reproduce = False
    if prediction_exists:
        frame = pd.read_csv(prediction_path)
        row_count_matches = len(frame) == report["test_records"]
        required = [
            "actual", "prediction", "lower_90", "upper_90",
            "lower_95", "upper_95", "grade_id"
        ]
        columns_present = all(c in frame.columns for c in required)
        if columns_present and row_count_matches:
            numeric = frame[
                ["actual", "prediction", "lower_90", "upper_90",
                 "lower_95", "upper_95"]
            ].to_numpy(dtype=float)
            intervals_finite = bool(np.isfinite(numeric).all())
            intervals_nested = bool(
                (frame["lower_95"] <= frame["lower_90"]).all()
                and
                (frame["upper_95"] >= frame["upper_90"]).all()
                and
                (frame["lower_90"] <= frame["upper_90"]).all()
                and
                (frame["lower_95"] <= frame["upper_95"]).all()
            )
            coverage_checks = []
            width_checks = []
            for level in ["90", "95"]:
                covered = (
                    (frame["actual"] >= frame[f"lower_{level}"])
                    &
                    (frame["actual"] <= frame[f"upper_{level}"])
                )
                calculated_record_coverage = float(covered.mean())
                recorded_record_coverage = float(
                    report["intervals"][level]["record_coverage"]
                )
                calculated_grade_coverage = float(
                    pd.DataFrame({
                        "grade_id": frame["grade_id"].astype(str),
                        "covered": covered.to_numpy()
                    }).groupby("grade_id")["covered"].all().mean()
                )
                recorded_grade_coverage = float(
                    report["intervals"][level]["whole_grade_coverage"]
                )
                calculated_width = float(
                    (frame[f"upper_{level}"] -
                     frame[f"lower_{level}"]).mean()
                )
                recorded_width = float(
                    report["intervals"][level]["mean_interval_width"]
                )
                coverage_checks.extend([
                    abs(
                        calculated_record_coverage
                        - recorded_record_coverage
                    ) < 1e-9,
                    abs(
                        calculated_grade_coverage
                        - recorded_grade_coverage
                    ) < 1e-9,
                ])
                width_checks.append(
                    abs(calculated_width - recorded_width) < 1e-7
                )
            coverage_reproduces = all(coverage_checks)
            widths_reproduce = all(width_checks)
    grade_groups_disjoint = all(
        report["grade_overlap_checks"].values()
    )
    checks[name] = {
        "model_exists": model_exists,
        "model_loads": model_loads,
        "prediction_file_exists": prediction_exists,
        "prediction_rows_match": row_count_matches,
        "intervals_finite": intervals_finite,
        "95_percent_contains_90_percent": intervals_nested,
        "coverage_reproduces_from_predictions": coverage_reproduces,
        "interval_widths_reproduce": widths_reproduce,
        "grade_groups_disjoint": grade_groups_disjoint,
    }
    print(f"\n{name}")
    for key, value in checks[name].items():
        print(f"  {key}: {value}")
all_pass = bool(checks) and all(
    all(item.values()) for item in checks.values()
)
print("\n" + "=" * 76)
print(f"ALL CORRECTED INTEGRITY CHECKS PASS: {all_pass}")
print("=" * 76)
print("Production integration remains NOT AUTHORIZED.")
