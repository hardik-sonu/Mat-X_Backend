from pathlib import Path
import pandas as pd
ROOT = Path.cwd()
EXPERIMENTS = {
    "13_feature": ROOT / "training_results" / "source_heldout_validation_v1",
    "2_feature": ROOT / "training_results" / "common_feature_source_heldout_v1",
}
for label, directory in EXPERIMENTS.items():
    print("\n" + "=" * 90)
    print(f"{label.upper()} PREDICTION FILE AUDIT")
    print("=" * 90)
    prediction_dir = directory / "predictions"
    if not prediction_dir.exists():
        print("Prediction directory not found:", prediction_dir)
        continue
    files = sorted(prediction_dir.glob("*.csv"))
    print("Prediction files:", len(files))
    for path in files:
        try:
            data = pd.read_csv(path, nrows=5)
            full_rows = sum(
                1 for _ in open(path, "r", encoding="utf-8-sig")
            ) - 1
            print("\nFile:", path.name)
            print("Rows:", full_rows)
            print("Columns:", list(data.columns))
            identifiers = [
                col for col in [
                    "source",
                    "grade_id",
                    "heat_id",
                    "record_id",
                    "material_id",
                    "sample_id",
                ]
                if col in data.columns
            ]
            print("Available identifiers:", identifiers)
        except Exception as exc:
            print("READ ERROR:", path.name, exc)
print("\n" + "=" * 90)
print("METRICS FILE INSPECTION")
print("=" * 90)
for label, directory in EXPERIMENTS.items():
    metrics = directory / "metrics.csv"
    if metrics.exists():
        data = pd.read_csv(metrics)
        print(f"\n{label}:")
        print("Rows:", len(data))
        print("Columns:", list(data.columns))
        print(data.head(3).to_string(index=False))
    else:
        print(f"\n{label}: metrics.csv not found")
print("\nRead-only audit. No files modified.")
