from pathlib import Path
import pandas as pd
ROOT = Path("training_results")
directories = [
    ROOT / "source_heldout_validation_v1",
    ROOT / "controlled_feature_ablation_v1",
    ROOT / "applicability_domain_v3_retry",
]
print("\n=== APPLICABILITY RECORD SCHEMA ===")
app_path = (
    ROOT
    / "applicability_domain_v3_retry"
    / "record_applicability.csv"
)
if app_path.exists():
    df = pd.read_csv(app_path, nrows=5)
    print("File:", app_path)
    print("Columns:", list(df.columns))
    print(df.head(2).to_string(index=False))
else:
    print("Applicability records file not found.")
for directory in directories:
    print(f"\n=== DIRECTORY: {directory} ===")
    if not directory.exists():
        print("NOT FOUND")
        continue
    csv_files = sorted(directory.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        continue
    for path in csv_files:
        try:
            sample = pd.read_csv(path, nrows=3)
            print(f"\nFILE: {path}")
            print(f"SIZE: {path.stat().st_size:,} bytes")
            print("COLUMNS:", list(sample.columns))
            print("SAMPLE ROWS:", len(sample))
        except Exception as exc:
            print(f"\nUNREADABLE: {path}")
            print(type(exc).__name__, str(exc))
print("\n=== PREDICTION ARTIFACT INVENTORY COMPLETE ===")
