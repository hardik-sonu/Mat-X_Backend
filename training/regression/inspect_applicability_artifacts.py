from pathlib import Path
import pandas as pd
ROOT = Path("training_results")
folders = [
    "applicability_domain_v3",
    "applicability_domain_v3_retry",
    "applicability_feature_eligibility_v3",
    "uncertainty_calibration_v1",
    "applicability_prediction_error_validation_v1",
]
for folder in folders:
    directory = ROOT / folder
    print("\n" + "=" * 85)
    print("DIRECTORY:", folder)
    print("=" * 85)
    if not directory.exists():
        print("NOT FOUND")
        continue
    for path in sorted(directory.glob("*.csv")):
        print(f"\nFILE: {path.name}")
        try:
            df = pd.read_csv(path)
            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())
            if len(df) > 0:
                print("\nFirst 3 rows:")
                print(df.head(3).to_string(index=False))
        except Exception as exc:
            print("READ ERROR:", str(exc))
print("\nARTIFACT INSPECTION COMPLETE")
