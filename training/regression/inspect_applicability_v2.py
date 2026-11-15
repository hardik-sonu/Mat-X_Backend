from pathlib import Path
import pandas as pd
path = Path(r"training_results\applicability_domain_v2\fold_applicability_summary.csv")
df = pd.read_csv(path)
print("Columns:")
print(df.columns.tolist())
print("\nAll fold thresholds and calibration counts:")
preferred = [
    c for c in [
        "target", "held_out_source", "variant",
        "train_records", "train_grades",
        "calibration_records", "calibration_grades",
        "test_records", "test_grades",
        "training_distance_q95", "training_distance_q99",
        "calibration_distance_q95", "calibration_distance_q99",
        "distance_q95", "distance_q99",
    ] if c in df.columns
]
if preferred:
    print(df[preferred].to_string(index=False))
else:
    print(df.to_string(index=False))
print("\nLargest q95 values:")
q95 = next((c for c in df.columns if "q95" in c.lower()), None)
if q95:
    print(df.nlargest(8, q95).to_string(index=False))
