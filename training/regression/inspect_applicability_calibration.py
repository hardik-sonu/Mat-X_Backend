import pandas as pd
path = r"training_results\applicability_domain_v2\fold_applicability_summary.csv"
df = pd.read_csv(path)
cols = [
    c for c in [
        "target", "held_out_source", "variant",
        "train_records", "train_grades",
        "calibration_records", "calibration_grades",
        "test_records", "test_grades",
        "calibration_reference_rows",
        "training_distance_q95", "training_distance_q99",
    ] if c in df.columns
]
print("Available columns:", df.columns.tolist())
print(df[cols].to_string(index=False))
print("\nCalibration grade counts:")
if "calibration_grades" in df.columns:
    print(df["calibration_grades"].describe().to_string())
