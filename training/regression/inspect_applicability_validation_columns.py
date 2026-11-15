from pathlib import Path
import pandas as pd
path = Path(
    "training_results/applicability_prediction_error_validation_v1/"
    "matched_applicability_predictions.csv"
)
if not path.exists():
    raise FileNotFoundError(path)
df = pd.read_csv(path, nrows=10)
print("=== INPUT COLUMN AUDIT ===")
for column in df.columns:
    print(column)
print("\n=== ROW PREVIEW ===")
print(df.head(3).to_string(index=False))
print("\n=== REQUIRED FIELD CHECK ===")
required = [
    "target",
    "held_out_source",
    "variant",
    "grade_id",
    "applicability_status",
    "absolute_error",
]
for column in required:
    print(f"{column}: {'FOUND' if column in df.columns else 'MISSING'}")
print("\n=== COLUMN AUDIT COMPLETE ===")
