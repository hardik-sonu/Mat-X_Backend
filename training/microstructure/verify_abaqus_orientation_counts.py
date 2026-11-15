import pandas as pd
from pathlib import Path
p = Path(
    "training_results/microstructure/abaqus_deep_audit/"
    "abaqus_ten_grain_orientation_validation.csv"
)
df = pd.read_csv(p)
print("=" * 60)
print("FINAL ABAQUS ORIENTATION COUNT VERIFICATION")
print("=" * 60)
print("TOTAL MATERIAL DEFINITIONS:", len(df))
print("VALID BASES:", int(df["valid_basis"].sum()))
print("INVALID BASES:", int((~df["valid_basis"]).sum()))
print("UNIQUE MODEL CONTEXTS:", df["context"].nunique())
print("\nMODEL DISTRIBUTION:")
df["model_index"] = df["context"].str.extract(r"root/(\d+)/")
print(
    df.groupby("model_index")
      .agg(
          materials=("material", "count"),
          valid=("valid_basis", "sum"),
          invalid=("valid_basis", lambda x: (~x).sum())
      )
      .to_string()
)
passed = len(df) == 60 and bool(df["valid_basis"].all())
print("\nFINAL RESULT:", "PASS" if passed else "REVIEW REQUIRED")
if not passed:
    raise SystemExit(1)
