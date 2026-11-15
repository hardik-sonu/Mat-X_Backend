import pandas as pd
path = (
    "training_results/applicability_prediction_error_validation_v1/"
    "matched_applicability_predictions.csv"
)
df = pd.read_csv(path)
keys = ["target", "held_out_source", "variant", "heat_id"]
print("Total rows:", len(df))
print("Unique heat IDs globally:", df["heat_id"].nunique())
print("Unique assessment heat keys:", len(df[keys].drop_duplicates()))
print("Duplicate assessment heat keys:", df.duplicated(keys).sum())
print("\nDuplicate counts by assessment:")
print(
    df.groupby(["target", "held_out_source", "variant"])
      .apply(lambda g: g["heat_id"].duplicated().sum(),
             include_groups=False)
      .to_string()
)
print("\nExample globally repeated heat IDs:")
counts = df.groupby("heat_id").size()
repeated = counts[counts > 1]
print(repeated.head(10).to_string())
print("\n=== DIAGNOSTIC COMPLETE ===")
