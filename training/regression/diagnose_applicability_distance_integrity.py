from pathlib import Path
import pandas as pd
for version in ["v1", "v2"]:
    path = Path(f"training_results/applicability_domain_{version}/record_applicability.csv")
    print(f"\n--- {version.upper()} ---")
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    distance_cols = [c for c in df.columns if "distance" in c.lower()]
    for col in distance_cols:
        print(f"\n{col}:")
        print(df[col].describe(percentiles=[.5, .95, .99]).to_string())
    if "variant" in df.columns:
        selected = df[
            (df["target"] == "tensile_strength") &
            (df["held_out_source"] == "EMK_spec") &
            (df["variant"] == "full_13")
        ]
        print("\nTensile / EMK / full_13:")
        for col in distance_cols:
            print(col, selected[col].describe().to_dict())
print("\n=== COMPARISON DISTANCE COLUMNS ===")
path = Path("training_results/applicability_domain_comparison_v1/fold_comparison.csv")
df = pd.read_csv(path)
print(df.columns.tolist())
print(df[
    (df["target"] == "tensile_strength") &
    (df["held_out_source"] == "EMK_spec") &
    (df["variant"] == "full_13")
].to_string(index=False))
print("\n=== EMPTY FAILURE FILE ===")
p = Path("training_results/applicability_domain_v2/execution_failures.csv")
print("Exists:", p.exists(), "Bytes:", p.stat().st_size if p.exists() else None)
print("\n=== CONTROLLED PREDICTION SCHEMA ===")
p = Path("training_results/controlled_feature_ablation_v1/predictions/tensile_strength/EMK_spec/full_13.csv")
pred = pd.read_csv(p)
print("Rows:", len(pred))
print("Columns:", pred.columns.tolist())
print(pred.head(2).to_string(index=False))
