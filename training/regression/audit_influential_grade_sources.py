from pathlib import Path
import pandas as pd
path = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv")
df = pd.read_csv(path, low_memory=False, dtype={"grade_id": str})
print("DATASET SHAPE:", df.shape)
print("SOURCE-LIKE COLUMNS:")
source_cols = [
    c for c in df.columns
    if any(term in c.lower() for term in ("source", "dataset", "origin", "database"))
]
print(source_cols if source_cols else "No obvious source column found")
audits = {
    "tensile_strength": ["1044", "9310"],
    "yield_strength": ["1038", "202"],
}
for target, grades in audits.items():
    print("\n" + "=" * 75)
    print("TARGET:", target)
    print("=" * 75)
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=["grade_id", target])
    for grade in grades:
        subset = work[work["grade_id"].astype(str) == grade]
        print(f"\nGRADE: {grade}")
        print("Record count:", len(subset))
        if subset.empty:
            print("NOT FOUND")
            continue
        print("Target min:", subset[target].min())
        print("Target median:", subset[target].median())
        print("Target max:", subset[target].max())
        for col in source_cols:
            print(f"\n{col} distribution:")
            print(subset[col].value_counts(dropna=False).to_string())
        print("\nAvailable grade metadata:")
        metadata = [
            c for c in df.columns
            if c.lower() in (
                "grade_id", "steel_family", "grade", "standard",
                "source", "dataset", "material", "material_name"
            )
        ]
        print(subset[metadata].drop_duplicates().head(10).to_string(index=False))
print("\nINFLUENTIAL GRADE SOURCE AUDIT COMPLETED")
