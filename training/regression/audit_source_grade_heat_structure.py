from pathlib import Path
import pandas as pd
DATASET = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset\SteelBench v1.0.csv"
)
df = pd.read_csv(DATASET, low_memory=False)
print("=" * 76)
print("MAT-X SOURCE / GRADE / HEAT STRUCTURE AUDIT")
print("=" * 76)
required = ["source", "grade_id", "heat_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")
for column in required:
    df[column] = df[column].astype("string").str.strip()
print("\n1. SOURCE SUMMARY")
summary = df.groupby("source", dropna=False).agg(
    records=("source", "size"),
    grades=("grade_id", "nunique"),
    heats=("heat_id", "nunique"),
)
print(summary.to_string())
print("\n2. GRADE-SOURCE OVERLAP")
grade_sources = (
    df.groupby("grade_id", dropna=False)["source"]
    .nunique()
)
print(f"Unique grades: {df['grade_id'].nunique(dropna=True)}")
print(f"Grades appearing in multiple sources: {(grade_sources > 1).sum()}")
print(f"Maximum sources per grade: {grade_sources.max()}")
print("\nGrades appearing in multiple sources:")
multi = grade_sources[grade_sources > 1].index
if len(multi):
    print(
        df[df["grade_id"].isin(multi)]
        .groupby("grade_id")["source"]
        .agg(lambda x: ", ".join(sorted(set(x.dropna()))))
        .to_string()
    )
else:
    print("None")
print("\n3. GRADE COUNTS BY SOURCE")
grade_table = (
    df.dropna(subset=["source", "grade_id"])
    .drop_duplicates(["source", "grade_id"])
    .groupby("source")
    .size()
)
print(grade_table.to_string())
print("\n4. HEAT IDS SHARED ACROSS SOURCES")
heat_sources = df.groupby("heat_id")["source"].nunique()
print(f"Heats appearing in multiple sources: {(heat_sources > 1).sum()}")
print("\n5. MISSING IDENTIFIERS")
for column in required:
    print(f"{column}: {df[column].isna().sum()} missing")
print("\n6. SOURCE-GRADE-HEAT SAMPLE")
print(
    df[required]
    .drop_duplicates()
    .sort_values(["source", "grade_id", "heat_id"])
    .groupby("source", dropna=False)
    .head(5)
    .to_string(index=False)
)
print("\nNo dataset, model, or calibration artifacts were modified.")
