from pathlib import Path
import pandas as pd
ROOT = Path.cwd()
DIRECTORIES = [
    ROOT / "training_results" / "source_heldout_validation_v1",
    ROOT / "training_results" / "common_feature_source_heldout_v1",
]
for directory in DIRECTORIES:
    print("\n" + "=" * 90)
    print("DIRECTORY:", directory)
    print("=" * 90)
    if not directory.exists():
        print("NOT FOUND")
        continue
    files = sorted(directory.rglob("*.csv"))
    print("Total CSV files:", len(files))
    for path in files:
        relative = path.relative_to(directory)
        try:
            data = pd.read_csv(path, nrows=3)
            print("\nFILE:", relative)
            print("Columns:", list(data.columns))
            if {"actual", "prediction"}.issubset(data.columns):
                print("PREDICTION FILE: YES")
                print("Sample:")
                print(data.head(2).to_string(index=False))
            else:
                print("PREDICTION FILE: NO")
        except Exception as exc:
            print("\nREAD ERROR:", relative, exc)
print("\nRead-only inspection. No files modified.")
