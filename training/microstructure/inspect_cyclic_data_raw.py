from pathlib import Path
import numpy as np
import pandas as pd
path = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Cyclic_data.dat"
)
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
print("=" * 90)
print("CYCLIC DATA: RAW FILE AND COLUMN FORENSIC INSPECTION")
print("=" * 90)
print("File:", path)
print("Exists:", path.exists())
print("Size:", path.stat().st_size, "bytes")
print("Total lines:", len(lines))
print("\nFIRST 30 RAW LINES:")
for i, line in enumerate(lines[:30], 1):
    print(f"{i:04d}: {line}")
print("\nLAST 15 RAW LINES:")
start = max(0, len(lines) - 15)
for i, line in enumerate(lines[start:], start + 1):
    print(f"{i:04d}: {line}")
print("\nNON-NUMERIC / COMMENT / POSSIBLE HEADER LINES:")
for i, line in enumerate(lines, 1):
    tokens = line.strip().split(";")
    try:
        [float(x.strip().replace("D", "E").replace("d", "e")) for x in tokens]
    except ValueError:
        print(f"{i:04d}: {line}")
records = []
for line_number, line in enumerate(lines, 1):
    tokens = line.strip().split(";")
    if len(tokens) != 6:
        continue
    try:
        values = [
            float(x.strip().replace("D", "E").replace("d", "e"))
            for x in tokens
        ]
        records.append(values)
    except ValueError:
        continue
df = pd.DataFrame(
    records,
    columns=[f"column_{i}" for i in range(1, 7)]
)
print("\nPARSED DATAFRAME:", df.shape)
print("\nCOLUMN STATISTICS:")
print(df.describe().to_string())
print("\nUNIQUE VALUES IN EACH COLUMN (UP TO 20):")
for col in df.columns:
    values = df[col].dropna().unique()
    print(f"\n{col}: {len(values)} unique")
    print(values[:20])
print("\nCORRELATION MATRIX:")
print(df.corr(numeric_only=True).to_string())
print("\nPOSSIBLE BINARY / CONSTANT COLUMNS:")
for col in df.columns:
    values = sorted(df[col].dropna().unique().tolist())
    if len(values) <= 10:
        print(col, values)
print("\nINSPECTION COMPLETE. Original file was not modified.")
