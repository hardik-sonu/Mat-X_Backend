from pathlib import Path
import re
import pandas as pd
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset"
)
files = [
    root / "Models" / "AbaqusInput_AmpTmp" / "Amplitude.inp",
    root / "Models" / "AbaqusInput_AmpTmp" / "Timepoints.inp",
    root / "Cyclic_data.dat",
]
for path in files:
    print("\n" + "=" * 100)
    print("FILE:", path)
    print("EXISTS:", path.exists())
    if not path.exists():
        continue
    print("SIZE:", path.stat().st_size, "bytes")
    if path.suffix.lower() == ".inp":
        lines = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()
        print("TOTAL LINES:", len(lines))
        for i, line in enumerate(lines[:100], start=1):
            print(f"{i:04d}: {line}")
    elif path.name == "Cyclic_data.dat":
        df = pd.read_csv(path, sep=";")
        df.columns = [
            c.strip().lstrip("#").strip()
            for c in df.columns
        ]
        print("COLUMNS:", list(df.columns))
        print("\nFIRST 12 RECORDS:")
        print(df.head(12).to_string(index=False))
        print("\nLAST 12 RECORDS:")
        print(df.tail(12).to_string(index=False))
        print("\nCYCLE COUNTS:")
        print(df.groupby("cycl").size().to_string())
        print("\nCYCLE DATA TIME RANGES:")
        print(
            df.groupby("cycl")["time"]
            .agg(["count", "min", "max"])
            .to_string()
        )
print("\nINSPECTION COMPLETE. NO FILES MODIFIED.")
