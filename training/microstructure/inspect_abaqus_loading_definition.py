from pathlib import Path
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
files = [
    root / "AbaqusInput_AmpTmp" / "Amplitude.inp",
    root / "AbaqusInput_AmpTmp" / "Timepoints.inp",
    root / "Mesh1_8k" / "Job_Hom_316L.inp",
]
for path in files:
    print("\n" + "=" * 100)
    print("FILE:", path)
    print("=" * 100)
    if not path.exists():
        print("FILE NOT FOUND")
        continue
    lines = path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()
    if path.name == "Job_Hom_316L.inp":
        ranges = [(85, 140)]
    else:
        ranges = [(1, len(lines))]
    for start, end in ranges:
        for i in range(start, min(end, len(lines)) + 1):
            print(f"{i:04d}: {lines[i-1]}")
print("\nLOADING DEFINITION INSPECTION COMPLETE.")
