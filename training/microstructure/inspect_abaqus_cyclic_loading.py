from pathlib import Path
import re
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
files = [
    root / "AbaqusInput_AmpTmp" / "Amplitude.inp",
    root / "AbaqusInput_AmpTmp" / "Timepoints.inp",
]
files += list(root.glob("Mesh*/Job_*.inp"))
for path in files:
    print("\n" + "=" * 100)
    print("FILE:", path)
    print("EXISTS:", path.exists())
    if not path.exists():
        continue
    lines = path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()
    print("TOTAL LINES:", len(lines))
    if len(lines) <= 250:
        for i, line in enumerate(lines, 1):
            print(f"{i:04d}: {line}")
    else:
        keywords = re.compile(
            r"^\s*\*(AMPLITUDE|STEP|STATIC|BOUNDARY|INCLUDE|OUTPUT|TIME|"
            r"NODE|ELEMENT|INITIAL CONDITIONS)",
            re.IGNORECASE
        )
        for i, line in enumerate(lines, 1):
            if keywords.search(line):
                print(f"{i:04d}: {line}")
print("\nINSPECTION COMPLETE. No input files modified.")
