from pathlib import Path
import re
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior\Dataset\Models"
)
model = root / "Mesh1_8k"
files = [
    model / "AbaqusInput" / "Hom_316L_bc.inp",
    model / "AbaqusInput" / "Hom_316L_ANodes.inp",
    model / "AbaqusInput" / "Hom_316L_grains.inp",
    model / "Job_Hom_316L.inp",
]
for path in files:
    print("\n" + "=" * 100)
    print("FILE:", path)
    print("=" * 100)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()
    print("TOTAL LINES:", len(lines))
    if "bc.inp" in path.name or "ANodes" in path.name:
        for i, line in enumerate(lines, 1):
            print(f"{i:05d}: {line}")
    else:
        patterns = re.compile(
            r"REFERENCE|RP1|N_RP|N_RPALL|"
            r"\*NODE|^\s*\*NSET|^\s*\*BOUNDARY|"
            r"^\s*\*EQUATION",
            re.IGNORECASE
        )
        for i, line in enumerate(lines, 1):
            if patterns.search(line):
                print(f"{i:05d}: {line}")
print("\nREFERENCE POINT AND BOUNDARY AUDIT COMPLETE.")
print("ORIGINAL FILES NOT MODIFIED.")
