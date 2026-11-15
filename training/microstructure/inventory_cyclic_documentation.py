from pathlib import Path
import re
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
extensions = {".pdf", ".docx", ".txt", ".md", ".csv", ".dat", ".inp"}
keywords = re.compile(
    r"cyclic_data|cyclic loading|strain.control|stress.control|"
    r"UMAT|user material|constitutive|fatigue|316L|"
    r"trueStress|trueStrain|amplitude|timepoints|unit system",
    re.IGNORECASE
)
files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)
print("=" * 100)
print("DOCUMENTATION AND SOURCE INVENTORY")
print("=" * 100)
print("Files found:", len(files))
for path in files:
    print(f"{path.stat().st_size:>12} bytes | {path.relative_to(root)}")
print("\n" + "=" * 100)
print("TEXTUAL EVIDENCE SEARCH")
print("=" * 100)
readable = {".txt", ".md", ".csv", ".dat", ".inp"}
matches = 0
for path in files:
    if path.suffix.lower() not in readable:
        continue
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                if keywords.search(line):
                    print(
                        f"\nFILE: {path.relative_to(root)}"
                        f"\nLINE: {line_number}"
                        f"\n{line.strip()[:500]}"
                    )
                    matches += 1
                    if matches >= 250:
                        print("\nSearch capped at 250 matching lines.")
                        break
    except Exception as exc:
        print("READ ERROR:", path, exc)
    if matches >= 250:
        break
print("\nTOTAL MATCHING LINES:", matches)
print("DOCUMENTATION INVENTORY COMPLETE. NO SOURCE FILES MODIFIED.")
