from pathlib import Path
import re
ROOT = Path(r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset")
terms = re.compile(
    r"Cyclic_data|trueStress|trueStrain|cyclic behavior|"
    r"cyclic loading|cycle number|stress amplitude|strain amplitude|"
    r"316L|temperature|MPa|hysteresis",
    re.IGNORECASE
)
extensions = {".txt", ".md", ".json", ".csv", ".inp", ".dat"}
print("=" * 90)
print("CYCLIC DATASET DOCUMENTATION SEARCH")
print("=" * 90)
matches = 0
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue
    # Avoid dumping large numerical EBSD files.
    if path.stat().st_size > 5_000_000:
        continue
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, 1):
                if terms.search(line):
                    print(
                        f"\nFILE: {path.relative_to(ROOT)}"
                        f"\nLINE {line_number}: {line.strip()[:500]}"
                    )
                    matches += 1
                    if matches >= 200:
                        raise StopIteration
    except StopIteration:
        break
    except (OSError, UnicodeError):
        continue
print("\nTOTAL MATCHED LINES:", matches)
print("\nCYCLIC DATASET DIRECTORY CONTENTS:")
folder = ROOT / "Grain orientation and cyclic behavior"
for path in sorted(folder.rglob("*")):
    if path.is_file():
        print(
            f"{path.stat().st_size:>12,} bytes | "
            f"{path.relative_to(ROOT)}"
        )
print("\nDOCUMENTATION SEARCH COMPLETE.")
