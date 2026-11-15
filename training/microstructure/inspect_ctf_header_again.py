from pathlib import Path
ROOT = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
) / "Grain orientation and cyclic behavior"
files = list(ROOT.rglob("*.ctf"))
print("CTF FILES:", len(files))
for path in files:
    print("\nFILE:", path)
    print("SIZE:", path.stat().st_size)
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if (
                "Phase" in stripped
                or "Euler1" in stripped
                or "Channel Text File" in stripped
            ):
                print(f"LINE {i}: {stripped!r}")
            if i >= 35:
                break
