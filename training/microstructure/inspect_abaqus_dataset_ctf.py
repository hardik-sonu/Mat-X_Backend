from pathlib import Path
root = Path(
    r"C:\Users\Ahmadcomputer\Downloads\MatX_Dataset"
    r"\Grain orientation and cyclic behavior"
)
files = list(root.rglob("*.ctf"))
print("CTF FILES FOUND:", len(files))
for path in files:
    print("\n" + "=" * 90)
    print("FILE:", path)
    print("SIZE:", path.stat().st_size, "bytes")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip())
    print("\nHEADER AND FIRST RECORDS:")
    for i, line in enumerate(lines[:60], start=1):
        print(f"{i:03d}: {line}")
    print("\nFIRST FIVE DATA RECORDS:")
    data_start = next(
        (i for i, line in enumerate(lines)
         if line.strip().lower().startswith("phase")),
        None
    )
    if data_start is not None:
        for line in lines[data_start:data_start + 6]:
            print(line)
    else:
        print("No Phase column header found in first 100 lines.")
