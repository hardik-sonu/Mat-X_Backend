from pathlib import Path
sections = {
    "training/regression/run_common_feature_source_heldout.py": [
        (20, 115),
    ],
    "training/regression/run_source_heldout_uncertainty_validation.py": [
        (123, 220),
    ],
}
for filename, ranges in sections.items():
    path = Path(filename)
    print("\n" + "=" * 100)
    print(filename)
    print("=" * 100)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for start, end in ranges:
        for i in range(start, min(end, len(lines)) + 1):
            print(f"{i:04d}: {lines[i-1]}")
print("\nRead-only inspection complete.")
