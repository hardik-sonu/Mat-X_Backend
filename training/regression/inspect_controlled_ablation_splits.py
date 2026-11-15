from pathlib import Path
checks = {
    "training/regression/run_common_feature_source_heldout.py": [
        (1, 125),
    ],
    "training/regression/run_grade_grouped_uncertainty_calibration.py": [
        (1, 180),
    ],
    "training/regression/run_source_heldout_uncertainty_validation.py": [
        (1, 190),
    ],
}
for filename, ranges in checks.items():
    path = Path(filename)
    print("\n" + "=" * 100)
    print("FILE:", filename)
    print("=" * 100)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for start, end in ranges:
        end = min(end, len(lines))
        for i in range(start, end + 1):
            print(f"{i:04d}: {lines[i-1]}")
print("\nRead-only inspection complete.")
