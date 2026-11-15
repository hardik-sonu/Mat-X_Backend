from pathlib import Path
FILES = [
    Path("training/regression/train_steelbench_tensile.py"),
    Path("training/regression/train_steelbench_yield.py"),
    Path("training/regression/run_grade_grouped_uncertainty_calibration.py"),
]
PATTERNS = [
    "GroupShuffleSplit",
    "grouped_split",
    "test_size",
    "random_state",
    "grade_id",
    "EMK0319",
    "EMK0320",
    "EMK0321",
    "yield_strength",
    "dropna",
    "train_test_split",
]
for path in FILES:
    print("\n" + "=" * 80)
    print("FILE:", path)
    print("=" * 80)
    if not path.exists():
        print("FILE NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    matched = set()
    for i, line in enumerate(lines):
        if any(pattern in line for pattern in PATTERNS):
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            for j in range(start, end):
                if j not in matched:
                    print(f"{j + 1:4}: {lines[j]}")
                    matched.add(j)
