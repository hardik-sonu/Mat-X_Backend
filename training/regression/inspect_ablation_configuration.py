from pathlib import Path
files = [
    Path("training/regression/run_source_heldout_uncertainty_validation.py"),
    Path("training/regression/run_grade_grouped_uncertainty_calibration.py"),
    Path("training/regression/run_common_feature_source_heldout.py"),
]
for path in files:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    keywords = [
        "GroupShuffleSplit",
        "ExtraTreesRegressor",
        "random_state",
        "n_estimators",
        "min_samples_leaf",
        "train_test_split",
        "calibration",
        "BASE_NUMERIC",
        "BASE_CATEGORICAL",
        "FEATURES",
        "def evaluate_variant",
        "def build_pipeline",
    ]
    for i, line in enumerate(lines, start=1):
        if any(keyword in line for keyword in keywords):
            start = max(1, i - 2)
            end = min(len(lines), i + 5)
            print(f"\n--- Lines {start}-{end} ---")
            for j in range(start, end + 1):
                print(f"{j:04d}: {lines[j-1]}")
print("\nRead-only inspection complete.")
