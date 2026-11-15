from pathlib import Path
FILES = {
    "PROMOTED_YIELD": (
        Path("training/regression/train_steelbench_yield.py"),
        27, 48,
    ),
    "CALIBRATION_YIELD": (
        Path("training/regression/run_grade_grouped_uncertainty_calibration.py"),
        341, 380,
    ),
    "PROMOTED_TENSILE": (
        Path("training/regression/train_steelbench_tensile.py"),
        25, 44,
    ),
    "CALIBRATION_TENSILE": (
        Path("training/regression/run_grade_grouped_uncertainty_calibration.py"),
        341, 380,
    ),
}
for name, (path, start, end) in FILES.items():
    print("\n" + "=" * 75)
    print(name, "|", path)
    print("=" * 75)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:4}: {lines[i]}")
