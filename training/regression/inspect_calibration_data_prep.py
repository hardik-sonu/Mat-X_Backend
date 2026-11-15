from pathlib import Path
path = Path(
    "training/regression/run_grade_grouped_uncertainty_calibration.py"
)
lines = path.read_text(encoding="utf-8").splitlines()
for start, end in [(285, 365)]:
    print("=" * 76)
    print(f"CALIBRATION SCRIPT LINES {start}-{end}")
    print("=" * 76)
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")
