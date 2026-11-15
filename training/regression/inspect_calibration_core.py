from pathlib import Path
path = Path(
    "training/regression/run_grade_grouped_uncertainty_calibration.py"
)
lines = path.read_text(encoding="utf-8").splitlines()
for start, end in [(1, 76), (77, 155), (156, 290)]:
    print("\n" + "=" * 72)
    print(f"LINES {start}-{end}")
    print("=" * 72)
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")
