from pathlib import Path
path = Path(
    "training/regression/run_grade_grouped_uncertainty_calibration.py"
)
lines = path.read_text(encoding="utf-8").splitlines()
sections = [
    (1, 125),
    (125, 290),
    (290, 490),
]
for start, end in sections:
    print("\n" + "=" * 90)
    print(f"SCRIPT LINES {start}-{min(end, len(lines))}")
    print("=" * 90)
    for index in range(start - 1, min(end, len(lines))):
        print(f"{index + 1:04d}: {lines[index]}")
