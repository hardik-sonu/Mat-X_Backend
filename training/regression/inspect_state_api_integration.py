from pathlib import Path
files = [
    "app/api/routes.py",
    "app/schemas/investigation.py",
    "app/schemas/material_identity.py",
    "app/services/process_history_service.py",
    "tests/test_engine_api.py",
]
for filename in files:
    path = Path(filename)
    print("\n" + "=" * 85)
    print(f"FILE: {filename}")
    print("=" * 85)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if filename.endswith("routes.py"):
        ranges = [(1, min(100, len(lines)))]
    elif filename.endswith("test_engine_api.py"):
        ranges = [(1, min(180, len(lines)))]
    else:
        ranges = [(1, len(lines))]
    for start, end in ranges:
        for number in range(start, end + 1):
            print(f"{number}: {lines[number - 1]}")
print("\n=== API INTEGRATION INSPECTION COMPLETE ===")
