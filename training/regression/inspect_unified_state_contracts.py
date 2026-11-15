from pathlib import Path
FILES = [
    "app/engines/active_investigation_engine.py",
    "app/engines/material_identity_engine.py",
    "app/engines/process_history_engine.py",
    "app/services/microstructure_service.py",
    "app/services/prediction_service.py",
    "app/schemas/investigation.py",
    "app/schemas/material_identity.py",
    "app/schemas/predictions.py",
    "app/api/routes.py",
]
for filename in FILES:
    path = Path(filename)
    print("\n" + "=" * 100)
    print(f"FILE: {filename}")
    print("=" * 100)
    if not path.exists():
        print("NOT FOUND")
        continue
    lines = path.read_text(
        encoding="utf-8-sig",
        errors="replace"
    ).splitlines()
    print(f"Total lines: {len(lines)}")
    for number, line in enumerate(lines, 1):
        print(f"{number:04d}: {line}")
print("\n=== CONTRACT INSPECTION COMPLETE ===")
print("Read-only inspection; no project files modified.")
