from pathlib import Path
import ast
checks = {
    "app/api/routes.py": (1, 33),
    "app/engines/active_investigation_engine.py": (1, 63),
    "app/services/process_history_service.py": (1, 57),
}
for filename, (start, end) in checks.items():
    path = Path(filename)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    print("\n" + "=" * 85)
    print(filename)
    print("=" * 85)
    for i in range(start, min(end, len(lines)) + 1):
        print(f"{i}: {lines[i-1]}")
print("\n=== EVIDENCE AND IMPORT INSPECTION COMPLETE ===")
