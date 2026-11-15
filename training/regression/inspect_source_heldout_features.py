from pathlib import Path
import joblib
script = Path(
    "training/regression/run_source_heldout_uncertainty_validation.py"
)
print("=" * 80)
print("SOURCE-HELD-OUT FEATURE DEFINITIONS")
print("=" * 80)
lines = script.read_text(encoding="utf-8-sig").splitlines()
for start, end in [(1, 130), (380, 410)]:
    print(f"\n--- Lines {start}-{end} ---")
    for number in range(start, min(end, len(lines)) + 1):
        print(f"{number:04}: {lines[number - 1]}")
print("\n" + "=" * 80)
print("PROMOTED MODEL FEATURE INSPECTION")
print("=" * 80)
for filename in [
    "steelbench_tensile_extratrees.joblib",
    "steelbench_yield_extratrees.joblib",
]:
    path = Path("models") / filename
    print(f"\n{filename}")
    if not path.exists():
        print("FILE NOT FOUND")
        continue
    model = joblib.load(path)
    print("Artifact type:", type(model).__name__)
    if isinstance(model, dict):
        print("Dictionary keys:", list(model.keys()))
        for key, value in model.items():
            print(f"  {key}: {type(value).__name__}")
            if hasattr(value, "feature_names_in_"):
                print("  Features:", list(value.feature_names_in_))
            if hasattr(value, "named_steps"):
                for step_name, step in value.named_steps.items():
                    if hasattr(step, "feature_names_in_"):
                        print(
                            f"  {step_name} features:",
                            list(step.feature_names_in_)
                        )
    elif hasattr(model, "feature_names_in_"):
        print("Features:", list(model.feature_names_in_))
    elif hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            print(f"Step: {step_name} ({type(step).__name__})")
            if hasattr(step, "feature_names_in_"):
                print("Features:", list(step.feature_names_in_))
