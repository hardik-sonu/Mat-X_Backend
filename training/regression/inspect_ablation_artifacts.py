from pathlib import Path
import joblib
directory = Path(
    "training_results/controlled_feature_ablation_v1/models"
)
for path in sorted(directory.glob("*full_13.joblib")):
    model = joblib.load(path)
    print(f"\nMODEL: {path.name}")
    print("TYPE:", type(model).__name__)
    if hasattr(model, "named_steps"):
        print("PIPELINE STEPS:", list(model.named_steps.keys()))
        for step_name, step in model.named_steps.items():
            print(
                "STEP:",
                step_name,
                "| TYPE:",
                type(step).__name__,
            )
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = step.get_feature_names_out()
                    print("TRANSFORMED FEATURES:", list(names))
                except Exception as exc:
                    print("FEATURE NAME ERROR:", repr(exc))
    estimator = (
        model.named_steps.get("model")
        if hasattr(model, "named_steps")
        else model
    )
    if estimator is not None and hasattr(
        estimator, "feature_importances_"
    ):
        print(
            "IMPORTANCE COUNT:",
            len(estimator.feature_importances_)
        )
