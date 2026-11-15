from pathlib import Path
import joblib
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "tensile_strength": MODEL_DIR / "steelbench_tensile_extratrees.joblib",
    "yield_strength": MODEL_DIR / "steelbench_yield_extratrees.joblib",
    "inclusion_classifier": MODEL_DIR / "matx_inclusion_eds_extratrees.joblib",
}
MODELS = {}
MODEL_ERRORS = {}
def load_models():
    MODELS.clear()
    MODEL_ERRORS.clear()
    for name, path in MODEL_FILES.items():
        try:
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {path.name}")
            MODELS[name] = joblib.load(path)
        except Exception as exc:
            MODEL_ERRORS[name] = str(exc)
def get_model_status():
    return {
        name: {
            "status": "available" if name in MODELS else "unavailable",
            "error": MODEL_ERRORS.get(name),
        }
        for name in MODEL_FILES
    }
load_models()
