from pathlib import Path
path = Path("app/services/prediction_service.py")
source = path.read_text(encoding="utf-8")
old_import = "from app.core.model_registry import MODELS\n"
new_import = (
    "from app.core.model_registry import MODELS\n"
    "from app.engines.applicability_engine import assess_applicability\n"
)
if source.count(old_import) != 1:
    raise RuntimeError("Could not safely identify the model registry import.")
source = source.replace(old_import, new_import, 1)
old_tensile = (
    '"input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),\n'
    '        "target": "Tensile strength",'
)
new_tensile = (
    '"input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),\n'
    '        "applicability": assess_applicability("tensile_strength", payload),\n'
    '        "target": "Tensile strength",'
)
if source.count(old_tensile) != 1:
    raise RuntimeError("Could not safely identify the tensile response.")
source = source.replace(old_tensile, new_tensile, 1)
old_yield = (
    '"input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),\n'
    '        "target": "Yield strength",'
)
new_yield = (
    '"input_evidence": build_input_evidence(model, payload, TENSILE_FEATURES),\n'
    '        "applicability": assess_applicability("yield_strength", payload),\n'
    '        "target": "Yield strength",'
)
if source.count(old_yield) != 1:
    raise RuntimeError("Could not safely identify the yield response.")
source = source.replace(old_yield, new_yield, 1)
path.write_text(source, encoding="utf-8")
print("Prediction service updated successfully.")
