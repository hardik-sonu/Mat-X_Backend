from pathlib import Path
path = Path("app/api/routes.py")
source = path.read_text(encoding="utf-8")
anchor = "from app.services.material_identity_service import identify_material"
addition = (
    "from app.schemas.prediction_responses import (\n"
    "    TensilePredictionResponse,\n"
    "    YieldPredictionResponse,\n"
    ")\n"
)
if source.count(anchor) != 1:
    raise RuntimeError("Could not safely locate route import insertion point.")
source = source.replace(anchor, addition + anchor, 1)
replacements = {
    '@router.post("/predict/tensile-strength")':
    '@router.post("/predict/tensile-strength", response_model=TensilePredictionResponse)',
    '@router.post("/predict/yield-strength")':
    '@router.post("/predict/yield-strength", response_model=YieldPredictionResponse)',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"Could not safely locate route decorator: {old}")
    source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
print("Prediction response schemas connected successfully.")
