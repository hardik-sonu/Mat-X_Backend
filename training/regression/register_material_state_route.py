from pathlib import Path
path = Path("app/api/routes.py")
source = path.read_text(encoding="utf-8-sig")
schema_import = "from app.schemas.material_state import MaterialStateReconstructionInput\n"
service_import = "from app.services.material_state_service import reconstruct_material_state\n"
if schema_import not in source:
    marker = "from app.schemas.material_identity import MaterialIdentityInput\n"
    if marker not in source:
        raise RuntimeError("Could not find schema import insertion point.")
    source = source.replace(marker, marker + schema_import, 1)
if service_import not in source:
    marker = "from app.services.material_identity_service import identify_material\n"
    if marker not in source:
        raise RuntimeError("Could not find service import insertion point.")
    source = source.replace(marker, marker + service_import, 1)
route_marker = "@router.get(\"/health\")"
route_code = '''@router.post("/material/state/reconstruct")
def material_state_reconstruction(
    request: MaterialStateReconstructionInput
):
    try:
        return reconstruct_material_state(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
'''
if "def material_state_reconstruction(" not in source:
    if route_marker not in source:
        raise RuntimeError("Could not find route insertion point.")
    source = source.replace(route_marker, route_code + route_marker, 1)
path.write_text(source, encoding="utf-8")
print("Registered POST /material/state/reconstruct")
