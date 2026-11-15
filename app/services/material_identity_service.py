from app.engines.material_identity_engine import identify_material_candidates
def identify_material(data):
    return identify_material_candidates(
        composition=data.composition,
        material_family=data.material_family,
        top_k=data.top_k,
    )
