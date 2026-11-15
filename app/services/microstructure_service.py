import hashlib
from pathlib import Path
from app.engines.microstructure_feature_engine import (
    DATASET_ROOT,
    discover_ebsd_maps,
    extract_ebsd_features,
)
def _map_id(file_path: Path) -> str:
    relative = file_path.resolve().relative_to(DATASET_ROOT.resolve())
    return hashlib.sha256(
        str(relative).encode("utf-8")
    ).hexdigest()[:16]
def list_microstructure_maps():
    maps = []
    for file_path in discover_ebsd_maps():
        relative = file_path.resolve().relative_to(DATASET_ROOT.resolve())
        maps.append({
            "map_id": _map_id(file_path),
            "source_file": file_path.name,
            "relative_path": str(relative),
            "format": "H5OINA",
        })
    return {
        "dataset_root": str(DATASET_ROOT),
        "map_count": len(maps),
        "maps": maps,
    }
def get_microstructure_features(map_id: str):
    if not map_id or len(map_id) != 16:
        raise ValueError("Invalid microstructure map ID.")
    for file_path in discover_ebsd_maps():
        if _map_id(file_path) == map_id:
            features = extract_ebsd_features(file_path)
            features["map_id"] = map_id
            return features
    raise FileNotFoundError(
        f"No EBSD map found for map ID: {map_id}"
    )
from app.engines.ebsd_provenance_engine import extract_ebsd_provenance
def get_unified_microstructure_record(map_id: str):
    """
    Combine EBSD microstructure features and characterization provenance
    for the same source map. Original source files remain unchanged.
    """
    if not map_id or len(map_id) != 16:
        raise ValueError("Invalid microstructure map ID.")
    for file_path in discover_ebsd_maps():
        if _map_id(file_path) == map_id:
            features = extract_ebsd_features(file_path)
            features["map_id"] = map_id
            provenance = extract_ebsd_provenance(file_path)
            if provenance.get("status") != "CHARACTERIZATION_METADATA_EXTRACTED":
                raise ValueError(
                    "Characterization provenance extraction was unsuccessful."
                )
            return {
                "engine": "MAT-X Unified Microstructure Record Engine",
                "engine_version": "1.0.0",
                "status": "UNIFIED_RECORD_GENERATED",
                "map_id": map_id,
                "microstructure_features": features,
                "characterization_provenance": provenance,
                "scientific_scope": {
                    "manufacturing_history_established": False,
                    "missing_values_imputed": False,
                    "grain_size_method_is_approximate": True,
                    "characterization_metadata_is_not_manufacturing_history": True,
                },
            }
    raise FileNotFoundError(
        f"No EBSD map found for map ID: {map_id}"
    )
