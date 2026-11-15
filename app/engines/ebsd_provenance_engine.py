from pathlib import Path
import hashlib
import h5py
import numpy as np
from app.engines.microstructure_feature_engine import (
    DATASET_ROOT,
    discover_ebsd_maps,
)
ENGINE_VERSION = "1.0.0"
FIELD_ALIASES = {
    "manufacturer": ["Manufacturer"],
    "acquisition_date": ["Acquisition Date"],
    "project_label": ["Project Label"],
    "specimen_label": ["Specimen Label"],
    "site_label": ["Site Label"],
    "beam_voltage": ["Beam Voltage"],
    "magnification": ["Magnification"],
    "working_distance": ["Working Distance"],
    "x_step": ["X Step"],
    "y_step": ["Y Step"],
    "x_cells": ["X Cells"],
    "y_cells": ["Y Cells"],
    "tilt_angle": ["Tilt Angle"],
    "stage_rotation": ["Stage Position/Rotation"],
    "stage_tilt": ["Stage Position/Tilt"],
}
def _clean(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        converted = [_clean(item) for item in value.tolist()]
        return converted[0] if len(converted) == 1 else converted
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, (list, tuple)):
        converted = [_clean(item) for item in value]
        return converted[0] if len(converted) == 1 else converted
    return value
def _find_metadata(h5, aliases):
    wanted = {item.lower() for item in aliases}
    found = []
    def inspect(name, obj):
        basename = name.rsplit("/", 1)[-1].lower()
        normalized = name.lower()
        matches = (
            basename in wanted
            or any(
                normalized.endswith("/" + alias.lower())
                for alias in aliases
            )
        )
        if not matches:
            return
        entry = {
            "source_path": name,
            "value": None,
            "unit": None,
        }
        if isinstance(obj, h5py.Dataset):
            try:
                entry["value"] = _clean(obj[()])
            except Exception:
                entry["value"] = None
        if "Unit" in obj.attrs:
            entry["unit"] = _clean(obj.attrs["Unit"])
        found.append(entry)
    h5.visititems(inspect)
    # Prefer the layered-image header where multiple copies exist.
    found.sort(
        key=lambda item: (
            "layered image" not in item["source_path"].lower(),
            len(item["source_path"])
        )
    )
    return found[0] if found else None
def extract_ebsd_provenance(file_path):
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    relative_path = str(
        file_path.relative_to(DATASET_ROOT.resolve())
    )
    fields = {}
    with h5py.File(file_path, "r") as h5:
        for field, aliases in FIELD_ALIASES.items():
            fields[field] = _find_metadata(h5, aliases)
    return {
        "engine": "MAT-X EBSD Characterization Provenance Engine",
        "version": ENGINE_VERSION,
        "status": "CHARACTERIZATION_METADATA_EXTRACTED",
        "source": {
            "dataset_root_relative_path": relative_path,
            "file_name": file_path.name,
            "sha256": hashlib.sha256(
                file_path.read_bytes()
            ).hexdigest(),
        },
        "characterization_metadata": fields,
        "manufacturing_history_established": False,
        "missing_values_imputed": False,
        "limitations": [
            "Acquisition metadata describes characterization, not manufacturing.",
            "Missing metadata is not inferred.",
            "Instrument manufacturer does not establish instrument model.",
            "Acquisition records do not establish independent specimen count.",
        ],
    }
def extract_all_ebsd_provenance():
    results = []
    errors = []
    for file_path in discover_ebsd_maps():
        try:
            results.append(extract_ebsd_provenance(file_path))
        except Exception as exc:
            errors.append({
                "file": str(file_path),
                "error": str(exc),
            })
    return {
        "engine": "MAT-X EBSD Characterization Provenance Engine",
        "version": ENGINE_VERSION,
        "files_discovered": len(results) + len(errors),
        "files_extracted": len(results),
        "failed_files": len(errors),
        "records": results,
        "errors": errors,
    }
