from pathlib import Path
import hashlib
import numpy as np
import h5py
from app.core.paths import DATASET_ROOT
DATA_PATH = "1/Data Processing/Data/"
HEADER_PATH = "1/EBSD/Header/"
PHASE_PATH = "1/Data Processing/Header/Phases/"
GRAIN_PATH = "1/Data Processing/Analyses/GrainSizing/Data/Grain Index"
KAM_PATH = "1/Data Processing/Analyses/KAM1/Data/Misorientation Angle"

def _sha256_file(file_path, chunk_size=1024 * 1024):
    """Calculate SHA-256 without loading the complete file into memory."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _validate_scan_geometry(x_cells, y_cells, x_step, y_step, total_points):
    """Validate EBSD grid dimensions and physical scan-step measurements."""
    if x_cells <= 0 or y_cells <= 0:
        raise ValueError("EBSD scan dimensions must be positive.")
    if x_cells * y_cells != total_points:
        raise ValueError("EBSD scan dimensions do not match point count.")
    if (
        not np.isfinite(x_step)
        or not np.isfinite(y_step)
        or x_step <= 0
        or y_step <= 0
    ):
        raise ValueError("EBSD scan steps must be finite and positive.")
    scan_area = float(x_cells * x_step * y_cells * y_step)
    if not np.isfinite(scan_area) or scan_area <= 0:
        raise ValueError("EBSD scan area is invalid.")
    return scan_area
def _scalar(h5, path, default=None):
    if path not in h5:
        return default
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return value.tolist()
    return value.item() if hasattr(value, "item") else value
def _text(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value) if value is not None else None
def _statistics(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{name}": None for name in
                ("mean", "median", "std", "p90", "p95", "min", "max")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }
def discover_ebsd_maps():
    return sorted(
        path for path in DATASET_ROOT.rglob("*.h5oina")
        if "EBSD" in str(path)
    )
def extract_ebsd_features(file_path):
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"EBSD file not found: {file_path}")
    if file_path.suffix.lower() != ".h5oina":
        raise ValueError("Only .h5oina files are supported by this extractor.")
    with h5py.File(file_path, "r") as h5:
        required = [
            DATA_PATH + "Euler",
            DATA_PATH + "Phase",
            DATA_PATH + "Error",
        ]
        missing = [path for path in required if path not in h5]
        if missing:
            raise ValueError(f"Required EBSD datasets missing: {missing}")
        phase = h5[DATA_PATH + "Phase"][:]
        error = h5[DATA_PATH + "Error"][:]
        euler = h5[DATA_PATH + "Euler"]
        if euler.ndim != 2 or euler.shape[1] != 3:
            raise ValueError("Unexpected Euler orientation array shape.")
        if len(phase) != len(error) or len(phase) != euler.shape[0]:
            raise ValueError("EBSD arrays have inconsistent point counts.")
        x_cells = int(_scalar(h5, HEADER_PATH + "X Cells", 0))
        y_cells = int(_scalar(h5, HEADER_PATH + "Y Cells", 0))
        x_step = float(_scalar(h5, HEADER_PATH + "X Step", np.nan))
        y_step = float(_scalar(h5, HEADER_PATH + "Y Step", np.nan))
        total_points = int(len(phase))
        scan_area = _validate_scan_geometry(
            x_cells, y_cells, x_step, y_step, total_points
        )
        hit_rate = _scalar(h5, HEADER_PATH + "Hit Rate")
        success_code = 1
        error_dataset = h5[DATA_PATH + "Error"]
        if "Success" in error_dataset.attrs:
            success_code = int(np.asarray(
                error_dataset.attrs["Success"]
            ).reshape(-1)[0])
        indexed = (error == success_code) & (phase > 0)
        indexed_points = int(np.count_nonzero(indexed))
        result = {
            "source_file": file_path.name,
            "source_path": str(file_path),
            "source_sha256": _sha256_file(file_path),
            "extraction_version": "1.0.0",
            "data_type": "EBSD",
            "total_points": total_points,
            "x_cells": x_cells,
            "y_cells": y_cells,
            "x_step_um": x_step,
            "y_step_um": y_step,
            "scan_area_um2": scan_area,
            "header_hit_rate_percent": (
                float(hit_rate) if hit_rate is not None else None
            ),
            "indexed_points": indexed_points,
            "indexed_fraction": (
                float(indexed_points / total_points)
                if total_points else None
            ),
            "unindexed_points": total_points - indexed_points,
            "unindexed_fraction": (
                float(1 - indexed_points / total_points)
                if total_points else None
            ),
            "phase_fractions": {},
            "kam_available": KAM_PATH in h5,
            "grain_index_available": GRAIN_PATH in h5,
            "provenance": "SOURCE_DATA_AND_DERIVED_FEATURES",
        }
        # Build phase definitions from the file metadata.
        phase_definitions = {}
        if PHASE_PATH in h5:
            for phase_key in h5[PHASE_PATH]:
                phase_group = PHASE_PATH + phase_key + "/"
                phase_definitions[int(phase_key)] = {
                    "name": _text(_scalar(
                        h5, phase_group + "Phase Name"
                    )),
                    "space_group": _scalar(
                        h5, phase_group + "Space Group"
                    ),
                }
        def build_phase_fractions(mask, denominator):
            fractions = {}
            selected_phase = phase[mask]
            ids, counts = np.unique(selected_phase, return_counts=True)
            for phase_id, count in zip(ids, counts):
                phase_id = int(phase_id)
                if phase_id == 0:
                    label = "Unindexed / no phase"
                    definition = {}
                else:
                    definition = phase_definitions.get(phase_id, {})
                    label = definition.get("name") or f"phase_{phase_id}"
                fractions[label] = {
                    "phase_id": phase_id,
                    "point_count": int(count),
                    "fraction": (
                        float(count / denominator)
                        if denominator else None
                    ),
                    "space_group": definition.get("space_group"),
                }
            return fractions
        all_points_mask = np.ones(total_points, dtype=bool)
        # Keep the legacy field, explicitly defined over all map points.
        result["phase_fractions"] = build_phase_fractions(
            all_points_mask, total_points
        )
        result["phase_fractions_all_points"] = result["phase_fractions"]
        # Phase fractions among successfully indexed points only.
        result["phase_fractions_indexed_points"] = build_phase_fractions(
            indexed, indexed_points
        )
        # Preserve counts for every indexing/error code.
        error_labels = {
            0: "NotAnalyzed",
            1: "Success",
            2: "NoSolution",
            3: "LowBandContrast",
            4: "LowBandSlope",
            5: "HighMAD",
            6: "UnexpectedError",
            7: "Replaced",
        }
        error_dataset = h5[DATA_PATH + "Error"]
        for attr_name, attr_value in error_dataset.attrs.items():
            try:
                code = int(np.asarray(attr_value).reshape(-1)[0])
                error_labels[code] = str(attr_name)
            except (TypeError, ValueError, IndexError):
                continue
        error_codes, error_counts = np.unique(error, return_counts=True)
        result["indexing_error_counts"] = {}
        for code, count in zip(error_codes, error_counts):
            code = int(code)
            label = error_labels.get(code, f"Unknown_{code}")
            result["indexing_error_counts"][label] = {
                "error_code": code,
                "point_count": int(count),
                "fraction": float(count / total_points),
            }
        result["indexing_success_code"] = int(success_code)
        result["indexing_success_count"] = indexed_points
        result["indexing_success_fraction"] = (
            float(indexed_points / total_points)
            if total_points else None
        )
        # Mean Angular Deviation is stored in radians.
        mad_path = DATA_PATH + "Mean Angular Deviation"
        if mad_path in h5:
            mad = h5[mad_path][:]
            result.update(_statistics(
                np.degrees(mad[indexed]), "mad_deg"
            ))
        else:
            result["mad_deg_mean"] = None
        # KAM is stored in radians; convert to degrees.
        if KAM_PATH in h5:
            kam = h5[KAM_PATH][:]
            if len(kam) != total_points:
                raise ValueError("KAM array length does not match EBSD map.")
            result.update(_statistics(
                np.degrees(kam[indexed]), "kam_deg"
            ))
            result["kam_provenance"] = "SOURCE_MEASUREMENT"
        else:
            result["kam_deg_mean"] = None
            result["kam_provenance"] = "UNAVAILABLE"
        # Grain-size estimates from pixel counts and scan step.
        # Equivalent circular diameter assumes each indexed grain is
        # represented by its indexed pixel area.
        if GRAIN_PATH in h5:
            grain_index = h5[GRAIN_PATH][:]
            if len(grain_index) != total_points:
                raise ValueError(
                    "Grain-index array length does not match EBSD map."
                )
            valid_grains = grain_index[indexed]
            valid_grains = valid_grains[valid_grains > 0]
            grain_ids, grain_counts = np.unique(
                valid_grains, return_counts=True
            )
            result["grain_count"] = int(len(grain_ids))
            if len(grain_counts) and np.isfinite(x_step) and np.isfinite(y_step):
                pixel_area = x_step * y_step
                grain_areas = grain_counts.astype(float) * pixel_area
                equivalent_diameters = np.sqrt(
                    4.0 * grain_areas / np.pi
                )
                result.update(_statistics(
                    equivalent_diameters, "grain_eq_diameter_um"
                ))
                result["grain_area_mean_um2"] = float(
                    np.mean(grain_areas)
                )
                result["grain_area_median_um2"] = float(
                    np.median(grain_areas)
                )
                result["grain_size_method"] = (
                    "2D equivalent circular diameter from indexed pixel area"
                )
            else:
                result["grain_size_method"] = "UNAVAILABLE"
        else:
            result["grain_count"] = None
            result["grain_size_method"] = "UNAVAILABLE"
        result["quality_notes"] = [
            "Grain-size values are 2D equivalent circular estimates.",
            "Only points classified as successful and with nonzero phase "
            "are used for indexed-point statistics.",
            "Phase fractions use all map points.",
            "KAM and MAD are converted from radians to degrees.",
            "Map-level features are not independent specimen-level samples.",
        ]
        return result
