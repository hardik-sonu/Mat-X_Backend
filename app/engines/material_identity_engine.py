from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
ENGINE_VERSION = "1.0.0"
from app.core.paths import STEELBENCH_PATH as DATASET_PATH
ELEMENTS = ["C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al"]
MIN_SHARED_ELEMENTS = 3
@lru_cache(maxsize=1)
def _load_reference_profiles():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Material identity dataset not found: {DATASET_PATH}"
        )
    df = pd.read_csv(DATASET_PATH)
    required = {"grade_id", "steel_family", *ELEMENTS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )
    profiles = []
    for grade_id, group in df.groupby("grade_id"):
        profile = {
            "grade_id": str(grade_id),
            "steel_family": str(group["steel_family"].iloc[0]),
            "record_count": int(len(group)),
            "source": str(group["source"].iloc[0]),
        }
        for element in ELEMENTS:
            profile[element] = (
                float(group[element].median())
                if group[element].notna().any()
                else None
            )
        profiles.append(profile)
    return pd.DataFrame(profiles)
@lru_cache(maxsize=1)
def _calculate_element_scales():
    df = pd.read_csv(DATASET_PATH)
    scales = {}
    for element in ELEMENTS:
        values = df[element].dropna()
        if values.empty:
            scales[element] = None
            continue
        q25 = float(values.quantile(0.25))
        q75 = float(values.quantile(0.75))
        iqr = q75 - q25
        scales[element] = iqr if iqr > 0 else None
    return scales
def identify_material_candidates(
    composition: dict,
    material_family: str | None = None,
    top_k: int = 10,
):
    if not isinstance(composition, dict) or not composition:
        raise ValueError("A non-empty chemical composition is required.")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer.")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")
    unknown = set(composition) - set(ELEMENTS)
    if unknown:
        raise ValueError(
            f"Unsupported chemical elements: {sorted(unknown)}"
        )
    query = {}
    for element, value in composition.items():
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Composition for {element} must be numeric."
            )
        if not np.isfinite(number) or number < 0 or number > 100:
            raise ValueError(
                f"Composition for {element} must be between 0 and 100."
            )
        query[element] = number
    if len(query) < MIN_SHARED_ELEMENTS:
        raise ValueError(
            "At least three measured chemical elements are required."
        )
    profiles = _load_reference_profiles()
    scales = _calculate_element_scales()
    if material_family:
        profiles = profiles[
            profiles["steel_family"] == material_family
        ]
        if profiles.empty:
            raise ValueError(
                f"No reference profiles found for family: {material_family}"
            )
    candidates = []
    for _, profile in profiles.iterrows():
        shared = [
            element
            for element in query
            if pd.notna(profile[element]) and scales[element] is not None
        ]
        if len(shared) < MIN_SHARED_ELEMENTS:
            continue
        normalized_errors = []
        for element in shared:
            difference = abs(
                query[element] - float(profile[element])
            )
            normalized_errors.append(
                difference / scales[element]
            )
        distance = float(np.sqrt(
            np.mean(np.square(normalized_errors))
        ))
        coverage = len(shared) / len(query)
        candidates.append({
            "grade_id": profile["grade_id"],
            "steel_family": profile["steel_family"],
            "reference_record_count": int(profile["record_count"]),
            "source": profile["source"],
            "shared_elements": shared,
            "shared_element_count": len(shared),
            "query_element_count": len(query),
            "element_coverage": round(coverage, 4),
            "normalized_composition_distance": round(distance, 6),
            "reference_composition": {
                element: (
                    float(profile[element])
                    if pd.notna(profile[element])
                    else None
                )
                for element in ELEMENTS
            },
            "match_type": "CHEMISTRY_BASED_CANDIDATE",
        })
    candidates.sort(
        key=lambda item: (
            item["normalized_composition_distance"],
            -item["element_coverage"],
            -item["reference_record_count"],
        )
    )
    selected = candidates[:top_k]
    return {
        "engine": "MAT-X Material Identity Engine",
        "engine_version": ENGINE_VERSION,
        "query_composition": query,
        "material_family_filter": material_family,
        "reference_grade_count": int(len(profiles)),
        "eligible_candidate_count": len(candidates),
        "recommendations": selected,
        "methodology": {
            "reference_profile": "Grade-level median composition",
            "distance": (
                "Root mean square of absolute composition differences "
                "normalized by dataset-wide elemental interquartile ranges"
            ),
            "minimum_shared_elements": MIN_SHARED_ELEMENTS,
            "confidence_calibration": "NOT_CALIBRATED",
            "definitive_identification": False,
        },
        "limitations": [
            "Candidates are limited to the SteelBench reference dataset.",
            "Dataset grade IDs are not verified commercial designations.",
            "Sparse reference groups may not represent full grade chemistry ranges.",
            "Missing elements are excluded from each pairwise comparison.",
            "Normalized distance is not a probability or calibrated confidence.",
            "Composition alone cannot establish material identity definitively.",
            "No experimental confirmation or external standards comparison is performed.",
        ],
        "status": (
            "CANDIDATE_MATCHING_ONLY"
            if selected
            else "INSUFFICIENT_REFERENCE_OVERLAP"
        ),
    }
