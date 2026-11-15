"""Read-only, provenance-preserving SteelBench v1.0 adapter."""
import csv
import math
from pathlib import Path
from typing import Any
from app.core.paths import STEELBENCH_PATH
DATASET_NAME = "SteelBench"
DATASET_VERSION = "1.0"
REQUIRED_COLUMNS = {
    "heat_id", "grade_id", "source",
    "C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al",
    "austenitize_T", "quench_medium", "temper_T",
    "tensile_strength", "yield_strength", "elongation",
    "reduction_area", "impact_J_avg", "condition",
    "hardness", "split", "data_tier", "steel_family",
}
COMPOSITION_COLUMNS = ("C", "Mn", "Cr", "Mo", "Ni", "Si", "V", "Cu", "Al")
PROCESS_COLUMNS = ("austenitize_T", "temper_T")
PROPERTY_COLUMNS = (
    "tensile_strength", "yield_strength", "elongation",
    "reduction_area", "impact_J_avg", "hardness",
)
class SteelBenchIntegrationError(RuntimeError):
    """Raised when the local SteelBench dataset cannot be safely read."""
def _number(value: Any, field: str, warnings: list[str]):
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        warnings.append(f"{field}: non-numeric source value preserved as unavailable.")
        return None
    if not math.isfinite(result):
        warnings.append(f"{field}: non-finite source value treated as unavailable.")
        return None
    return result
def _normalize_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    return normalized or None
def _load_records(path: Path) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"SteelBench dataset not found: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SteelBenchIntegrationError("SteelBench CSV has no header.")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise SteelBenchIntegrationError("SteelBench CSV contains duplicate column names.")
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise SteelBenchIntegrationError(
                    "SteelBench CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            records = []
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                # DictReader stores surplus columns under the None key.
                # Check this before required-field validation.
                if None in row:
                    raise SteelBenchIntegrationError(
                        f"SteelBench CSV row {line_number} has extra fields."
                    )
                if all(
                    value is None or not str(value).strip()
                    for value in row.values()
                ):
                    continue
                heat_id = str(row.get("heat_id") or "").strip()
                grade_id = str(row.get("grade_id") or "").strip()
                source = str(row.get("source") or "").strip()
                if not heat_id or not grade_id or not source:
                    raise SteelBenchIntegrationError(
                        f"SteelBench CSV row {line_number} lacks heat_id, grade_id, or source."
                    )
                warnings = []
                composition = {
                    key: _number(row.get(key), key, warnings)
                    for key in COMPOSITION_COLUMNS
                }
                process = {
                    "austenitize_temperature_C": _number(
                        row.get("austenitize_T"), "austenitize_T", warnings
                    ),
                    "quench_medium": (row.get("quench_medium") or "").strip() or None,
                    "temper_temperature_C": _number(
                        row.get("temper_T"), "temper_T", warnings
                    ),
                }
                properties = {
                    "tensile_strength_source_value": _number(
                        row.get("tensile_strength"), "tensile_strength", warnings
                    ),
                    "yield_strength_source_value": _number(
                        row.get("yield_strength"), "yield_strength", warnings
                    ),
                    "elongation_percent": _number(
                        row.get("elongation"), "elongation", warnings
                    ),
                    "reduction_area_percent": _number(
                        row.get("reduction_area"), "reduction_area", warnings
                    ),
                    "impact_energy_J": _number(
                        row.get("impact_J_avg"), "impact_J_avg", warnings
                    ),
                    # The source does not establish a universal hardness scale.
                    "hardness_source_value": _number(
                        row.get("hardness"), "hardness", warnings
                    ),
                }
                raw_condition = (row.get("condition") or "").strip() or None
                if raw_condition and raw_condition.casefold() in {
                    "#name?", "#n/a", "n/a", "na", "null"
                }:
                    warnings.append(
                        "condition contains a spreadsheet/error-like source value; "
                        "raw value retained without interpretation."
                    )
                records.append({
                    "heat_id": heat_id,
                    "grade_id": grade_id,
                    "source": source,
                    "steel_family": (row.get("steel_family") or "").strip() or None,
                    "composition_wt_pct": composition,
                    "process_history": process,
                    "mechanical_properties": properties,
                    "condition_raw": raw_condition,
                    "split": (row.get("split") or "").strip() or None,
                    "data_tier": (row.get("data_tier") or "").strip() or None,
                    "source_provenance": {
                        "dataset": DATASET_NAME,
                        "dataset_version": DATASET_VERSION,
                        "record_id": heat_id,
                        "classification": "SOURCE_DATA_AS_PUBLISHED",
                        "source_name": source,
"mechanical_property_unit_status": "UNVERIFIED",
                    },
                    "quality_warnings": warnings,
                })
            return records, list(reader.fieldnames)
    except UnicodeDecodeError as exc:
        raise SteelBenchIntegrationError(
            "SteelBench CSV encoding is not supported; expected UTF-8."
        ) from exc
    except csv.Error as exc:
        raise SteelBenchIntegrationError(
            f"SteelBench CSV parsing failed: {exc}"
        ) from exc
def search_steelbench_records(
    *,
    heat_id: str | None = None,
    grade_id: str | None = None,
    steel_family: str | None = None,
    source: str | None = None,
    data_tier: str | None = None,
    query: str | None = None,
    limit: int = 25,
    offset: int = 0,
) -> dict:
    """Search and paginate the local SteelBench v1.0 source dataset."""
    if isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    if isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be nonnegative.")
    records, columns = _load_records(Path(STEELBENCH_PATH))
    filters = {
        "heat_id": _normalize_filter(heat_id),
        "grade_id": _normalize_filter(grade_id),
        "steel_family": _normalize_filter(steel_family),
        "source": _normalize_filter(source),
        "data_tier": _normalize_filter(data_tier),
    }
    text_query = _normalize_filter(query)
    filtered = []
    for record in records:
        if any(
            expected is not None
            and _normalize_filter(str(record.get(field) or "")) != expected
            for field, expected in filters.items()
        ):
            continue
        if text_query:
            searchable = " ".join(
                str(record.get(field) or "")
                for field in ("heat_id", "grade_id", "source", "steel_family", "condition_raw")
            ).casefold()
            if text_query not in searchable:
                continue
        filtered.append(record)
    total = len(filtered)
    page = filtered[offset:offset + limit]
    return {
        "status": "SUCCESS",
        "provider": "Local MAT-X dataset",
        "dataset": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "classification": "SOURCE_DATA_AS_PUBLISHED",
        "record_count": len(page),
        "total_matches": total,
        "limit": limit,
        "offset": offset,
        "columns": columns,
        "records": page,
        "scientific_notice": (
            "Records are served from the local SteelBench v1.0 CSV without "
            "modifying the source file. Values retain source attribution. "
            "Blank or non-numeric fields are represented as unavailable; "
            "raw condition text is preserved and not scientifically interpreted. "
            "Hardness is reported as a source value because a universal scale "
            "is not established by this adapter. Dataset inclusion does not "
            "independently validate the original measurement."
        ),
    }
