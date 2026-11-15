"""Response contracts for the local SteelBench adapter."""
from typing import Any
from pydantic import BaseModel, ConfigDict
class SteelBenchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    dataset_version: str
    record_id: str
    classification: str
    source_name: str
    mechanical_property_unit_status: str | None = None
class SteelBenchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heat_id: str
    grade_id: str
    source: str
    steel_family: str | None = None
    composition_wt_pct: dict[str, float | None]
    process_history: dict[str, Any]
    mechanical_properties: dict[str, float | None]
    condition_raw: str | None = None
    split: str | None = None
    data_tier: str | None = None
    source_provenance: SteelBenchProvenance
    quality_warnings: list[str]
class SteelBenchSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    provider: str
    dataset: str
    dataset_version: str
    classification: str
    record_count: int
    total_matches: int
    limit: int
    offset: int
    columns: list[str]
    records: list[SteelBenchRecord]
    scientific_notice: str
