"""Pydantic response schemas for AFLOW integration."""
from pydantic import BaseModel, ConfigDict
class AflowSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    dataset: str
    classification: str
    record_id: str
    record_locator: str
    data_origin: str
    citation: str
class AflowStructureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auid: str
    aurl: str
    compound: str
    species: list[str]
    number_of_species: int
    space_group: int | None = None
    pearson_symbol: str | None = None
    band_gap: float | None = None
    formation_enthalpy: float | None = None
    source: AflowSource
class AflowSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    exact_species_count: int | None = None
    page: int
    page_size: int
    require_properties: bool
    aflux_summons: str
class AflowSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    provider: str
    dataset: str
    classification: str
    query: AflowSearchQuery
    record_count: int
    total_matches: int | None = None
    records: list[AflowStructureRecord]
    property_units: dict[str, str]
    scientific_notice: str
