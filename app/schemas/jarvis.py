from typing import Any
from pydantic import BaseModel, ConfigDict
class JarvisSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    dataset: str
    classification: str
    record_id: str
    endpoint: str
    data_origin: str
class JarvisStructureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    jarvis_id: str
    formula: str | None = None
    elements: list[str] | None = None
    density: float | None = None
    space_group: str | None = None
    formation_energy: float | None = None
    band_gap: float | None = None
    reference: Any = None
    source: JarvisSource
class JarvisSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elements: list[str]
    page_limit: int
    filter: str
class JarvisSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    provider: str
    dataset: str
    classification: str
    query: JarvisSearchQuery
    record_count: int
    more_data_available: bool | None = None
    records: list[JarvisStructureRecord]
    scientific_notice: str
