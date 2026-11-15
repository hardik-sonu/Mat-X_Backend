"""Response contracts for local dataset discovery."""
from typing import Any
from pydantic import BaseModel, ConfigDict
class LocalDatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    format: str
    size_bytes: int
    modified_at: str
class LocalDatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    dataset_directory: str
    dataset_count: int
    datasets: list[LocalDatasetFile]
class LocalDatasetSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    dataset: str
    format: str
    columns: list[str]
    total_source_records: int
    total_matches: int
    record_count: int
    limit: int
    offset: int
    records: list[dict[str, Any]]
    source_provenance: dict[str, Any]
    scientific_notice: str
