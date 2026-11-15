"""Response contracts for Zenodo public-record search."""
from pydantic import BaseModel, ConfigDict
class ZenodoCreator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    affiliation: str | None = None
    orcid: str | None = None
class ZenodoFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    download_url: str | None = None
class ZenodoProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str
    classification: str
    record_id: str | None = None
    doi: str | None = None
class ZenodoRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str
    title: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    description: str | None = None
    creators: list[ZenodoCreator]
    keywords: list[str]
    resource_type: str | None = None
    license: str | None = None
    record_url: str | None = None
    api_url: str | None = None
    files: list[ZenodoFile]
    source_provenance: ZenodoProvenance
class ZenodoSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    provider: str
    query: str
    page: int
    page_size: int
    record_count: int
    total_matches: int
    records: list[ZenodoRecord]
    source_provenance: dict
    scientific_notice: str
