from typing import Any
from pydantic import BaseModel, ConfigDict, StrictStr
class EvidenceCreateInput(BaseModel):
    """Validated API contract for an immutable scientific evidence record."""
    model_config = ConfigDict(extra="forbid")
    material_id: StrictStr
    evidence_category: StrictStr
    measured_quantity: StrictStr
    source_type: StrictStr
    provenance_classification: StrictStr
    investigation_id: StrictStr | None = None
    value: Any = None
    unit: StrictStr | None = None
    uncertainty_value: Any = None
    measurement_method: StrictStr | None = None
    test_standard: StrictStr | None = None
    source_name: StrictStr | None = None
    source_record_id: StrictStr | None = None
    doi_or_url: StrictStr | None = None
    specimen_id: StrictStr | None = None
    observed_at: StrictStr | None = None
    original_filename: StrictStr | None = None
    stored_filename: StrictStr | None = None
    sha256: StrictStr | None = None
    media_type: StrictStr | None = None
