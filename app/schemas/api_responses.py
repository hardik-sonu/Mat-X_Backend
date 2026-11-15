from typing import Any
from pydantic import BaseModel, ConfigDict
class MATXResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
class DamageAssessmentResponse(MATXResponse):
    engine: str
    engine_version: str
    material_id: str
    status: str
    evidence_status: str
    observation_count: int
    excluded_record_count: int
    observations: list[dict[str, Any]]
    diagnosis_performed: bool
    severity_assessed: bool
    remaining_life_predicted: bool
    limitations: list[str]
class MaterialStateResponse(MATXResponse):
    engine: str
    engine_version: str
    status: str
    material: dict[str, Any]
    material_identity_assessment: dict[str, Any]
    evidence: dict[str, Any]
    damage_condition_assessment: dict[str, Any]
    evidence_registry: dict[str, Any]
    evidence_summary: dict[str, Any]
    active_investigation: dict[str, Any]
    scientific_scope: dict[str, Any]
    limitations: list[str]
class EngineeringReportResponse(MATXResponse):
    report: dict[str, Any]
    material: dict[str, Any]
    executive_summary: dict[str, Any]
    source_evidence: dict[str, Any]
    validated_evidence_registry: dict[str, Any]
    material_identity_assessment: dict[str, Any] | None
    damage_condition_assessment: dict[str, Any] | None
    user_reported_damage_observation: dict[str, Any]
    active_investigation: dict[str, Any]
    prediction_results: dict[str, Any]
    scientific_scope: dict[str, Any]
    limitations: list[str]
    engineering_disclaimer: str
class HealthResponse(MATXResponse):
    status: str
    service: str
    models_available: int
    models_registered: int
    models: dict[str, Any]
class ModelsResponse(MATXResponse):
    registered_models: dict[str, Any]
class PredictionPersistenceResponse(MATXResponse):
    status: str
    prediction_id: str | None = None
    material_id: str | None = None

class InclusionPredictionResponse(MATXResponse):
    predicted_class: int | str
    classification: str
    probabilities: dict[str, Any]
    model: str | dict[str, Any]
    scope: Any
    validation: dict[str, Any]
    limitations: list[str]
    persistence: PredictionPersistenceResponse
class MicrostructureMapsResponse(MATXResponse):
    dataset_root: str
    map_count: int
    maps: list[dict[str, Any]]
class FatigueResponse(MATXResponse):
    engine: str
    engine_version: str
    source_file: str
    cycle_feature_records: int
    independent_specimens: int
    geometry_groups: list[str]
    specimen_results: list[dict[str, Any]]
    scientific_scope: list[str]
    status: str
class ActiveInvestigationResponse(MATXResponse):
    engine: str
    engine_version: str
    material_id: str
    material_family: str
    investigation_goal: str
    evidence_summary: dict[str, Any]
    recommendations: list[dict[str, Any]]
    recommendation_count: int
    methodology: dict[str, Any]
    limitations: list[str]
    status: str
class MaterialIdentityResponse(MATXResponse):
    engine: str
    engine_version: str
    query_composition: dict[str, float]
    material_family_filter: str | None
    reference_grade_count: int
    eligible_candidate_count: int
    recommendations: list[dict[str, Any]]
    methodology: dict[str, Any]
    limitations: list[str]
    status: str
class ProcessHistoryResponse(MATXResponse):
    engine: str
    version: str
    status: str
    dataset: str
    total_records: int
    unique_heats: int
    process_fields: list[str]
    status_distribution: dict[str, int]
    field_availability: dict[str, Any]
    methodology: dict[str, Any]
    limitations: list[str]
    outputs: dict[str, Any]
class ProcessHistoryAnalysisResponse(MATXResponse):
    top_observed_process_combinations: list[dict[str, Any]]
    complete_documented_routes: list[dict[str, Any]]
    steel_family_summary: list[dict[str, Any]]
    grade_summary: list[dict[str, Any]]
    returned_record_limits: dict[str, int]
class ProcessLookupResponse(MATXResponse):
    engine: str
    status: str
    lookup_type: str
    record_count: int
    records: list[dict[str, Any]]
    limitations: list[str]
class EbsdProvenanceResponse(MATXResponse):
    engine: str
    version: str
    files_discovered: int
    files_extracted: int
    failed_files: int
    records: list[dict[str, Any]]
    errors: list[dict[str, Any]]
class UnifiedMicrostructureResponse(MATXResponse):
    engine: str
    engine_version: str
    status: str
    map_id: str
    microstructure_features: dict[str, Any]
    characterization_provenance: dict[str, Any]
    scientific_scope: dict[str, bool]
class EvidenceRecordResponse(MATXResponse):
    evidence_id: str
    material_id: str
    evidence_category: str
    measured_quantity: str
    value: Any = None
    unit: str | None = None
    uncertainty_value: float | None = None
    measurement_method: str | None = None
    test_standard: str | None = None
    source_type: str
    source_name: str | None = None
    source_record_id: str | None = None
    doi_or_url: str | None = None
    specimen_id: str | None = None
    investigation_id: str | None = None
    observed_at: str | None = None
    provenance_classification: str
    validation_status: str
    original_filename: str | None = None
    stored_filename: str | None = None
    sha256: str | None = None
    media_type: str | None = None
    created_at: str
