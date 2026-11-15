from app.schemas.api_responses import PredictionPersistenceResponse
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict
class ResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
class FeatureEvidenceResponse(ResponseSchema):
    status: Literal["PROVIDED", "IMPUTED_BY_MODEL"]
    supplied_value: float | str | None
    imputation_strategy: str | None
    imputation_value: float | str | None
class InputEvidenceResponse(ResponseSchema):
    method: str
    total_features: int
    provided_feature_count: int
    imputed_feature_count: int
    provided_feature_fraction: float
    provided_features: list[str]
    imputed_features: list[str]
    feature_details: dict[str, FeatureEvidenceResponse]
    interpretation: str
class ApplicabilityThresholdsResponse(ResponseSchema):
    q90: float
    q95: float
    calibration_unit: str | None = None
    calibration_grades: int
    q90_rank: int | None = None
    q95_rank: int | None = None
    calibration_records: int | None = None
class RangeViolationResponse(ResponseSchema):
    feature: str
    value: float
    type: str
    q01: float | None = None
    q99: float | None = None
    minimum: float | None = None
    maximum: float | None = None
class ApplicabilityResponse(ResponseSchema):
    status: Literal[
        "IN_DOMAIN",
        "BORDERLINE",
        "OUT_OF_DOMAIN",
        "INSUFFICIENT_EVIDENCE",
    ]
    method: str
    distance: float | None
    thresholds: ApplicabilityThresholdsResponse
    eligible_dimensions: int
    provided_feature_count: int
    missing_features: list[str]
    unseen_categories: dict[str, str]
    range_violations: list[RangeViolationResponse]
    interpretation: str
    confidence_probability: float | None
class PredictionValidationResponse(ResponseSchema):
    method: str
    training_records: int | None = None
    test_records: int
    MAE: float | None = None
    RMSE: float | None = None
    R2: float | None = None
    median_baseline_MAE: float | None = None
class PredictiveUncertaintyResponse(ResponseSchema):
    status: Literal["NOT_AVAILABLE"]
    interval_lower: float | None
    interval_upper: float | None
    confidence_level: float | None
    method: str | None
    calibration_status: Literal["NOT_PROMOTED"]
    reason: str
class TensilePredictionResponse(ResponseSchema):
    prediction: float
    input_evidence: InputEvidenceResponse
    applicability: ApplicabilityResponse
    predictive_uncertainty: PredictiveUncertaintyResponse
    target: str
    target_unit: str
    model: str
    validation: PredictionValidationResponse
    limitations: list[str]
    persistence: PredictionPersistenceResponse | None = None
    explainability: dict | None = None
class YieldPredictionResponse(ResponseSchema):
    prediction: float
    input_evidence: InputEvidenceResponse
    applicability: ApplicabilityResponse
    predictive_uncertainty: PredictiveUncertaintyResponse
    target: str
    target_unit: str
    model: str
    validation: PredictionValidationResponse
    limitations: list[str]
    persistence: PredictionPersistenceResponse | None = None
    explainability: dict | None = None
