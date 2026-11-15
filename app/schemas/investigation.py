import math
from typing import Any, Literal
from app.core.characterization import validate_uncertainty_categories
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
class ActiveInvestigationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    material_id: str = Field(min_length=1, max_length=128)
    material_family: str = Field(min_length=1, max_length=128)
    evidence: dict[str, Any]
    uncertainty: dict[str, float] = Field(default_factory=dict)
    information_gain_inputs: dict | None = None
    investigation_goal: Literal["material_state_reconstruction"] = (
        "material_state_reconstruction"
    )
    top_k: int = Field(default=5, ge=1, le=7)
    @field_validator("uncertainty", mode="before")
    @classmethod
    def validate_uncertainty(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Uncertainty must be a dictionary.")
        validate_uncertainty_categories(value)
        for category, score in value.items():
            if isinstance(score, bool):
                raise ValueError(
                    f"Uncertainty for {category} cannot be Boolean."
                )
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Uncertainty for {category} must be numeric."
                )
            if not math.isfinite(numeric_score):
                raise ValueError(
                    f"Uncertainty for {category} must be finite."
                )
            if not 0.0 <= numeric_score <= 1.0:
                raise ValueError(
                    f"Uncertainty for {category} must be between 0 and 1."
                )
        return value
