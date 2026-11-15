import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.core.characterization import validate_uncertainty_categories
def _validate_numeric_dict(
    value: Any,
    field_name: str,
    allow_none: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
):
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        if item is None and allow_none:
            normalized[key] = None
            continue
        if isinstance(item, bool):
            raise ValueError(f"{field_name}.{key} cannot be Boolean.")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{field_name}.{key} must be a finite number.")
        if not math.isfinite(number):
            raise ValueError(f"{field_name}.{key} must be finite.")
        if minimum is not None and number < minimum:
            raise ValueError(
                f"{field_name}.{key} must be at least {minimum}."
            )
        if maximum is not None and number > maximum:
            raise ValueError(
                f"{field_name}.{key} must be at most {maximum}."
            )
        normalized[key] = number
    return normalized
class MaterialStateReconstructionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    material_id: str = Field(min_length=1, max_length=128)
    material_family: str = Field(min_length=1, max_length=128)
    composition: dict[str, float | None] | None = None
    measured_properties: dict[str, float] = Field(default_factory=dict)
    heat_id: str | None = Field(default=None, min_length=1, max_length=128)
    grade_id: str | None = Field(default=None, min_length=1, max_length=128)
    microstructure_map_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=16,
    )
    damage_condition: str | None = Field(default=None, max_length=2000)
    uncertainty: dict[str, float] = Field(default_factory=dict)
    identity_top_k: int = Field(default=5, ge=1, le=50)
    investigation_top_k: int = Field(default=5, ge=1, le=7)
    @field_validator("composition", mode="before")
    @classmethod
    def validate_composition(cls, value):
        if value is None:
            return None
        return _validate_numeric_dict(
            value,
            "composition",
            allow_none=True,
            minimum=0.0,
            maximum=100.0,
        )
    @field_validator("measured_properties", mode="before")
    @classmethod
    def validate_measured_properties(cls, value):
        if value is None:
            return {}
        return _validate_numeric_dict(
            value,
            "measured_properties",
        )
    @field_validator("uncertainty", mode="before")
    @classmethod
    def validate_uncertainty(cls, value):
        if value is None:
            return {}
        normalized = _validate_numeric_dict(value, "uncertainty")
        validate_uncertainty_categories(normalized)
        for category, score in normalized.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Uncertainty for {category} must be between 0 and 1."
                )
        return normalized
