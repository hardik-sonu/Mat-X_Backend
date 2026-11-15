from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
class MaterialIdentityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    composition: dict[str, Optional[float]] = Field(
        min_length=1,
        description="Measured elemental composition in weight percent."
    )
    material_family: Optional[str] = Field(default=None, min_length=1, max_length=128)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("composition", mode="before")
    @classmethod
    def validate_composition(cls, value):
        import math
        if not isinstance(value, dict) or not value:
            raise ValueError("A non-empty chemical composition is required.")
        normalized = {}
        for element, item in value.items():
            if not isinstance(element, str) or not element.strip():
                raise ValueError(
                    "Composition element symbols must be non-empty strings."
                )
            if item is None:
                normalized[element] = None
                continue
            if isinstance(item, bool):
                raise ValueError(
                    f"Composition for {element} cannot be Boolean."
                )
            try:
                number = float(item)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"Composition for {element} must be a finite number."
                )
            if not math.isfinite(number) or not 0.0 <= number <= 100.0:
                raise ValueError(
                    f"Composition for {element} must be between 0 and 100 wt.%."
                )
            normalized[element] = number
        return normalized
