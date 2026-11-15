from pydantic import BaseModel, ConfigDict, Field, StrictStr
class DamageAssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    material_id: StrictStr = Field(min_length=1, max_length=128)
