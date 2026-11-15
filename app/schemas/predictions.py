from pydantic import BaseModel, ConfigDict
from typing import Optional
class TensilePredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    C: Optional[float] = None
    Mn: Optional[float] = None
    Cr: Optional[float] = None
    Mo: Optional[float] = None
    Ni: Optional[float] = None
    Si: Optional[float] = None
    V: Optional[float] = None
    Cu: Optional[float] = None
    Al: Optional[float] = None
    austenitize_T: Optional[float] = None
    temper_T: Optional[float] = None
    quench_medium: Optional[str] = None
    steel_family: Optional[str] = None
    material_id: Optional[str] = None
class YieldPredictionInput(TensilePredictionInput):
    pass

class InclusionPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]
    material_id: Optional[str] = None
