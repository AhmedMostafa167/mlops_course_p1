from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    distance_km: float = Field(..., description="distance in km", gt=0)
    passengers: int = Field(..., description="number of passengers", ge=1, le=8)

class PredictResponse(BaseModel):
    duration: float = Field(..., description="duration in minutes")