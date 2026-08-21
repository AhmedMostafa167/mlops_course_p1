from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input for one taxi-duration prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pu_do": "74_75",
                "trip_distance": 1.5,
            }
        }
    )

    pu_do: str = Field(..., min_length=1, description="Pickup/dropoff pair")
    trip_distance: float = Field(
        ...,
        gt=0,
        lt=200,
        description="Trip distance in miles",
    )


class PredictionResponse(BaseModel):
    """Output for one taxi-duration prediction."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class PredictionBatchRequest(BaseModel):
    """Input for a batch of taxi-duration predictions."""

    items: list[PredictionRequest] = Field(..., min_length=1, max_length=10_000)


class PredictionBatchResponse(BaseModel):
    """Output for a batch of taxi-duration predictions."""

    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float


class MetadataResponse(BaseModel):
    """Metadata describing the loaded model artifact."""

    model_version: str
    training_date: str
    features: list[str]
    framework: str
    artifact_hash: str
