from fastapi import FastAPI, Depends
from src.model import RideDurationModel
from src.schemas import PredictRequest, PredictResponse
from configs.config import get_settings
from functools import lru_cache


settings = get_settings()

@lru_cache
def get_model() -> RideDurationModel:
    RideDurationModel(model_path=settings.model_path_resolved)

app = FastAPI(title="Ride Duration service")


@app.get("/health")
def health():
    return {"status": "alive and kicking"}

@app.post("/predict")
def predict(request: PredictRequest, 
            model: RideDurationModel = Depends(get_model)):
    return PredictResponse(duration=model.predict([request.distance_km, request.passengers]))
