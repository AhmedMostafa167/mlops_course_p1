from fastapi import FastAPI
from src.model import RideDurationModel
from src.schemas import PredictRequest, PredictResponse
from configs.config import get_settings

settings = get_settings()

app = FastAPI(title="Ride Duration service")

model = RideDurationModel(model_path=settings.model_path_resolved)

@app.get("/health")
def health():
    return {"status": "alive and kicking"}

@app.post("/predict")
def predict(request: PredictRequest):
    return PredictResponse(duration=model.predict([request.distance_km, request.passengers]))
