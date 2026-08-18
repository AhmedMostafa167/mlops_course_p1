from fastapi import FASTAPI
from src.model import RideDurationModel
from src.schemas import PredictRequest, PredictResponse
from configs.config import get_settings

settings = get_settings()

app = FASTAPI(title="Ride Duration service")

model = RideDurationModel(model_path=settings.MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "alive and kicking"}

@app.post("/predict")
def predict(request: PredictRequest):
    return PredictResponse(duration=model.predict([request.distance_km, request.passengers]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)