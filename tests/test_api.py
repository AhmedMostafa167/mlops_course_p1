import pytest
from unittest.mock import MagicMock
from src.api import app, get_model
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive and kicking"}
    
    
def test_predict_valid():
    mock_model = MagicMock()
    mock_model.predict.return_value = 10
    app.dependency_overrides[get_model] = lambda: mock_model
    response = client.post("/predict", json={"distance_km": 5.0, "passengers": 2})
    
    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"duration": 10}
    
def test_predict_calls_model_correctly():
    mock_model = MagicMock()
    mock_model.predict.return_value = 10
    app.dependency_overrides[get_model] = lambda: mock_model
    response = client.post("/predict", json={"distance_km": 5.0, "passengers": 2})
    app.dependency_overrides.clear()    
    mock_model.predict.assert_called_once_with([5.0, 2])


    
def test_predict_missing_field():
    response = client.post("/predict", json={"distance_km":5.0})
    assert response.status_code == 422
    
def test_predict_empty_body():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
def test_predict_wrong_distance_type():
    response = client.post("/predict", json={"distance_km": "five", "passengers": 2})
    assert response.status_code == 422
    
def test_predict_wrong_passengers_type():
    response = client.post("/predict", json={"distance_km": 5.0, "passengers": 2.5})
    assert response.status_code == 422
    
def test_predict_negative_distance():
    response = client.post("/predict", json={"distance_km": -5.0, "passengers": 2})
    assert response.status_code == 422
    
def test_predict_zero_passengers():
    response = client.post("/predict", json={"distance_km": 5.0, "passengers": 0})
    assert response.status_code == 422
    
def test_predict_larger_than_8_passengrs():
    response = client.post("/predict", json={"distance_km": 5.0, "passengers": 9})
    assert response.status_code == 422    
    
