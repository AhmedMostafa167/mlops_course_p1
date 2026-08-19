import pytest
from src.model import RideDurationModel, modelBase, RideDurationNet
from configs.config import get_settings

settings = get_settings()

@pytest.fixture(scope="module")
def model():
    return RideDurationModel(settings.model_path_resolved)

def test_model():
    model = RideDurationModel(settings.model_path_resolved)
    assert isinstance(model, modelBase)
    assert isinstance(model._model, RideDurationNet)
    
def test_model_loads(model):
    assert model is not None
    
def test_model_returns_float(model):
    predict = model.predict([5.0, 2])
    assert isinstance(predict, float)
    
def test_predict_reasonable_range(model):
    predict = model.predict([20.0, 2])
    assert predict > 0
    
def test_predict_deterministic(model):
    p1 = model.predict([20.0, 2])
    p2 = model.predict([20.0, 2])
    assert p1 == p2
    
    
def test_missing_inputs(model):
    with pytest.raises(Exception):
        model.predict([])
        
def test_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        RideDurationModel(model_path="missing_file.pt")