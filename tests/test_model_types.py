import numpy as np
import pytest
import torch
from xgboost import XGBRegressor

from prodml.model_types import MLPRegressor, infer_model_type, validate_model_type


def test_validate_model_type_rejects_unknown_type():
    with pytest.raises(ValueError):
        validate_model_type("not-a-real-model")


def test_infer_model_type_detects_xgboost():
    assert infer_model_type(XGBRegressor()) == "xgboost"


def test_infer_model_type_detects_mlp():
    assert infer_model_type(MLPRegressor(input_dim=3)) == "mlp"


def test_infer_model_type_rejects_unknown_model():
    with pytest.raises(ValueError):
        infer_model_type(object())


def test_mlp_predict_returns_expected_shape():
    torch.manual_seed(0)
    model = MLPRegressor(input_dim=4, hidden_dim=8)
    features = np.random.rand(5, 4).astype(np.float32)

    predictions = model.predict(features)

    assert predictions.shape == (5,)
