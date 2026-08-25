from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app, get_predictor
from prodml.config import get_settings
from prodml.features import prepare_features
from prodml.predict import DurationPredictor
from prodml.train import evaluate_model, fit_model, save_model


@pytest.fixture
def raw_trip_df() -> pd.DataFrame:
    """Small raw dataset covering valid, boundary, and invalid rows."""
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 11:00:00",
                    "2024-01-01 12:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 14:00:00",
                    "2024-01-01 15:00:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:10:00",  # valid
                    "2024-01-01 11:01:00",  # duration lower boundary
                    "2024-01-01 13:00:00",  # duration upper boundary
                    "2024-01-01 14:00:30",  # duration below lower boundary
                    "2024-01-01 15:10:00",  # distance above upper boundary
                    "2024-01-01 16:10:00",  # null pickup location
                ]
            ),
            "PULocationID": [74, 74, 75, 74, 74, None],
            "DOLocationID": [75, 76, 75, 76, 75, 76],
            "trip_distance": [1.5, 0.01, 100.0, 1.0, 100.1, 1.0],
        }
    )


@pytest.fixture
def prepared_trip_df(raw_trip_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_features(raw_trip_df)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    """Engineered records sufficient for a deterministic tiny model."""
    return pd.DataFrame(
        {
            "PU_DO": ["74_75", "74_76", "75_75", "75_76", "74_75", "75_76"],
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "duration": [8.0, 12.0, 16.0, 20.0, 24.0, 28.0],
        }
    )


@pytest.fixture
def trained_components(train_frame):
    model, vectorizer = fit_model(train_frame)
    metrics = evaluate_model(model, vectorizer, train_frame)
    return model, vectorizer, metrics


@pytest.fixture
def artifact_path(tmp_path, trained_components):
    model, vectorizer, metrics = trained_components
    return save_model(
        model,
        vectorizer,
        metrics,
        model_path=tmp_path / "model.pkl",
    )


@pytest.fixture
def predictor(artifact_path) -> DurationPredictor:
    return DurationPredictor.load(artifact_path)


@pytest.fixture
def mock_predictor():
    predictor = Mock()
    predictor.predict_one.return_value = 12.5
    predictor.predict_batch.return_value = [12.5, 18.0]
    return predictor


@pytest.fixture
def api_settings(artifact_path):
    settings = get_settings()
    return SimpleNamespace(
        model_path=artifact_path,
        year=settings.year,
        month=settings.month,
        features=settings.features,
    )


@pytest.fixture
def api_client(mock_predictor, api_settings, monkeypatch):
    """Client for HTTP contract tests without invoking model startup."""
    monkeypatch.setattr("prodml.api.main.get_settings", lambda: api_settings)
    previous_predictor = getattr(app.state, "predictor", None)
    app.state.predictor = mock_predictor
    app.dependency_overrides[get_predictor] = lambda: mock_predictor

    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()
        app.state.predictor = previous_predictor
