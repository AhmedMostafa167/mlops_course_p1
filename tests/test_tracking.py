import subprocess
from unittest.mock import MagicMock

import mlflow.pytorch  # noqa: F401  (forces the submodule to load before we patch it)
import numpy as np
import pytest

from prodml.model_types import MLPRegressor
from prodml.tracking import ExperimentTracker


@pytest.fixture
def mock_mlflow(monkeypatch):
    mocks = {
        "set_tags": MagicMock(),
        "log_params": MagicMock(),
        "log_metrics": MagicMock(),
        "log_figure": MagicMock(),
        "log_artifact": MagicMock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(f"prodml.tracking.mlflow.{name}", mock)
    monkeypatch.setattr("prodml.tracking.mlflow.pytorch.log_model", MagicMock())
    monkeypatch.setattr(
        "prodml.tracking.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="pandas==2.0\n"),
    )
    return mocks


def test_lr_tracker_logs_everything_except_params_and_model(
    mock_mlflow, predictor, train_frame
):
    tracker = ExperimentTracker(
        model=predictor.model, vectorizer=predictor.vectorizer, model_type="lr"
    )

    tracker.log_all(metrics=predictor.metrics, validation_df=train_frame)

    mock_mlflow["set_tags"].assert_called_once()
    mock_mlflow["log_params"].assert_not_called()  # autolog already captured these
    mock_mlflow["log_metrics"].assert_called_once_with(predictor.metrics)
    assert mock_mlflow["log_figure"].call_count == 2  # residual + feature-importance
    assert mock_mlflow["log_artifact"].call_count == 2  # requirements + vectorizer


def test_mlp_tracker_logs_params(mock_mlflow):
    model = MLPRegressor(input_dim=2, hidden_dim=4)
    tracker = ExperimentTracker(model=model, vectorizer=None, model_type="mlp")

    tracker.log_params()

    mock_mlflow["log_params"].assert_called_once()


def test_mlp_log_model_requires_input_example(mock_mlflow):
    model = MLPRegressor(input_dim=2, hidden_dim=4)
    tracker = ExperimentTracker(model=model, vectorizer=None, model_type="mlp")

    with pytest.raises(ValueError):
        tracker.log_model(input_example=None)


def test_mlp_log_model_logs_when_input_example_given(mock_mlflow):
    model = MLPRegressor(input_dim=2, hidden_dim=4)
    tracker = ExperimentTracker(model=model, vectorizer=None, model_type="mlp")

    tracker.log_model(input_example=np.zeros((1, 2), dtype=np.float32))
