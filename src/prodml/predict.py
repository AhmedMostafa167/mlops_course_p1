import pickle
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from structlog import get_logger

from prodml.config import get_settings
from prodml.features import to_feature_dicts
from prodml.persistence import load_model

logger = get_logger(__name__)


def timed[R](func: Callable[..., R]) -> Callable[..., R]:
    """Print how long the decorated function takes to run."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(
            "prediction_completed",
            function=func.__name__,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return result

    return wrapper


class DurationPredictor:
    """Predict taxi-trip duration from engineered feature dictionaries."""

    def __init__(
        self,
        model: LinearRegression,
        vectorizer: DictVectorizer,
        features: list[str],
        target: str,
        metrics: dict[str, float],
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.features = features
        self.target = target
        self.metrics = metrics

    @classmethod
    def load(cls, model_path: Path | None = None) -> Self:
        """Load a persisted duration model and return a ready predictor."""
        path = model_path or get_settings().model_path
        artifacts = load_model(path)

        return cls(
            model=artifacts["model"],
            vectorizer=artifacts["dv"],
            features=artifacts["features"],
            target=artifacts["target"],
            metrics=artifacts["metrics"],
        )

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Predict one trip duration in minutes."""
        frame = pd.DataFrame([features])
        transformed = self.vectorizer.transform(to_feature_dicts(frame))
        return float(self.model.predict(transformed)[0])

    def predict_batch(self, features: list[dict[str, Any]]) -> list[float]:
        """Predict durations for a batch of engineered feature dictionaries."""
        frame = pd.DataFrame(features)
        transformed = self.vectorizer.transform(to_feature_dicts(frame))
        return [float(value) for value in self.model.predict(transformed)]

    @classmethod
    def load_from_registry(cls, model_uri: str | None = None) -> Self:
        """Load the current Production model + its vectorizer from MLflow directly."""
        settings = get_settings()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        uri = model_uri or settings.PRODUCTION_MODEL_URI

        pyfunc_model = mlflow.pyfunc.load_model(uri)
        run_id = pyfunc_model.metadata.run_id
        run = mlflow.get_run(run_id)

        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="vectorizer/vectorizer.pkl"
        )
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        return cls(
            model=pyfunc_model,
            vectorizer=vectorizer,
            features=settings.features,
            target=settings.target,
            metrics=dict(run.data.metrics),
        )


# Backward-compatible alias for code written before the interface was renamed.
BaselinePredictor = DurationPredictor
