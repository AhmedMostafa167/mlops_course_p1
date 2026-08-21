from functools import wraps
from pathlib import Path
from structlog import get_logger
import pickle
from time import perf_counter
from typing import Any, Callable, TypeVar, Self

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import get_settings
from prodml.features import to_feature_dicts
from prodml.logging_config import configure_logging

R = TypeVar("R")
logger = get_logger(__name__)


def timed(func: Callable[..., R]) -> Callable[..., R]:
    """Print how long the decorated function takes to run."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info("%s took %.2f ms", func.__name__, elapsed_ms)
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
        with path.open("rb") as f_in:
            artifacts: dict[str, Any] = pickle.load(f_in)

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


# Backward-compatible alias for code written before the interface was renamed.
BaselinePredictor = DurationPredictor
