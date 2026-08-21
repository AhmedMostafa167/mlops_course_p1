from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import get_settings
from prodml.features import get_target, to_feature_dicts


Metrics = dict[str, float]


def fit_model(train_df: Any) -> tuple[LinearRegression, DictVectorizer]:
    """Fit the baseline vectorizer and linear regression model."""
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(to_feature_dicts(train_df))
    y_train = get_target(train_df).to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, vectorizer


def evaluate_model(
    model: LinearRegression,
    vectorizer: DictVectorizer,
    validation_df: Any) -> Metrics:
    """Calculate validation RMSE and MAE in minutes."""
    X_valid = vectorizer.transform(to_feature_dicts(validation_df))
    y_valid = get_target(validation_df).to_numpy()
    predictions = model.predict(X_valid)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
        "mae": float(mean_absolute_error(y_valid, predictions)),
    }


def save_model(
    model: LinearRegression,
    vectorizer: DictVectorizer,
    metrics: Metrics,
    model_path: Path | None = None) -> Path:
    """Persist the fitted model, vectorizer, metadata, and metrics."""
    settings = get_settings()
    settings.ensure_project_directories()
    output_path = model_path or settings.model_path
    artifacts: dict[str, Any] = {
        "model": model,
        "dv": vectorizer,
        "features": settings.features,
        "target": settings.target,
        "data_url": settings.data_url,
        "year": settings.year,
        "month": settings.month,
        "metrics": metrics,
    }
    with output_path.open("wb") as f_out:
        pickle.dump(artifacts, f_out)
    return output_path
