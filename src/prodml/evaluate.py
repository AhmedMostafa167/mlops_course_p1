from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from structlog import get_logger

from prodml.features import get_target, to_feature_dicts
from prodml.logging_config import configure_logging

configure_logging()
logger = get_logger(__name__)

Metrics = dict[str, float]


def evaluate_model(
    model: Any,
    vectorizer: DictVectorizer,
    validation_df: Any,
) -> Metrics:
    """Calculate validation RMSE, MAE, and R²."""
    logger.info("evaluation_started")
    X_valid = vectorizer.transform(to_feature_dicts(validation_df))
    y_valid = get_target(validation_df).to_numpy(dtype=np.float32)
    predictions = np.asarray(model.predict(X_valid)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    r2 = float(r2_score(y_valid, predictions))

    logger.info("evaluation_completed", rmse=rmse, mae=mae, r2=r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}