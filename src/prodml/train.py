from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import get_settings
from prodml.features import get_target, to_feature_dicts, prepare_features
from prodml.logging_config import configure_logging
from prodml.data import download_data, load_data, train_validation_split

configure_logging()

from structlog import get_logger

logger = get_logger(__name__)
Metrics = dict[str, float]


def fit_model(train_df: Any) -> tuple[LinearRegression, DictVectorizer]:
    """Fit the baseline vectorizer and linear regression model."""
    logger.info("model_fitting_started")
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(to_feature_dicts(train_df))
    y_train = get_target(train_df).to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("model_fitting_completed")
    return model, vectorizer


def evaluate_model(
    model: LinearRegression,
    vectorizer: DictVectorizer,
    validation_df: Any) -> Metrics:
    """Calculate validation RMSE and MAE in minutes."""
    
    logger.info("evaluation_started")
    X_valid = vectorizer.transform(to_feature_dicts(validation_df))
    y_valid = get_target(validation_df).to_numpy()
    predictions = model.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    logger.info("evaluation_completed", rmse=rmse, mae=mae)
    return {
        "rmse": rmse,
        "mae": mae,
    }


def save_model(
    model: LinearRegression,
    vectorizer: DictVectorizer,
    metrics: Metrics,
    model_path: Path | None = None,
) -> Path:
    """Persist the fitted model, vectorizer, metadata, and metrics."""
    settings = get_settings()
    settings.ensure_project_directories()

    output_path = model_path or settings.model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    logger.info("model_save_started", path=str(output_path))

    with output_path.open("wb") as f_out:
        pickle.dump(artifacts, f_out)

    logger.info(
        "model_saved",
        path=str(output_path),
        rmse=metrics["rmse"],
        mae=metrics["mae"],
    )
    return output_path


def main() -> None:
    download_data()
    data = load_data()
    prepared = prepare_features(data)
    train_df, validation_df = train_validation_split(prepared)
    model, vectorizer = fit_model(train_df=train_df)
    metrics = evaluate_model(model, vectorizer, validation_df)
    model_path = save_model(model, vectorizer, metrics)
    logger.info(
        "training_finished",
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        model_path=str(model_path),
    )


if __name__ == "__main__":
    main()