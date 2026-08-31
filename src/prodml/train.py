from __future__ import annotations

import argparse
import os
from time import perf_counter
from typing import Any, Sequence

import mlflow
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from structlog import get_logger
from torch import nn
from xgboost import XGBRegressor

from prodml.config import get_settings
from prodml.data import download_data, load_data, train_validation_split
from prodml.evaluate import evaluate_model
from prodml.features import get_target, prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging
from prodml.model_types import MLPRegressor, ModelType, validate_model_type
from prodml.persistence import save_model
from prodml.tracking import ExperimentTracker

configure_logging()
logger = get_logger(__name__)


def fit_lr(X_train: Any, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train: Any, y_train: np.ndarray, **params: Any) -> XGBRegressor:
    """Fit an XGBRegressor."""
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        **params,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_dim: int = 16,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> MLPRegressor:
    """Fit a small deterministic MLP for tabular regression."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(random_state)
    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim)
    features = torch.as_tensor(X_train, dtype=torch.float32)
    targets = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def fit_model(
    train_df: Any,
    model_type: ModelType = "lr",
) -> tuple[Any, DictVectorizer]:
    """Fit a selected estimator and the feature vectorizer."""
    model_type = validate_model_type(model_type)
    logger.info("model_fitting_started", model_type=model_type)

    vectorizer = DictVectorizer(sparse=model_type != "mlp", dtype=np.float32)
    X_train = vectorizer.fit_transform(to_feature_dicts(train_df))
    y_train = get_target(train_df).to_numpy(dtype=np.float32)

    if model_type == "lr":
        model = fit_lr(X_train, y_train)
    elif model_type == "xgboost":
        model = fit_xgboost(
            X_train,
            y_train,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    else:
        model = fit_mlp(np.asarray(X_train, dtype=np.float32), y_train)

    logger.info("model_fitting_completed", model_type=model_type)
    return model, vectorizer


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a taxi duration model")
    parser.add_argument(
        "--model",
        dest="model_type",
        choices=("lr", "xgboost", "mlp"),
        default="lr",
        help="Estimator to train (default: lr)",
    )
    args = parser.parse_args(argv)
    model_type: ModelType = args.model_type
    settings = get_settings()
    settings.export_s3_env()

    logger.info(
        "mlflow_tracking_setup_started",
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    mlflow.set_tracking_uri(uri=settings.MLFLOW_TRACKING_URI)
    logger.info(
        "mlflow_experiment_setup_started",
        experiment_name=settings.EXPERIMENT_NAME,
    )
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    logger.info("mlflow_experiment_setup_completed")

    # Autologging covers hyperparameters + the model artifact for lr/xgboost automatically,
    # plus xgboost gets its own feature-importance plot for free. MLP is a raw torch training
    # loop (not Lightning), which neither integration hooks into, so ExperimentTracker logs
    # its params/model manually instead — see tracking.py.
    mlflow.sklearn.autolog(log_models=True, log_datasets=False)
    mlflow.xgboost.autolog(log_models=True, log_datasets=False)

    download_data()
    data = load_data()
    logger.info("feature_preparation_started", rows=len(data))
    prepared = prepare_features(data)
    train_df, validation_df = train_validation_split(prepared)
    logger.info(
        "feature_preparation_completed",
        prepared_rows=len(prepared),
        train_rows=len(train_df),
        validation_rows=len(validation_df),
    )

    run_name = {
        "lr": "lr_with_dictvectorizer",
        "xgboost": "xgboost_with_dictvectorizer",
        "mlp": "mlp_with_dictvectorizer",
    }[model_type]
    with mlflow.start_run(run_name=run_name):
        start = perf_counter()
        model, vectorizer = fit_model(train_df=train_df, model_type=model_type)
        mlflow.log_metric("train_time", perf_counter() - start)

        metrics = evaluate_model(model, vectorizer, validation_df)

        input_example = None
        if model_type == "mlp":
            input_example = np.asarray(
                vectorizer.transform(to_feature_dicts(train_df.iloc[:1])),
                dtype=np.float32,
            )

        tracker = ExperimentTracker(model, vectorizer, model_type)
        tracker.log_all(metrics, validation_df, input_example=input_example)

        model_path = save_model(model, vectorizer, metrics, model_type=model_type)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        mlflow.log_metric("model_size_mb", size_mb)

        logger.info(
            "training_finished",
            model_type=model_type,
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            model_path=str(model_path),
        )


if __name__ == "__main__":
    main()