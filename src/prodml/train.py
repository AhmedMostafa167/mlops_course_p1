from __future__ import annotations
import os
import argparse
import pickle
import tempfile
from pathlib import Path
from typing import Any, Literal, Sequence
import subprocess
import mlflow
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prodml.config import get_settings
from prodml.data import download_data, load_data, train_validation_split
from prodml.features import get_target, prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging

configure_logging()

from structlog import get_logger

logger = get_logger(__name__)

Metrics = dict[str, float]
ModelType = Literal["lr", "xgboost", "mlp"]


class MLPRegressor(nn.Module):
    """Small dense neural network with an sklearn-like prediction method."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(features, dtype=torch.float32)
        return self(tensor).squeeze(-1).cpu().numpy()


def _validate_model_type(model_type: str) -> ModelType:
    if model_type not in {"lr", "xgboost", "mlp"}:
        raise ValueError(
            f"Unsupported model type {model_type!r}; choose lr, xgboost, or mlp"
        )
    return model_type  # type: ignore[return-value]


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_dim: int = 32,
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
    model_type = _validate_model_type(model_type)
    logger.info("model_fitting_started", model_type=model_type)

    vectorizer = DictVectorizer(sparse=model_type != "mlp", dtype=np.float32)
    X_train = vectorizer.fit_transform(to_feature_dicts(train_df)) 
    y_train = get_target(train_df).to_numpy(dtype=np.float32)

    if model_type == "lr":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(X_train, y_train, verbose=False)
    else:
        model = fit_mlp(
            np.asarray(X_train, dtype=np.float32),
            y_train,
        )

    logger.info("model_fitting_completed", model_type=model_type)
    return model, vectorizer


def evaluate_model(
    model: Any,
    vectorizer: DictVectorizer,
    validation_df: Any,
) -> Metrics:
    """Calculate validation RMSE and MAE in minutes."""
    logger.info("evaluation_started")
    X_valid = vectorizer.transform(to_feature_dicts(validation_df))
    y_valid = get_target(validation_df).to_numpy(dtype=np.float32)
    predictions = np.asarray(model.predict(X_valid)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    r2 = float(r2_score(y_valid, predictions))

    logger.info("evaluation_completed", rmse=rmse, mae=mae, r2=r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _infer_model_type(model: Any) -> ModelType:
    if isinstance(model, LinearRegression):
        return "lr"
    if isinstance(model, XGBRegressor):
        return "xgboost"
    if isinstance(model, MLPRegressor):
        return "mlp"
    raise ValueError(f"Cannot infer model type for {type(model).__name__}")


def save_model(
    model: Any,
    vectorizer: DictVectorizer,
    metrics: Metrics,
    model_path: Path | None = None,
    model_type: ModelType | None = None,
) -> Path:
    """Persist the fitted model, vectorizer, metadata, and metrics."""
    settings = get_settings()
    settings.ensure_project_directories()
    model_type = _validate_model_type(model_type or _infer_model_type(model))

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
        "model_type": model_type,
    }

    logger.info("model_save_started", path=str(output_path), model_type=model_type)
    with output_path.open("wb") as f_out:
        pickle.dump(artifacts, f_out)

    logger.info(
        "model_saved",
        path=str(output_path),
        model_type=model_type,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
    )
    return output_path


def _log_model_to_mlflow(
    model: Any,
    model_type: ModelType,
    input_example: np.ndarray | None = None,
) -> None:
    if model_type == "lr":
        mlflow.sklearn.log_model(model, "model")
    elif model_type == "xgboost":
        mlflow.xgboost.log_model(model, "model")
    else:
        if input_example is None:
            raise ValueError("input_example is required when logging an MLP")
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
        )


def _log_model_params(model: Any, model_type: ModelType) -> None:
    params: dict[str, Any]
    if model_type == "lr":
        params = {
            "fit_intercept": model.fit_intercept,
            "copy_X": model.copy_X,
            "n_jobs": model.n_jobs,
        }
    elif model_type == "xgboost":
        params = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
        }
    else:
        params = {
            "hidden_dim": model.network[0].out_features,
            "epochs": 250,
            "learning_rate": 1e-3,
        }
    mlflow.log_params({"model_type": model_type, **params})

def _log_run_tags(model_type: ModelType) -> None:
    framework = {"lr": "sklearn", "xgboost": "xgboost", "mlp": "pytorch"}[model_type]
    mlflow.set_tags({
        "framework": framework,
        "author": "Ahmed Mostafa",
    })

def _log_feature_importance_plot(
    model: Any,
    vectorizer: DictVectorizer,
    model_type: ModelType,
    n_top: int = 15,
) -> None:
    """Log native feature importance — LR coefficients or XGBoost's built-in importances.
    No-op for MLP, which has no native importance measure."""

    feature_names = vectorizer.get_feature_names_out()

    if model_type == "lr":
        importances = np.abs(model.coef_)
        xlabel = "|coefficient|"
    else:
        importances = model.feature_importances_
        xlabel = "importance (gain)"

    order = np.argsort(importances)[-n_top:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names[order], importances[order])
    ax.set_xlabel(xlabel)
    ax.set_title("Feature importance")
    fig.tight_layout()

    mlflow.log_figure(fig, "plots/feature_importance.png")

def _log_requirements_artifact() -> None:
    """Snapshot resolved dependency versions so this run's environment can be reproduced later,
    independent of what pyproject.toml resolves to by the time someone reruns it."""
    result = subprocess.run(
        ["uv", "export", "--no-hashes"],
        capture_output=True,
        text=True,
        check=True,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        req_path = Path(tmp_dir) / "requirements.txt"
        req_path.write_text(result.stdout)
        mlflow.log_artifact(str(req_path))


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
        _log_requirements_artifact()
        model, vectorizer = fit_model(train_df=train_df, model_type=model_type)
        metrics = evaluate_model(model, vectorizer, validation_df)
        if model_type!="mlp":
            _log_feature_importance_plot(model, vectorizer, model_type)        
        mlflow.log_metrics(metrics)
        _log_model_params(model, model_type)
        _log_run_tags(model_type)
        with tempfile.TemporaryDirectory() as tmp_dir:
            vectorizer_path = Path(tmp_dir) / "vectorizer.pkl"
            with vectorizer_path.open("wb") as f_out:
                pickle.dump(vectorizer, f_out)
            mlflow.log_artifact(str(vectorizer_path), artifact_path="vectorizer")

        input_example = None
        if model_type == "mlp":
            input_example = np.asarray(
                vectorizer.transform(to_feature_dicts(train_df.iloc[:1])),
                dtype=np.float32,
            )
        _log_model_to_mlflow(
            model,
            model_type,
            input_example=input_example,
        )

        model_path = save_model(model, vectorizer, metrics, model_type=model_type)
        logger.info(
            "training_finished",
            model_type=model_type,
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            model_path=str(model_path),
        )
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        mlflow.log_metric("model_size_mb", size_mb)


if __name__ == "__main__":
    main()
