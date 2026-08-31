from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from structlog import get_logger

from prodml.config import get_settings
from prodml.evaluate import Metrics
from prodml.logging_config import configure_logging
from prodml.model_types import ModelType, infer_model_type, validate_model_type

configure_logging()
logger = get_logger(__name__)


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
    model_type = validate_model_type(model_type or infer_model_type(model))

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


def load_model(model_path: Path | None = None) -> dict[str, Any]:
    """Load a persisted model bundle (model, vectorizer, metadata, metrics)."""
    path = model_path or get_settings().model_path
    logger.info("model_load_started", path=str(path))
    with path.open("rb") as f_in:
        artifacts: dict[str, Any] = pickle.load(f_in)
    logger.info(
        "model_load_completed",
        path=str(path),
        model_type=artifacts.get("model_type"),
    )
    return artifacts