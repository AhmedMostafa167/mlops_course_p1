from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from structlog import get_logger

from prodml.config import get_settings
from prodml.logging_config import configure_logging

configure_logging()
logger = get_logger(__name__)


def promote_if_better(candidate_version: str, metric: str = "mae") -> bool:
    """Promote a registered version to Production if it beats the current one."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    name = settings.REGISTERED_MODEL_NAME

    candidate_mv = client.get_model_version(name, candidate_version)
    candidate_metric = client.get_run(candidate_mv.run_id).data.metrics[metric]

    production = client.get_latest_versions(name, stages=["Production"])
    production_metric = (
        client.get_run(production[0].run_id).data.metrics[metric]
        if production
        else float("inf")
    )

    logger.info(
        "promotion_check", candidate=candidate_metric, production=production_metric
    )

    if candidate_metric <= production_metric:
        client.transition_model_version_stage(
            name=name,
            version=candidate_version,
            stage="Production",
            archive_existing_versions=True,
        )
        Path("metrics/production.json").write_text(
            json.dumps({"mae": candidate_metric}, indent=2)
        )
        logger.info("model_promoted", version=candidate_version)
        return True

    client.transition_model_version_stage(
        name=name, version=candidate_version, stage="Staging"
    )
    logger.info("model_kept_in_staging", version=candidate_version)
    return False
