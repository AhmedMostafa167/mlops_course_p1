from __future__ import annotations

import subprocess
import sys

from structlog import get_logger

from prodml.config import get_settings
from prodml.logging_config import configure_logging

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    settings.export_s3_env()  # mlflow server, launched below, inherits this process's env

    command = [
        "mlflow",
        "server",
        "--backend-store-uri",
        settings.backend_store_uri,
        "--artifacts-destination",
        f"s3://{settings.ARTIFACTS_BUCKET}",
        "--host",
        "0.0.0.0",
        "--port",
        str(settings.MLFLOW_PORT),
    ]
    logger.info("mlflow_server_starting", command=" ".join(command))
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
