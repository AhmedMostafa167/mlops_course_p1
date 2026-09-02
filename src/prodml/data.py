from __future__ import annotations

import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import structlog

from prodml.config import get_settings

logger = structlog.get_logger(__name__)


REQUIRED_COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def download_data(
    data_url: str | None = None,
    data_path: Path | None = None,
) -> Path:
    """Download the TLC Parquet file if it is not already present.

    The download is written to a temporary file first, with periodic progress
    logs and a socket timeout, so an interrupted download cannot look like a
    running training job or leave a corrupt final artifact.
    """
    settings = get_settings()
    settings.ensure_project_directories()
    data_url = data_url or settings.data_url
    data_path = data_path or settings.data_path

    if data_path.exists():
        logger.info("data_already_present", path=str(data_path))
        return data_path

    partial_path = data_path.with_name(f"{data_path.name}.part")
    logger.info("data_download_started", url=data_url, path=str(data_path))
    downloaded_bytes = 0
    last_logged_at = 0.0

    try:
        with (
            urlopen(data_url, timeout=30) as response,
            partial_path.open("wb") as output,
        ):
            total_bytes = int(response.headers.get("Content-Length") or 0)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded_bytes += len(chunk)
                now = time.monotonic()
                if now - last_logged_at >= 5:
                    logger.info(
                        "data_download_progress",
                        downloaded_mb=round(downloaded_bytes / 1024**2, 1),
                        total_mb=round(total_bytes / 1024**2, 1)
                        if total_bytes
                        else None,
                    )
                    last_logged_at = now
        partial_path.replace(data_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        logger.exception("data_download_failed", url=data_url, path=str(data_path))
        raise

    logger.info(
        "data_download_completed",
        path=str(data_path),
        downloaded_mb=round(downloaded_bytes / 1024**2, 1),
    )
    return data_path


def load_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the raw monthly TLC data needed by the baseline."""
    path = data_path or get_settings().data_path
    logger.info("data_load_started", path=str(path))
    data = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    logger.info("data_load_completed", rows=len(data), columns=len(data.columns))
    return data


def train_validation_split(
    df: pd.DataFrame,
    validation_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split records chronologically, keeping the final fraction for validation."""
    fraction = (
        get_settings().validation_fraction
        if validation_fraction is None
        else validation_fraction
    )
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    split_index = int(len(df) * (1 - fraction))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("DataFrame is too small for the requested split")

    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()
