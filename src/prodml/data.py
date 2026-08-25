from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from prodml.config import get_settings
import structlog

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
    data_path: Path | None = None,) -> Path:
    """Download the TLC Parquet file if it is not already present."""
    settings = get_settings()
    settings.ensure_project_directories()
    data_url = data_url or settings.data_url
    data_path = data_path or settings.data_path

    if not data_path.exists():
        urlretrieve(data_url, data_path)
    return data_path


def load_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the raw monthly TLC data needed by the baseline."""
    path = data_path or get_settings().data_path
    return pd.read_parquet(path, columns=REQUIRED_COLUMNS)


def train_validation_split(
    df: pd.DataFrame,
    validation_fraction: float | None = None,) -> tuple[pd.DataFrame, pd.DataFrame]:
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
