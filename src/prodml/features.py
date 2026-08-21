from typing import Any

import pandas as pd


FEATURE_COLUMNS = ["PU_DO", "trip_distance"]
TARGET_COLUMN = "duration"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the baseline target and features, then remove malformed records."""
    prepared = df.copy()

    pickup = pd.to_datetime(prepared["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(prepared["lpep_dropoff_datetime"])
    prepared["duration"] = (dropoff - pickup).dt.total_seconds() / 60

    prepared = prepared.rename(
        columns={
            "PULocationID": "PUlocationID",
            "DOLocationID": "DOlocationID",
        }
    )
    prepared["PU_DO"] = (
        prepared["PUlocationID"].astype("Int64").astype(str)
        + "_"
        + prepared["DOlocationID"].astype("Int64").astype(str)
    )

    return prepared[
        prepared["duration"].between(1, 60)
        & prepared["trip_distance"].between(0.01, 100)
        & prepared["PUlocationID"].notna()
        & prepared["DOlocationID"].notna()
    ].copy()


def to_feature_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert engineered model features to DictVectorizer input records."""
    return df[FEATURE_COLUMNS].to_dict(orient="records")


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the duration target as a pandas Series."""
    return df[TARGET_COLUMN]
