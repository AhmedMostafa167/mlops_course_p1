import pandas as pd
import pytest

from prodml.data import download_data, load_data, train_validation_split


def test_download_data_skips_existing_file(tmp_path):
    data_path = tmp_path / "trip_data.parquet"
    data_path.write_bytes(b"already here")

    result = download_data(
        data_url="http://example.invalid/never-called", data_path=data_path
    )

    assert result == data_path
    assert data_path.read_bytes() == b"already here"


def test_download_data_cleans_up_partial_file_on_failure(tmp_path, monkeypatch):
    data_path = tmp_path / "trip_data.parquet"

    def _raise(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr("prodml.data.urlopen", _raise)

    with pytest.raises(OSError):
        download_data(data_url="http://example.invalid", data_path=data_path)

    assert not data_path.with_name(f"{data_path.name}.part").exists()
    assert not data_path.exists()


def test_load_data_returns_required_columns(tmp_path):
    data_path = tmp_path / "trip_data.parquet"
    frame = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(["2024-01-01"]),
            "lpep_dropoff_datetime": pd.to_datetime(["2024-01-01 00:10:00"]),
            "PULocationID": [74],
            "DOLocationID": [75],
            "trip_distance": [1.2],
            "extra_column": ["ignored"],
        }
    )
    frame.to_parquet(data_path)

    loaded = load_data(data_path=data_path)

    assert list(loaded.columns) == [
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]
    assert len(loaded) == 1


def test_train_validation_split_rejects_bad_fraction():
    df = pd.DataFrame({"a": range(10)})
    with pytest.raises(ValueError):
        train_validation_split(df, validation_fraction=1.5)


def test_train_validation_split_rejects_too_small_frame():
    df = pd.DataFrame({"a": range(2)})
    with pytest.raises(ValueError):
        train_validation_split(df, validation_fraction=0.99)


def test_train_validation_split_keeps_chronological_order():
    df = pd.DataFrame({"a": range(10)})
    train_df, val_df = train_validation_split(df, validation_fraction=0.2)
    assert list(train_df["a"]) == list(range(8))
    assert list(val_df["a"]) == list(range(8, 10))
