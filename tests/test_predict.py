import math

import pytest

from prodml.predict import DurationPredictor


@pytest.fixture
def record():
    return {"PU_DO": "74_75", "trip_distance": 2.0}


def test_predictor_loads_from_artifact(artifact_path):
    predictor = DurationPredictor.load(artifact_path)

    assert predictor.features == ["PU_DO", "trip_distance"]
    assert predictor.target == "duration"
    assert set(predictor.metrics) == {"rmse", "mae"}


def test_predict_one_returns_finite_float(predictor, record):
    prediction = predictor.predict_one(record)

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)


def test_predict_one_is_deterministic(predictor, record):
    first = predictor.predict_one(record)
    second = predictor.predict_one(record)

    assert first == pytest.approx(second)


def test_single_and_batch_prediction_agree(predictor, record):
    single = predictor.predict_one(record)
    batch = predictor.predict_batch([record])[0]

    assert single == pytest.approx(batch)


def test_batch_prediction_preserves_input_length(predictor):
    records = [
        {"PU_DO": "74_75", "trip_distance": 1.0},
        {"PU_DO": "74_76", "trip_distance": 3.0},
        {"PU_DO": "75_76", "trip_distance": 5.0},
    ]

    predictions = predictor.predict_batch(records)

    assert len(predictions) == len(records)
    assert all(math.isfinite(value) for value in predictions)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DurationPredictor.load(tmp_path / "missing.pkl")
