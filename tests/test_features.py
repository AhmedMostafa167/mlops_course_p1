import pandas as pd

from prodml.features import get_target, to_feature_dicts


def test_prepare_features_calculates_duration_and_location_pair(prepared_trip_df):
    row = prepared_trip_df.iloc[0]

    assert row["duration"] == 10.0
    assert row["PU_DO"] == "74_75"


def test_prepare_features_keeps_valid_boundaries(prepared_trip_df):
    assert set(prepared_trip_df["duration"]) == {1.0, 10.0, 60.0}
    assert 0.01 in set(prepared_trip_df["trip_distance"])
    assert 100.0 in set(prepared_trip_df["trip_distance"])


def test_prepare_features_removes_invalid_rows(prepared_trip_df):
    assert len(prepared_trip_df) == 3
    assert prepared_trip_df["PUlocationID"].notna().all()
    assert prepared_trip_df["DOlocationID"].notna().all()
    assert prepared_trip_df["duration"].between(1, 60).all()
    assert prepared_trip_df["trip_distance"].between(0.01, 100).all()


def test_feature_and_target_projection():
    frame = pd.DataFrame(
        {
            "PU_DO": ["1_2"],
            "trip_distance": [2.5],
            "duration": [9.0],
            "ignored": ["not used"],
        }
    )

    assert to_feature_dicts(frame) == [{"PU_DO": "1_2", "trip_distance": 2.5}]
    assert get_target(frame).tolist() == [9.0]
