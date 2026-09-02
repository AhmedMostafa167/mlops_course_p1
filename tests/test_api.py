import hashlib


def test_health_reports_loaded_predictor(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metadata_reports_artifact_details(api_client, artifact_path):
    response = api_client.get("/metadata")
    body = response.json()

    assert response.status_code == 200
    assert body["model_version"] == "linear-regression-2024-01"
    assert body["features"] == ["PU_DO", "trip_distance"]
    assert body["framework"] == "scikit-learn"
    assert body["training_date"]
    assert (
        body["artifact_hash"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )


def test_predict_returns_contract_and_preserves_request_id(api_client, mock_predictor):
    request_id = "request-123"
    response = api_client.post(
        "/predict",
        headers={"X-Request-ID": request_id},
        json={"pu_do": "74_75", "trip_distance": 1.5},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] == 12.5
    assert body["model_version"] == "linear-regression-2024-01"
    assert body["correlation_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id
    assert body["latency_ms"] >= 0
    mock_predictor.predict_one.assert_called_once_with(
        {"PU_DO": "74_75", "trip_distance": 1.5}
    )


def test_predict_generates_correlation_id(api_client):
    response = api_client.post(
        "/predict",
        json={"pu_do": "74_75", "trip_distance": 1.5},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["correlation_id"]
    assert response.headers["X-Request-ID"] == body["correlation_id"]


def test_predict_batch_returns_ordered_predictions(api_client, mock_predictor):
    response = api_client.post(
        "/predict/batch",
        json={
            "items": [
                {"pu_do": "74_75", "trip_distance": 1.5},
                {"pu_do": "74_76", "trip_distance": 2.5},
            ]
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["predictions"] == [12.5, 18.0]
    assert body["latency_ms"] >= 0
    mock_predictor.predict_batch.assert_called_once_with(
        [
            {"PU_DO": "74_75", "trip_distance": 1.5},
            {"PU_DO": "74_76", "trip_distance": 2.5},
        ]
    )


def test_validation_error_has_safe_shape(api_client):
    response = api_client.post(
        "/predict",
        json={"pu_do": "", "trip_distance": 0},
    )
    body = response.json()

    assert response.status_code == 422
    assert body["detail"] == "Request validation failed"
    assert body["correlation_id"]
    assert response.headers["X-Request-ID"] == body["correlation_id"]


def test_batch_validation_rejects_empty_items(api_client):
    response = api_client.post("/predict/batch", json={"items": []})

    assert response.status_code == 422
    assert response.json()["detail"] == "Request validation failed"


def test_unexpected_predictor_error_is_safe(api_client, mock_predictor):
    mock_predictor.predict_one.side_effect = RuntimeError("private failure details")

    response = api_client.post(
        "/predict",
        json={"pu_do": "74_75", "trip_distance": 1.5},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "private failure details" not in response.text
    assert response.json()["correlation_id"]
