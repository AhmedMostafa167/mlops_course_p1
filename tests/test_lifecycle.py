from fastapi.testclient import TestClient

from prodml.api.main import app


def test_health_reports_unhealthy_without_predictor(api_client):
    app.state.predictor = None

    response = api_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unhealthy"}


def test_lifespan_loads_and_clears_predictor(api_settings, monkeypatch):
    monkeypatch.setattr("prodml.api.main.get_settings", lambda: api_settings)

    with TestClient(app) as client:
        assert app.state.predictor is not None
        assert client.get("/health").status_code == 200

    assert app.state.predictor is None
