"""Regression test: /api/v1/reports/health must not 500 (bool_parsing bug)."""

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_reports_health_returns_200_with_string_status(client):
    response = client.get("/api/v1/reports/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("healthy", "error")
    assert isinstance(body["openai_configured"], bool)
