"""Tests for labelled production market-data fallbacks."""

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    from api.main import app

    overrides = []
    try:
        from api.auth_api import get_current_user, get_current_user_if_configured

        app.dependency_overrides[get_current_user] = lambda: "test"
        app.dependency_overrides[get_current_user_if_configured] = lambda: "test"
        overrides.extend([get_current_user, get_current_user_if_configured])
    except Exception:
        pass

    try:
        yield TestClient(app)
    finally:
        for dep in overrides:
            app.dependency_overrides.pop(dep, None)


def test_quotes_use_labelled_fallback_when_live_market_data_disabled(client, monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_MARKET_DATA", "true")

    response = client.get("/api/v1/data/quotes?symbols=AAPL,INVALID")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fallback"
    assert "Live quote provider unavailable" in data["warning"]
    assert data["quotes"][0]["symbol"] == "AAPL"
    assert data["quotes"][0]["price"] is not None
    assert data["quotes"][1]["symbol"] == "INVALID"
    assert data["quotes"][1]["price"] is None


def test_market_summary_uses_labelled_fallback_when_live_market_data_disabled(client, monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_MARKET_DATA", "true")

    response = client.get("/api/v1/ai/market-summary?symbols=AAPL,MSFT")

    assert response.status_code == 200
    data = response.json()
    assert data["market_tone"] == "Fallback demo data"
    assert data["analyses"]["AAPL"]["source"] == "fallback"
    assert data["analyses"]["AAPL"]["price"] is not None
    assert data["analyses"]["MSFT"]["source"] == "fallback"


def test_sample_data_uses_labelled_fallback_when_live_market_data_disabled(client, monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_MARKET_DATA", "true")

    response = client.get("/api/v1/backtest/sample-data?symbol=AAPL&period=1mo")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fallback"
    assert "Live historical provider unavailable" in data["warning"]
    assert data["symbol"] == "AAPL"
    assert len(data["candles"]) >= 20
