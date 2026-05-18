"""
Contract smoke tests for terminal-critical API routes.

These tests validate that frontend-called routes are mounted and return
non-404 responses with expected response structure.
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app

    try:
        from api.auth_api import get_current_user
        app.dependency_overrides[get_current_user] = lambda: {"sub": "test"}
    except Exception:
        get_current_user = None

    try:
        yield TestClient(app)
    finally:
        if get_current_user is not None:
            app.dependency_overrides.pop(get_current_user, None)


def test_quant_backtest_route_is_mounted(client):
    with patch("core.backtest_api_adapter.run_backtest_contract") as mock_run_backtest:
        mock_run_backtest.return_value = {
            "model_name": "sma_cross",
            "symbol": "AAPL",
            "strategy": "sma_cross",
            "period": {"start": "2024-01-01", "end": "2024-03-01"},
            "metrics": {"sharpe_ratio": 1.2},
            "equity_curve": [],
            "trades": [],
            "status": "success",
        }
        response = client.post(
            "/api/v1/quant/backtest",
            json={
                "symbol": "AAPL",
                "strategy": "sma_cross",
                "start_date": "2024-01-01",
                "end_date": "2024-03-01",
                "initial_capital": 100000,
                "commission": 0.001,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data.get("symbol") == "AAPL"
    assert "metrics" in data


def test_equity_search_route_is_mounted(client):
    with patch("core.db.search_company_profiles") as mock_search:
        mock_search.return_value = [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 1,
            }
        ]
        response = client.get("/api/v1/equity/search?q=AAPL")

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data.get("count", 0) >= 1


def test_news_sentiment_route_is_mounted(client):
    with patch("api.news_sentiment_api._fetch_news_finnhub") as mock_finnhub:
        with patch("api.news_sentiment_api._score_sentiment_vader") as mock_sentiment:
            mock_finnhub.return_value = [
                {
                    "title": "Apple beats earnings",
                    "summary": "Strong quarter",
                    "url": "https://example.com/news/apple",
                    "published": "2026-01-01T00:00:00Z",
                    "source": "Example",
                }
            ]
            mock_sentiment.return_value = 0.6
            response = client.get("/api/v1/news/AAPL?limit=20&days=7")

    assert response.status_code == 200
    data = response.json()
    assert data.get("symbol") == "AAPL"
    assert "items" in data
    assert "aggregate_sentiment_7d" in data


def test_backtest_run_and_quant_backtest_share_contract_shape(client):
    def _mock_contract(**kwargs):
        return {
            "model_name": kwargs.get("model_name", "sma_cross"),
            "symbol": kwargs["symbol"].upper(),
            "strategy": kwargs["strategy"],
            "period": {"start": kwargs["start_date"], "end": kwargs.get("end_date") or "2024-12-31"},
            "metrics": {"sharpe_ratio": 1.1, "total_return_pct": 4.2},
            "equity_curve": [{"date": "2024-01-01", "equity": 100000.0}],
            "trades": [],
            "status": "success",
        }

    with patch("core.backtest_api_adapter.run_backtest_contract", side_effect=_mock_contract):
        run_response = client.post(
            "/api/v1/backtest/run",
            json={
                "model_name": "default",
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "commission": 0.001,
                "position_size": 0.2,
            },
        )
        quant_response = client.post(
            "/api/v1/quant/backtest",
            json={
                "symbol": "AAPL",
                "strategy": "sma_cross",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "commission": 0.001,
            },
        )

    assert run_response.status_code == 200
    assert quant_response.status_code == 200
    run_json = run_response.json()
    quant_json = quant_response.json()
    assert run_json["symbol"] == quant_json["symbol"] == "AAPL"
    assert run_json["metrics"] == quant_json["metrics"]
    assert run_json["equity_curve"] == quant_json["equity_curve"]
    assert run_json["trades"] == quant_json["trades"]
    assert run_json["status"] == quant_json["status"] == "success"
