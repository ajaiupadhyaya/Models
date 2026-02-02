"""
Unit tests for api/risk_api.py.

Mocks DataFetcher.get_stock_data to return synthetic price data so
VaR/CVaR/volatility/drawdown/Sharpe are computed without external APIs.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_stock_data():
    """Synthetic daily close prices: 252 days, small returns for stable risk metrics."""
    import numpy as np

    np.random.seed(42)
    n = 252
    returns = np.random.randn(n) * 0.01
    close = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": close})


@pytest.fixture
def client():
    """FastAPI TestClient for the app (risk router included)."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_risk_metrics_returns_200_and_shape(client, mock_stock_data):
    """GET /api/v1/risk/metrics/{ticker} returns 200 and expected keys."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = mock_stock_data
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/AAPL?period=1y")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["period"] == "1y"
        assert "var_95_pct" in data
        assert "var_99_pct" in data
        assert "cvar_95_pct" in data
        assert "cvar_99_pct" in data
        assert "volatility_annual_pct" in data
        assert "max_drawdown_pct" in data
        assert "sharpe_ratio" in data


def test_risk_metrics_no_data_returns_404(client):
    """When get_stock_data returns None, API returns 404."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = None
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/UNKNOWN?period=1y")
        assert response.status_code == 404
        body = response.json()
        assert "No price data" in body.get("detail", "") or "No price data" in body.get("error", "")


def test_risk_metrics_empty_data_returns_404(client):
    """When get_stock_data returns empty DataFrame, API returns 404."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = pd.DataFrame()
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/TICK?period=1y")
        assert response.status_code == 404


def test_risk_metrics_insufficient_data_returns_400(client):
    """When returns have fewer than 20 points, API returns 400."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        # Few days so pct_change().dropna() has < 20 rows
        instance = MagicMock()
        instance.get_stock_data.return_value = pd.DataFrame({"Close": [100.0] * 15})
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/TICK?period=1y")
        assert response.status_code == 400
        body = response.json()
        assert "Insufficient data" in body.get("detail", "") or "Insufficient data" in body.get("error", "")


def test_risk_metrics_numeric_types(client, mock_stock_data):
    """All returned risk metrics are numbers (float)."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = mock_stock_data
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/MSFT?period=1y")
        assert response.status_code == 200
        data = response.json()
        for key in ("var_95_pct", "var_99_pct", "cvar_95_pct", "cvar_99_pct",
                    "volatility_annual_pct", "max_drawdown_pct", "sharpe_ratio"):
            assert isinstance(data[key], (int, float)), f"{key} should be numeric"


def test_risk_metrics_no_close_column_returns_404(client):
    """When DataFrame has no Close column, API returns 404."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = pd.DataFrame({"Open": [100.0] * 30, "High": [101.0] * 30})
        MockFetcher.return_value = instance

        response = client.get("/api/v1/risk/metrics/TICK?period=1y")
        assert response.status_code == 404
        body = response.json()
        assert "No price data" in body.get("detail", "") or "No price data" in body.get("error", "")
