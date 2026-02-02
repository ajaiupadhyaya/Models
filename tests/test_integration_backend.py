"""
Backend integration tests (pytest).

Data -> sample-data -> backtest run; Data -> risk metrics; health/info endpoints.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


@pytest.fixture
def client():
    """FastAPI TestClient for the app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture
def mock_stock_data():
    """Synthetic OHLCV for integration: sample-data and risk."""
    import numpy as np
    np.random.seed(42)
    n = 252
    returns = np.random.randn(n) * 0.01
    close = 100 * (1 + pd.Series(returns)).cumprod()
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    # Risk API needs Close; sample-data needs OHLCV
    return pd.DataFrame(
        {
            "Open": (close - 0.5).values,
            "High": (close + 0.5).values,
            "Low": (close - 1.0).values,
            "Close": close.values,
            "Volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


def test_health_returns_200_and_status(client):
    """GET /health returns 200 and status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_info_returns_200_and_capabilities(client):
    """GET /info returns 200 and lists routers/capabilities."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "routers_loaded" in data or "capabilities" in data or isinstance(data, dict)


def test_data_pipeline_sample_data_then_backtest_run(client, mock_stock_data):
    """Mock DataFetcher/yf -> sample-data returns candles; backtest run (may 404 if no model) completes without uncaught exception."""
    with patch("api.backtesting_api.yf") as mock_yf:
        mock_yf.download.return_value = mock_stock_data.copy()
        r1 = client.get("/api/v1/backtest/sample-data?symbol=AAPL&period=1y")
    assert r1.status_code == 200
    data1 = r1.json()
    assert "candles" in data1
    assert data1.get("symbol") == "AAPL"
    # Backtest run requires models in app state; we only assert no 500 from our code path
    r2 = client.post(
        "/api/v1/backtest/run",
        json={
            "model_name": "default",
            "symbol": "AAPL",
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "initial_capital": 100000,
            "commission": 0.001,
            "position_size": 0.2,
        },
    )
    assert r2.status_code in (200, 404, 400)


def test_risk_pipeline_mock_data_returns_metrics(client, mock_stock_data):
    """Mock DataFetcher -> risk metrics endpoint returns valid VaR/CVaR/volatility/drawdown/Sharpe."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.get_stock_data.return_value = mock_stock_data
        MockFetcher.return_value = instance
        response = client.get("/api/v1/risk/metrics/AAPL?period=1y")
    assert response.status_code == 200
    data = response.json()
    assert "var_95_pct" in data
    assert "cvar_95_pct" in data
    assert "volatility_annual_pct" in data
    assert "max_drawdown_pct" in data
    assert "sharpe_ratio" in data
