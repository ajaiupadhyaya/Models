"""
Unit tests for api/backtesting_api.py.

Mocks yfinance and DataFetcher for sample-data; mocks engine and data for run endpoint.
"""

import numpy as np
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
def sample_ohlcv_df():
    """Minimal OHLCV DataFrame for sample-data."""
    n = 21
    dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": 100.0 + np.arange(n, dtype=float),
            "High": 101.0 + np.arange(n, dtype=float),
            "Low": 99.0 + np.arange(n, dtype=float),
            "Close": 100.5 + np.arange(n, dtype=float),
            "Volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


def test_sample_data_valid_returns_candles(client, sample_ohlcv_df):
    """GET sample-data with valid symbol returns 200 and candles with schema."""
    with patch("api.backtesting_api.yf") as mock_yf:
        mock_yf.download.return_value = sample_ohlcv_df.copy()
        with patch("api.backtesting_api.get_settings", return_value=MagicMock(data=MagicMock(sample_data_source_default="yfinance"))):
            response = client.get("/api/v1/backtest/sample-data?symbol=AAPL&period=3mo")
    assert response.status_code == 200
    data = response.json()
    assert "candles" in data
    assert data["symbol"] == "AAPL"
    assert data["period"] == "3mo"
    if data["candles"]:
        c = data["candles"][0]
        assert "date" in c
        assert "open" in c and "high" in c and "low" in c and "close" in c
        assert "volume" in c


def test_sample_data_no_data_returns_fallback_for_supported_symbol(client):
    """When no live data is found, sample-data returns labelled fallback candles."""
    with patch("api.backtesting_api.yf.download") as mock_download:
        mock_download.return_value = pd.DataFrame()
        response = client.get("/api/v1/backtest/sample-data?symbol=AAPL&period=1d")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "fallback"
    assert data["warning"]
    assert len(data["candles"]) >= 1


def test_sample_data_no_data_unknown_symbol_returns_empty_candles(client):
    """Unknown symbols do not receive synthetic fallback prices."""
    with patch("api.backtesting_api.yf.download") as mock_download:
        mock_download.return_value = pd.DataFrame()
        response = client.get("/api/v1/backtest/sample-data?symbol=INVALID&period=1d")
    assert response.status_code == 200
    data = response.json()
    assert data["candles"] == []
    assert "error" in data


def test_run_backtest_returns_200_with_mocked_engine(client, sample_ohlcv_df):
    """POST /backtest/run returns legacy response from unified service path."""
    with patch("core.backtest_service.run_backtest") as mock_run_backtest:
        mock_run_backtest.return_value = {
            "symbol": "AAPL",
            "strategy": "sma_cross",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "equity_curve": [{"date": "2024-01-01", "equity": 100000.0}],
            "trades": [],
            "metrics": {"sharpe_ratio": 1.0, "total_return_pct": 2.5},
        }
        response = client.post(
            "/api/v1/backtest/run",
            json={
                "model_name": "default",
                "symbol": "AAPL",
                "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "initial_capital": 100000,
                "commission": 0.001,
                "position_size": 0.2,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "default"
    assert data["symbol"] == "AAPL"
    assert "period" in data
    assert "metrics" in data
    assert "equity_curve" in data
    assert "trades" in data
    assert data["status"] == "success"


def test_technical_backtest_uses_unified_service_contract(client):
    with patch("core.backtest_api_adapter.run_backtest_contract") as mock_contract:
        mock_contract.return_value = {
            "model_name": "sma_cross_20_50",
            "symbol": "AAPL",
            "period": {"start": "2024-01-01", "end": "2024-03-01"},
            "metrics": {"sharpe_ratio": 0.9},
            "equity_curve": [{"date": "2024-01-01", "equity": 100000}],
            "trades": [],
            "status": "success",
        }
        response = client.post(
            "/api/v1/backtest/technical",
            json={
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-03-01",
                "strategy": "sma_cross",
                "fast_period": 20,
                "slow_period": 50,
                "initial_capital": 100000,
                "commission": 0.001,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "sma_cross_20_50"
    assert data["symbol"] == "AAPL"
    assert data["status"] == "success"
    assert "metrics" in data
