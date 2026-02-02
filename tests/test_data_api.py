"""
Unit tests for api/data_api.py.

Mocks DataFetcher and FRED for macro endpoint; asserts series shape and keys.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """FastAPI TestClient for the app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_macro_returns_series_structure_when_configured(client):
    """GET macro with mocked FRED returns 200 with series array and keys."""
    # _series_to_list expects iterable with .items() (e.g. pandas Series)
    mock_series = pd.Series([3.7], index=pd.DatetimeIndex(["2024-01-01"]))

    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.fred = True
        instance.get_economic_indicator = MagicMock(return_value=mock_series)
        MockFetcher.return_value = instance
        with patch("config.get_settings", return_value=MagicMock(data=MagicMock(fred_configured=True))):
            response = client.get("/api/v1/data/macro")

    assert response.status_code == 200
    data = response.json()
    assert "series" in data
    assert isinstance(data["series"], list)
    if data["series"]:
        s = data["series"][0]
        assert "series_id" in s
        assert "description" in s
        assert "data" in s


def test_macro_no_fred_key_returns_error_or_empty_series(client):
    """When FRED not configured, macro returns error message and/or empty series."""
    with patch("core.data_fetcher.DataFetcher") as MockFetcher:
        instance = MagicMock()
        instance.fred = False
        MockFetcher.return_value = instance
        response = client.get("/api/v1/data/macro")
    assert response.status_code == 200
    data = response.json()
    assert "series" in data
    assert data["series"] == []
    assert "error" in data
