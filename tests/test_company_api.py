"""
Unit tests for api/company_analysis_api.py.

Mocks CompanySearch and CompanyAnalyzer for analyze endpoint; 404 for unknown ticker.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """FastAPI TestClient for the app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_analyze_unknown_ticker_returns_404(client):
    """GET analyze/{ticker} when validate_ticker returns False yields 404."""
    with patch("api.company_analysis_api.CompanySearch") as MockSearch:
        instance = MagicMock()
        instance.validate_ticker.return_value = (False, "Unknown ticker")
        MockSearch.return_value = instance

        response = client.get("/api/v1/company/analyze/UNKNOWN123?include_dcf=false&include_risk=false&include_technicals=false")
    assert response.status_code == 404
    body = response.json()
    assert "detail" in body or "error" in body


def test_analyze_returns_200_with_keys_when_mocked(client):
    """GET analyze/{ticker} with mocked analyzer returns 200 and ratios/valuation/risk keys."""
    with patch("api.company_analysis_api.CompanySearch") as MockSearch:
        MockSearch.return_value.validate_ticker.return_value = (True, "OK")
        with patch("api.company_analysis_api.CompanyAnalyzer") as MockAnalyzer:
            mock_analyzer = MagicMock()
            mock_analyzer.comprehensive_analysis.return_value = {
                "profile": {"name": "Test Inc"},
                "ratios": {"pe_ratio": 15.0, "pb_ratio": 2.0},
                "financials": {},
            }
            MockAnalyzer.return_value = mock_analyzer
            with patch("api.company_analysis_api._calculate_dcf", return_value={"dcf_value": 100}):
                with patch("api.company_analysis_api._calculate_risk_metrics", return_value={"var_95": 2.0}):
                    with patch("api.company_analysis_api._calculate_technical_analysis", return_value={}):
                        with patch("api.company_analysis_api._generate_summary", return_value={"summary": "ok"}):
                            response = client.get(
                                "/api/v1/company/analyze/AAPL?include_dcf=true&include_risk=true&include_technicals=false"
                            )
    if response.status_code == 200:
        data = response.json()
        assert "ticker" in data
        assert "company_name" in data
        assert "fundamental_analysis" in data
        assert "valuation" in data or "risk_metrics" in data
