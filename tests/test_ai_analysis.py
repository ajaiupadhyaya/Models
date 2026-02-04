"""
Unit tests for core/ai_analysis.py with mocked LLM.
Fixed mock response -> expected structure; no PII/sensitive data in responses.
"""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


def test_ai_service_without_client_returns_graceful_message():
    """When LLM client is not configured, methods return graceful messages, not errors."""
    from core.ai_analysis import AIAnalysisService
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
        svc = AIAnalysisService(api_key=None)
    result = svc.analyze_price_chart("AAPL", pd.DataFrame({"Close": [100], "Volume": [1e6]}))
    assert isinstance(result, str)
    assert "unavailable" in result.lower() or "disabled" in result.lower() or "configured" in result.lower()


def test_sentiment_analysis_structure_with_mock():
    """With mocked LLM returning fixed JSON, sentiment_analysis returns expected keys."""
    from core.ai_analysis import AIAnalysisService
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "sentiment": "bullish",
        "score": 0.8,
        "reasoning": "Positive market context."
    })
    service = AIAnalysisService(api_key="test-key-not-used")
    if not service.client:
        pytest.skip("OpenAI client not available (no API key or package)")
    with patch.object(service.client.chat.completions, "create", return_value=mock_response):
        result = service.sentiment_analysis("The market is bullish.")
    assert "sentiment" in result
    assert result["sentiment"] in ("bullish", "bearish", "neutral", "error")
    assert "score" in result
    assert not any(k in str(result).lower() for k in ("password", "api_key", "secret"))


def test_generate_trading_insight_structure_with_mock():
    """With mocked LLM returning fixed JSON, generate_trading_insight returns action and reasoning."""
    from core.ai_analysis import AIAnalysisService
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "action": "HOLD",
        "reasoning": "Uncertain outlook.",
        "risk_level": "medium",
    })
    service = AIAnalysisService(api_key="test-key-not-used")
    if not service.client:
        pytest.skip("OpenAI client not available")
    with patch.object(service.client.chat.completions, "create", return_value=mock_response):
        result = service.generate_trading_insight("AAPL", 100.0, 102.0, 0.7)
    assert "action" in result
    assert result["action"] in ("BUY", "SELL", "HOLD")
    assert "reasoning" in result or "risk_level" in result
    assert not any(k in str(result).lower() for k in ("password", "api_key", "secret"))


def test_analyze_price_chart_returns_string():
    """analyze_price_chart returns a string (plain-English analysis or error message)."""
    from core.ai_analysis import AIAnalysisService
    df = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1e6, 1.1e6, 1.2e6]})
    service = AIAnalysisService(api_key=None)
    result = service.analyze_price_chart("AAPL", df)
    assert isinstance(result, str)
    assert len(result) > 0
