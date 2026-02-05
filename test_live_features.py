#!/usr/bin/env python3
"""
Comprehensive test of live API features.
Tests all critical endpoints that the frontend uses.
"""

import requests
import sys
from typing import Dict, Any, List, Tuple

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

# Define test cases: (name, endpoint, params, expected_keys)
TEST_CASES: List[Tuple[str, str, Dict, List[str]]] = [
    ("Health Check", "/health", {}, ["status", "models_loaded"]),
    ("Stock Data", "/api/v1/backtest/sample-data", {"symbol": "AAPL", "period": "1mo"}, ["candles"]),
    ("Company Search", "/api/v1/company/search", {"query": "Apple"}, ["results"]),
    ("Quick Predict", "/api/v1/predictions/quick-predict", {"symbol": "AAPL"}, ["signal", "recommendation"]),
    ("Quotes (Single)", "/api/v1/data/quotes", {"symbols": "AAPL"}, ["quotes"]),
    ("Quotes (Multi)", "/api/v1/data/quotes", {"symbols": "AAPL,MSFT,GOOGL"}, ["quotes"]),
    ("Macro Data", "/api/v1/data/macro", {}, []),  # May need FRED API key
    ("AI Stock Analysis", "/api/v1/ai/stock-analysis/MSFT", {"include_prediction": "true"}, ["symbol", "technical_analysis"]),
]


def test_endpoint(name: str, endpoint: str, params: Dict, expected_keys: List[str]) -> Tuple[bool, str]:
    """Test a single endpoint."""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=TIMEOUT)
        
        if response.status_code != 200:
            return False, f"Status {response.status_code}"
        
        data = response.json()
        
        # Check for error field
        if "error" in data and data.get("error"):
            return False, f"API Error: {data['error']}"
        
        # Check expected keys
        missing = [k for k in expected_keys if k not in data]
        if missing:
            return False, f"Missing keys: {missing}"
        
        # Validate data is not empty
        if "quotes" in data:
            quotes = data["quotes"]
            if not quotes:
                return False, "Empty quotes array"
            # Check if prices are populated
            has_price = any(q.get("price") is not None for q in quotes)
            if not has_price:
                return False, "All quotes have null prices"
        
        if "candles" in data:
            if not data["candles"]:
                return False, "Empty candles array"
        
        if "results" in data:
            if not data["results"]:
                return False, "Empty results array"
        
        return True, "OK"
        
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection failed"
    except Exception as e:
        return False, f"Exception: {str(e)[:100]}"


def main():
    """Run all tests."""
    print("=" * 80)
    print("🧪 Testing Live API Features")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, endpoint, params, expected_keys in TEST_CASES:
        print(f"Testing: {name:30s} ", end="", flush=True)
        success, message = test_endpoint(name, endpoint, params, expected_keys)
        
        if success:
            print(f"✅ PASS - {message}")
            passed += 1
        elif "FRED_API_KEY" in message or "OpenAI" in message:
            print(f"⚠️  SKIP - {message}")
            skipped += 1
        else:
            print(f"❌ FAIL - {message}")
            failed += 1
    
    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80)
    
    if failed > 0:
        print()
        print("❌ Some features are not operational!")
        sys.exit(1)
    else:
        print()
        print("✅ All critical features are operational!")
        sys.exit(0)


if __name__ == "__main__":
    main()
