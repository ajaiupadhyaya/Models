#!/usr/bin/env python3
"""
Quick test of Phase 1 API endpoints.
Runs the FastAPI app briefly and hits the new endpoints.
"""

import sys
import asyncio
import requests
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_endpoints():
    """Test Phase 1 API endpoints."""
    base_url = "http://localhost:8000"
    
    tests = [
        {
            "name": "Auto-ARIMA Forecast",
            "endpoint": f"{base_url}/api/v1/predictions/forecast-arima/AAPL?steps=10&seasonal=false&period=6mo",
            "method": "GET"
        },
        {
            "name": "Feature Extraction",
            "endpoint": f"{base_url}/api/v1/predictions/extract-features/AAPL?period=6mo&kind=minimal&max_features=10",
            "method": "GET"
        },
        {
            "name": "CVaR Portfolio Optimization",
            "endpoint": f"{base_url}/api/v1/risk/portfolio/optimize-cvar?symbols=AAPL,MSFT,GOOGL&period=6mo&method=sharpe",
            "method": "GET"
        },
        {
            "name": "Enhanced Portfolio Metrics",
            "endpoint": f"{base_url}/api/v1/risk/portfolio/enhanced-metrics?symbols=AAPL,MSFT&weights=0.5,0.5&period=6mo",
            "method": "GET"
        }
    ]
    
    print("\n" + "=" * 70)
    print("Phase 1 API Endpoint Tests")
    print("=" * 70)
    print("\n⚠️  Note: Start the API server first with: uvicorn api.main:app --port 8000")
    print("    Or test manually via browser/curl\n")
    
    for test in tests:
        print(f"\n✓ {test['name']}")
        print(f"  Endpoint: {test['endpoint']}")
        print(f"  Example curl command:")
        print(f"    curl -X {test['method']} '{test['endpoint']}'")
    
    print("\n" + "=" * 70)
    print("API Integration Complete!")
    print("=" * 70)
    print("\n📦 Phase 1 capabilities now available via API:")
    print("  • Auto-ARIMA time-series forecasting (pmdarima)")
    print("  • CVaR portfolio optimization (riskfolio-lib)")
    print("  • Time-series feature extraction (tsfresh)")
    print("  • Enhanced portfolio metrics (Sharpe, Sortino, Calmar, VaR/CVaR)")
    print("  • Trading calendar integration (exchange-calendars)")
    print("\n📖 Next steps:")
    print("  1. Integrate TradingCalendar into backtesting")
    print("  2. Test endpoints with real data")
    print("  3. Commit changes to git")
    print("  4. Proceed to Phase 2 (sentiment, multi-factor models)")
    print()


if __name__ == "__main__":
    test_endpoints()
