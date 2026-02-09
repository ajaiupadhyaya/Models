#!/usr/bin/env python3
"""
Live API endpoint testing - requires running server.
Tests actual HTTP requests to all critical endpoints.
Run this AFTER starting the API server with: uvicorn api.main:app
"""

import requests
import sys
import json
from typing import Dict, Any, List, Tuple
import time

BASE_URL = "http://localhost:8000"
TIMEOUT = 10


def test_health_endpoints() -> Dict[str, Any]:
    """Test health and info endpoints."""
    print("\n" + "=" * 70)
    print("HEALTH & INFO ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    tests = [
        ("GET /health", f"{BASE_URL}/health", ["status", "models_loaded"]),
        ("GET /info", f"{BASE_URL}/info", ["api_version", "routers_loaded"]),
    ]
    
    for name, url, expected_keys in tests:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    print(f"  ⚠️  {name} - Missing keys: {missing}")
                    results["failed"].append((name, f"Missing keys: {missing}"))
                else:
                    print(f"  ✅ {name}")
                    results["passed"].append(name)
            else:
                print(f"  ❌ {name} - Status {response.status_code}")
                results["failed"].append((name, f"Status {response.status_code}"))
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            results["failed"].append((name, str(e)))
    
    return results


def test_data_endpoints() -> Dict[str, Any]:
    """Test data API endpoints."""
    print("\n" + "=" * 70)
    print("DATA API ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    tests = [
        ("Health Check", f"{BASE_URL}/api/v1/data/health-check", ["sources"]),
        ("Quotes", f"{BASE_URL}/api/v1/data/quotes?symbols=AAPL,MSFT", ["quotes"]),
        ("Macro Data", f"{BASE_URL}/api/v1/data/macro", []),  # May have error if no FRED key
        ("Yield Curve", f"{BASE_URL}/api/v1/data/yield-curve", []),
    ]
    
    for name, url, expected_keys in tests:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                
                # Check for error field (non-critical for optional features)
                if "error" in data and name not in ["Macro Data", "Yield Curve"]:
                    print(f"  ⚠️  {name} - API Error: {data['error']}")
                    results["failed"].append((name, data['error']))
                elif expected_keys:
                    missing = [k for k in expected_keys if k not in data]
                    if missing:
                        print(f"  ⚠️  {name} - Missing keys: {missing}")
                        results["failed"].append((name, f"Missing keys: {missing}"))
                    else:
                        print(f"  ✅ {name}")
                        results["passed"].append(name)
                else:
                    print(f"  ✅ {name}")
                    results["passed"].append(name)
            else:
                print(f"  ❌ {name} - Status {response.status_code}")
                results["failed"].append((name, f"Status {response.status_code}"))
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            results["failed"].append((name, str(e)))
    
    return results


def test_predictions_endpoints() -> Dict[str, Any]:
    """Test predictions API endpoints."""
    print("\n" + "=" * 70)
    print("PREDICTIONS API ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    # Quick predict (GET)
    try:
        url = f"{BASE_URL}/api/v1/predictions/quick-predict?symbol=AAPL"
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "signal" in data or "prediction" in data:
                print(f"  ✅ Quick Predict")
                results["passed"].append("quick-predict")
            else:
                print(f"  ⚠️  Quick Predict - Unexpected response format")
                results["failed"].append(("quick-predict", "Unexpected format"))
        else:
            print(f"  ❌ Quick Predict - Status {response.status_code}")
            results["failed"].append(("quick-predict", f"Status {response.status_code}"))
    except Exception as e:
        print(f"  ❌ Quick Predict - {e}")
        results["failed"].append(("quick-predict", str(e)))
    
    return results


def test_company_endpoints() -> Dict[str, Any]:
    """Test company analysis endpoints."""
    print("\n" + "=" * 70)
    print("COMPANY ANALYSIS ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    tests = [
        ("Company Search", f"{BASE_URL}/api/v1/company/search?query=Apple", ["results"]),
        ("Validate Ticker", f"{BASE_URL}/api/v1/company/validate/AAPL", ["valid"]),
    ]
    
    for name, url, expected_keys in tests:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    print(f"  ⚠️  {name} - Missing keys: {missing}")
                    results["failed"].append((name, f"Missing keys: {missing}"))
                else:
                    print(f"  ✅ {name}")
                    results["passed"].append(name)
            else:
                print(f"  ❌ {name} - Status {response.status_code}")
                results["failed"].append((name, f"Status {response.status_code}"))
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            results["failed"].append((name, str(e)))
    
    return results


def test_backtesting_endpoints() -> Dict[str, Any]:
    """Test backtesting API endpoints."""
    print("\n" + "=" * 70)
    print("BACKTESTING ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    # Sample data endpoint
    try:
        url = f"{BASE_URL}/api/v1/backtest/sample-data?symbol=AAPL&period=1mo"
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if "data" in data or "candles" in data:
                print(f"  ✅ Sample Data")
                results["passed"].append("sample-data")
            else:
                print(f"  ⚠️  Sample Data - Unexpected format")
                results["failed"].append(("sample-data", "Unexpected format"))
        else:
            print(f"  ❌ Sample Data - Status {response.status_code}")
            results["failed"].append(("sample-data", f"Status {response.status_code}"))
    except Exception as e:
        print(f"  ❌ Sample Data - {e}")
        results["failed"].append(("sample-data", str(e)))
    
    return results


def test_monitoring_endpoints() -> Dict[str, Any]:
    """Test monitoring API endpoints."""
    print("\n" + "=" * 70)
    print("MONITORING ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    tests = [
        ("System Metrics", f"{BASE_URL}/api/v1/monitoring/system", []),
        ("System Stats", f"{BASE_URL}/api/v1/monitoring/system/stats", []),
    ]
    
    for name, url, expected_keys in tests:
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                print(f"  ✅ {name}")
                results["passed"].append(name)
            else:
                print(f"  ❌ {name} - Status {response.status_code}")
                results["failed"].append((name, f"Status {response.status_code}"))
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            results["failed"].append((name, str(e)))
    
    return results


def test_models_endpoints() -> Dict[str, Any]:
    """Test models API endpoints."""
    print("\n" + "=" * 70)
    print("MODELS API ENDPOINTS")
    print("=" * 70)
    
    results = {"passed": [], "failed": []}
    
    try:
        url = f"{BASE_URL}/api/v1/models/"
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ List Models - Found {len(data.get('models', []))} models")
            results["passed"].append("list-models")
        else:
            print(f"  ❌ List Models - Status {response.status_code}")
            results["failed"].append(("list-models", f"Status {response.status_code}"))
    except Exception as e:
        print(f"  ❌ List Models - {e}")
        results["failed"].append(("list-models", str(e)))
    
    return results


def main():
    """Run live API tests."""
    print("=" * 70)
    print("LIVE API ENDPOINT TESTING")
    print("=" * 70)
    print(f"Testing API at: {BASE_URL}")
    print("Note: Server must be running (uvicorn api.main:app)")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly!")
            print("Start server with: python -m uvicorn api.main:app --reload")
            return False
    except Exception as e:
        print("❌ Cannot connect to API server!")
        print(f"Error: {e}")
        print("\nStart server with:")
        print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return False
    
    print("✅ Server is running!\n")
    time.sleep(0.5)
    
    # Run all tests
    all_results = {}
    all_results["health"] = test_health_endpoints()
    all_results["data"] = test_data_endpoints()
    all_results["predictions"] = test_predictions_endpoints()
    all_results["company"] = test_company_endpoints()
    all_results["backtesting"] = test_backtesting_endpoints()
    all_results["monitoring"] = test_monitoring_endpoints()
    all_results["models"] = test_models_endpoints()
    
    # Summary
    print("\n" + "=" * 70)
    print("LIVE TEST SUMMARY")
    print("=" * 70)
    
    total_passed = sum(len(r.get("passed", [])) for r in all_results.values())
    total_failed = sum(len(r.get("failed", [])) for r in all_results.values())
    
    print(f"\n✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    
    if total_failed > 0:
        print("\n🔴 FAILURES:")
        for test_name, result in all_results.items():
            if result.get("failed"):
                print(f"\n  {test_name.upper()}:")
                for failure in result["failed"]:
                    if isinstance(failure, tuple):
                        print(f"    - {failure[0]}: {failure[1]}")
                    else:
                        print(f"    - {failure}")
    
    print("\n" + "=" * 70)
    
    if total_failed == 0:
        print("🟢 ALL LIVE TESTS PASSED")
        print("\n✅ API is fully operational and responding correctly")
        print("✅ All endpoints are working as expected")
        return True
    elif total_passed > total_failed * 2:
        print("🟡 MOSTLY WORKING")
        print("\nCore functionality is operational.")
        print("Review failures above - may be optional features.")
        return True
    else:
        print("🔴 MULTIPLE FAILURES DETECTED")
        print("\nReview error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
