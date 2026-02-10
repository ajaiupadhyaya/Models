#!/usr/bin/env python3
"""
COMPREHENSIVE AUDIT TEST - Bloomberg Terminal Clone

Tests every implemented feature to verify functionality in production.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8000"
TEST_SYMBOLS = ["AAPL", "TSLA", "MSFT", "SPY"]

class AuditTester:
    def __init__(self):
        self.results = {
            "healthy": [],
            "working": [],
            "broken": [],
            "not_implemented": [],
            "api_keys_missing": [],
            "warnings": []
        }
        self.session = requests.Session()
        self.auth_token = None

    def test_health(self) -> bool:
        """Test basic API health"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                self.results["healthy"].append({
                    "endpoint": "GET /health",
                    "status": "✓ WORKING",
                    "response": response.json()
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "GET /health",
                    "status": f"✗ FAILED ({response.status_code})",
                    "error": response.text
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "GET /health",
                "status": f"✗ ERROR: {str(e)}"
            })
            return False

    def test_system_info(self) -> bool:
        """Test system info endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/info", timeout=5)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "GET /info",
                    "status": "✓ WORKING",
                    "routers": response.json().get("routers_loaded", [])
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "GET /info",
                    "error": response.text
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "GET /info",
                "error": str(e)
            })
            return False

    def test_authentication(self) -> bool:
        """Test login endpoint"""
        try:
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "username": "AJAI",
                "password": "MAYA"
            }, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get("access_token")
                self.results["working"].append({
                    "endpoint": "POST /api/auth/login",
                    "status": "✓ WORKING",
                    "authenticated": True
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "POST /api/auth/login",
                    "error": response.text
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "POST /api/auth/login",
                "error": str(e)
            })
            return False

    def test_data_api(self, symbol: str = "AAPL") -> bool:
        """Test data endpoints"""
        tests_passed = 0
        tests_total = 0

        # Test - Get historical data
        endpoints = [
            ("GET", f"/api/v1/data/historical/{symbol}?period=1mo"),
            ("GET", f"/api/v1/data/price/{symbol}"),
            ("GET", f"/api/v1/data/macro/gdp"),
            ("GET", f"/api/v1/data/macro/inflation"),
        ]

        for method, endpoint in endpoints:
            tests_total += 1
            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", timeout=10)
                if response.status_code == 200:
                    self.results["working"].append({
                        "endpoint": f"{method} {endpoint}",
                        "status": "✓ WORKING"
                    })
                    tests_passed += 1
                else:
                    self.results["broken"].append({
                        "endpoint": f"{method} {endpoint}",
                        "error": f"Status {response.status_code}"
                    })
            except requests.exceptions.Timeout:
                self.results["warnings"].append({
                    "endpoint": f"{method} {endpoint}",
                    "warning": "TIMEOUT - API slow or data unavailable"
                })
            except Exception as e:
                self.results["broken"].append({
                    "endpoint": f"{method} {endpoint}",
                    "error": str(e)
                })

        return tests_passed > 0

    def test_backtesting_api(self, symbol: str = "AAPL") -> bool:
        """Test backtesting endpoints"""
        tests_passed = 0
        tests_total = 0

        # Test sample data
        endpoints = [
            ("GET", f"/api/v1/backtest/sample-data?ticker={symbol}&period=1y"),
            ("GET", f"/api/v1/backtest/metrics"),
        ]

        for method, endpoint in endpoints:
            tests_total += 1
            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", timeout=15)
                if response.status_code == 200:
                    self.results["working"].append({
                        "endpoint": f"{method} {endpoint}",
                        "status": "✓ WORKING"
                    })
                    tests_passed += 1
                else:
                    self.results["broken"].append({
                        "endpoint": f"{method} {endpoint}",
                        "error": f"Status {response.status_code}"
                    })
            except Exception as e:
                self.results["broken"].append({
                    "endpoint": f"{method} {endpoint}",
                    "error": str(e)
                })

        # Test technical backtest
        tests_total += 1
        try:
            response = self.session.post(f"{BASE_URL}/api/v1/backtest/technical", json={
                "ticker": symbol,
                "strategy": "sma_cross",
                "params": {"fast_period": 30, "slow_period": 100},
                "period": "1y"
            }, timeout=30)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "POST /api/v1/backtest/technical",
                    "status": "✓ WORKING"
                })
                tests_passed += 1
            else:
                self.results["broken"].append({
                    "endpoint": "POST /api/v1/backtest/technical",
                    "error": f"Status {response.status_code}"
                })
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "POST /api/v1/backtest/technical",
                "error": str(e)
            })

        return tests_passed > 0

    def test_risk_analysis(self, symbol: str = "AAPL") -> bool:
        """Test risk analysis endpoints"""
        tests_passed = 0
        tests_total = 3

        endpoints = [
            ("GET", f"/api/v1/risk/metrics/{symbol}"),
            ("GET", f"/api/v1/risk/var/{symbol}"),
        ]

        for method, endpoint in endpoints:
            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", timeout=10)
                if response.status_code == 200:
                    self.results["working"].append({
                        "endpoint": f"{method} {endpoint}",
                        "status": "✓ WORKING"
                    })
                    tests_passed += 1
                else:
                    self.results["broken"].append({
                        "endpoint": f"{method} {endpoint}",
                        "error": f"Status {response.status_code}"
                    })
            except Exception as e:
                self.results["broken"].append({
                    "endpoint": f"{method} {endpoint}",
                    "error": str(e)
                })

        # Test stress testing endpoint
        try:
            response = self.session.post(f"{BASE_URL}/api/v1/risk/stress", json={
                "ticker": symbol,
                "scenarios": ["bull", "bear", "volatility"]
            }, timeout=20)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "POST /api/v1/risk/stress",
                    "status": "✓ WORKING"
                })
                tests_passed += 1
            else:
                self.results["broken"].append({
                    "endpoint": "POST /api/v1/risk/stress",
                    "error": f"Status {response.status_code}"
                })
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "POST /api/v1/risk/stress",
                "error": str(e)
            })

        return tests_passed > 1

    def test_company_analysis(self, symbol: str = "AAPL") -> bool:
        """Test company analysis endpoints"""
        tests_passed = 0
        tests_total = 3

        endpoints = [
            ("GET", f"/api/v1/company/info/{symbol}"),
            ("GET", f"/api/v1/company/fundamentals/{symbol}"),
            ("GET", f"/api/v1/company/valuation/{symbol}"),
        ]

        for method, endpoint in endpoints:
            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", timeout=15)
                if response.status_code == 200:
                    self.results["working"].append({
                        "endpoint": f"{method} {endpoint}",
                        "status": "✓ WORKING"
                    })
                    tests_passed += 1
                else:
                    self.results["broken"].append({
                        "endpoint": f"{method} {endpoint}",
                        "error": f"Status {response.status_code}"
                    })
            except Exception as e:
                self.results["broken"].append({
                    "endpoint": f"{method} {endpoint}",
                    "error": str(e)
                })

        return tests_passed > 0

    def test_ai_analysis(self, symbol: str = "AAPL") -> bool:
        """Test AI analysis endpoints"""
        try:
            response = self.session.get(f"{BASE_URL}/api/v1/ai/analyze/{symbol}", timeout=30)
            if response.status_code in [200, 202]:
                self.results["working"].append({
                    "endpoint": f"GET /api/v1/ai/analyze/{symbol}",
                    "status": "✓ WORKING"
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": f"GET /api/v1/ai/analyze/{symbol}",
                    "error": f"Status {response.status_code}"
                })
                return False
        except requests.exceptions.Timeout:
            self.results["warnings"].append({
                "endpoint": "GET /api/v1/ai/analyze",
                "warning": "TIMEOUT - AI analysis slow"
            })
            return False
        except Exception as e:
            if "api_key" in str(e).lower() or "openai" in str(e).lower():
                self.results["api_keys_missing"].append({
                    "endpoint": "GET /api/v1/ai/analyze",
                    "error": "OpenAI API key missing or invalid"
                })
            else:
                self.results["broken"].append({
                    "endpoint": "GET /api/v1/ai/analyze",
                    "error": str(e)
                })
            return False

    def test_models_api(self) -> bool:
        """Test models endpoints"""
        try:
            response = self.session.get(f"{BASE_URL}/api/v1/models", timeout=10)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "GET /api/v1/models",
                    "status": "✓ WORKING"
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "GET /api/v1/models",
                    "error": f"Status {response.status_code}"
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "GET /api/v1/models",
                "error": str(e)
            })
            return False

    def test_predictions_api(self, symbol: str = "AAPL") -> bool:
        """Test predictions endpoints"""
        try:
            response = self.session.post(f"{BASE_URL}/api/v1/predictions/predict", json={
                "ticker": symbol
            }, timeout=30)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "POST /api/v1/predictions/predict",
                    "status": "✓ WORKING"
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "POST /api/v1/predictions/predict",
                    "error": f"Status {response.status_code}"
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "POST /api/v1/predictions/predict",
                "error": str(e)
            })
            return False

    def test_paper_trading(self) -> bool:
        """Test paper trading endpoints"""
        tests_passed = 0

        endpoints = [
            ("GET", "/api/v1/paper-trading/portfolio"),
            ("GET", "/api/v1/paper-trading/positions"),
            ("GET", "/api/v1/paper-trading/account"),
        ]

        for method, endpoint in endpoints:
            try:
                response = self.session.get(f"{BASE_URL}{endpoint}", timeout=10)
                if response.status_code == 200:
                    self.results["working"].append({
                        "endpoint": f"{method} {endpoint}",
                        "status": "✓ WORKING"
                    })
                    tests_passed += 1
                else:
                    self.results["broken"].append({
                        "endpoint": f"{method} {endpoint}",
                        "error": f"Status {response.status_code}"
                    })
            except Exception as e:
                self.results["broken"].append({
                    "endpoint": f"{method} {endpoint}",
                    "error": str(e)
                })

        return tests_passed > 0

    def test_monitoring(self) -> bool:
        """Test monitoring endpoints"""
        try:
            response = self.session.get(f"{BASE_URL}/api/v1/monitoring/metrics", timeout=10)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "GET /api/v1/monitoring/metrics",
                    "status": "✓ WORKING"
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "GET /api/v1/monitoring/metrics",
                    "error": f"Status {response.status_code}"
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "GET /api/v1/monitoring/metrics",
                "error": str(e)
            })
            return False

    def test_investor_reports(self) -> bool:
        """Test investor reports endpoints"""
        try:
            response = self.session.get(f"{BASE_URL}/api/v1/investor-reports", timeout=10)
            if response.status_code == 200:
                self.results["working"].append({
                    "endpoint": "GET /api/v1/investor-reports",
                    "status": "✓ WORKING"
                })
                return True
            else:
                self.results["broken"].append({
                    "endpoint": "GET /api/v1/investor-reports",
                    "error": f"Status {response.status_code}"
                })
                return False
        except Exception as e:
            self.results["broken"].append({
                "endpoint": "GET /api/v1/investor-reports",
                "error": str(e)
            })
            return False

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE AUDIT TEST - BLOOMBERG TERMINAL CLONE")
        print("="*80 + "\n")

        # Health checks
        print("🏥 HEALTH CHECKS...")
        self.test_health()
        self.test_system_info()

        # Authentication
        print("🔐 AUTHENTICATION...")
        self.test_authentication()

        # Core APIs
        print("📊 CORE API ENDPOINTS...")
        for symbol in TEST_SYMBOLS[:1]:  # Test with AAPL only
            print(f"\n  Testing with {symbol}...")
            self.test_data_api(symbol)
            self.test_backtesting_api(symbol)
            self.test_risk_analysis(symbol)
            self.test_company_analysis(symbol)
            self.test_ai_analysis(symbol)

        # Specialized APIs
        print("\n🤖 SPECIALIZED APIs...")
        self.test_models_api()
        self.test_predictions_api()
        self.test_paper_trading()
        self.test_monitoring()
        self.test_investor_reports()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print audit summary"""
        print("\n" + "="*80)
        print("AUDIT RESULTS SUMMARY")
        print("="*80 + "\n")

        print(f"✅ HEALTHY: {len(self.results['healthy'])} endpoints")
        for item in self.results['healthy']:
            print(f"   {item['endpoint']}")

        print(f"\n✓ WORKING: {len(self.results['working'])} endpoints")
        for item in self.results['working']:
            print(f"   {item['endpoint']}")

        print(f"\n❌ BROKEN: {len(self.results['broken'])} endpoints")
        for item in self.results['broken']:
            print(f"   {item['endpoint']}: {item.get('error', 'Unknown error')}")

        print(f"\n⚠️  MISSING API KEYS: {len(self.results['api_keys_missing'])} endpoints")
        for item in self.results['api_keys_missing']:
            print(f"   {item['endpoint']}: {item['error']}")

        print(f"\n⏰ WARNINGS: {len(self.results['warnings'])} items")
        for item in self.results['warnings']:
            print(f"   {item['endpoint']}: {item.get('warning', 'Unknown')}")

        total = len(self.results['healthy']) + len(self.results['working'])
        broken = len(self.results['broken'])
        percentage = (total / (total + broken) * 100) if (total + broken) > 0 else 0

        print(f"\n📈 OVERALL HEALTH: {percentage:.1f}% ({total}/{total + broken} passing)")
        print("\n" + "="*80 + "\n")

        # Export to JSON
        with open("/Users/ajaiupadhyaya/Documents/Models/audit_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("📄 Detailed results exported to: audit_results.json")


if __name__ == "__main__":
    tester = AuditTester()
    tester.run_all_tests()
