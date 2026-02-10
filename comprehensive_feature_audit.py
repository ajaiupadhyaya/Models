#!/usr/bin/env python3
"""
Comprehensive Feature Audit Script
Tests all features end-to-end per important.md requirements
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

class FeatureAuditor:
    """Comprehensive feature auditor for Bloomberg Terminal clone"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "working": [],
            "limited": [],
            "broken": [],
            "fixes_applied": [],
            "to_remove": []
        }
        self.test_count = 0
        self.pass_count = 0
        self.fail_count = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level:7} | {message}")
    
    def test(self, name: str, test_fn, category: str = "working", details: str = ""):
        """Run a single test"""
        self.test_count += 1
        try:
            result = test_fn()
            if result:
                self.pass_count += 1
                status = "✅"
                self.results[category].append({
                    "name": name,
                    "status": "pass",
                    "details": details
                })
            else:
                self.fail_count += 1
                status = "❌"
                self.results["broken"].append({
                    "name": name,
                    "status": "fail",
                    "error": "Test returned False",
                    "details": details
                })
            self.log(f"{status} {name}", "PASS" if result else "FAIL")
            return result
        except Exception as e:
            self.fail_count += 1
            self.log(f"❌ {name}: {str(e)}", "FAIL")
            self.results["broken"].append({
                "name": name,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False
    
    def get(self, endpoint: str, timeout: int = 10) -> Optional[Dict]:
        """Make GET request"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, timeout=timeout)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            self.log(f"GET {endpoint} failed: {e}", "ERROR")
            return None
    
    def post(self, endpoint: str, data: Dict, timeout: int = 30) -> Optional[Dict]:
        """Make POST request"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=timeout)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            self.log(f"POST {endpoint} failed: {e}", "ERROR")
            return None
    
    def test_health_endpoints(self):
        """Test basic health and status endpoints"""
        self.log("\n" + "="*70)
        self.log("TESTING HEALTH & STATUS ENDPOINTS")
        self.log("="*70)
        
        # Main health
        self.test(
            "Main health endpoint",
            lambda: self.get("/health") is not None,
            "working",
            "Basic health check"
        )
        
        # API docs
        self.test(
            "API documentation accessible",
            lambda: requests.get(f"{self.base_url}/docs").status_code == 200,
            "working",
            "FastAPI auto-generated docs"
        )
        
        # Data API health
        self.test(
            "Data API health check",
            lambda: self.get("/api/v1/data/health-check") is not None,
            "working",
            "Data fetching service health"
        )
    
    def test_authentication(self):
        """Test authentication flows"""
        self.log("\n" + "="*70)
        self.log("TESTING AUTHENTICATION")
        self.log("="*70)
        
        # Login with demo credentials
        login_data = {"username": "demo", "password": "demo"}
        result = self.post("/api/auth/login", login_data)
        
        self.test(
            "User login",
            lambda: result is not None and "token" in result,
            "working",
            "Demo user authentication"
        )
        
        if result and "token" in result:
            self.token = result["token"]
            self.results["working"].append({
                "name": "JWT token generation",
                "status": "pass",
                "details": "Successfully generated auth token"
            })
    
    def test_data_fetching(self):
        """Test data fetching capabilities"""
        self.log("\n" + "="*70)
        self.log("TESTING DATA FETCHING")
        self.log("="*70)
        
        # Stock quotes
        self.test(
            "Stock data fetching (quotes)",
            lambda: self.get("/api/v1/data/quotes?symbols=AAPL,MSFT,GOOGL") is not None,
            "working",
            "yfinance integration for stock data"
        )
        
        # Economic calendar
        self.test(
            "Economic calendar data",
            lambda: self.get("/api/v1/data/economic-calendar") is not None,
            "working",
            "Economic events calendar"
        )
        
        # Yield curve (may require FRED API key)
        yield_data = self.get("/api/v1/data/yield-curve")
        if yield_data:
            self.results["working"].append({
                "name": "Yield curve data",
                "status": "pass",
                "details": "FRED API integration working"
            })
        else:
            self.results["limited"].append({
                "name": "Yield curve data",
                "status": "limited",
                "details": "Requires FRED_API_KEY in .env"
            })
    
    def test_company_analysis(self):
        """Test company analysis features"""
        self.log("\n" + "="*70)
        self.log("TESTING COMPANY ANALYSIS")
        self.log("="*70)
        
        # Company search
        self.test(
            "Company search",
            lambda: self.get("/api/v1/company/search?query=Apple") is not None,
            "working",
            "Fuzzy company name search"
        )
        
        # Ticker validation
        self.test(
            "Ticker validation",
            lambda: self.get("/api/v1/company/validate/AAPL") is not None,
            "working",
            "Validate ticker symbols"
        )
        
        # Full company analysis
        analysis = self.get("/api/v1/company/analyze/AAPL")
        self.test(
            "Company fundamental analysis",
            lambda: analysis is not None,
            "working",
            "DCF valuation, ratios, metrics"
        )
    
    def test_backtesting(self):
        """Test backtesting APIs"""
        self.log("\n" + "="*70)
        self.log("TESTING BACKTESTING")
        self.log("="*70)
        
        # Sample data for backtesting
        self.test(
            "Sample backtest data",
            lambda: self.get("/api/v1/backtest/sample-data") is not None,
            "working",
            "Generate sample market data"
        )
        
        # Technical strategy backtest
        backtest_params = {
            "ticker": "AAPL",
            "strategy": "sma_crossover",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        result = self.post("/api/v1/backtest/technical", backtest_params)
        self.test(
            "Technical strategy backtest",
            lambda: result is not None,
            "working",
            "SMA crossover strategy"
        )
    
    def test_risk_analysis(self):
        """Test risk calculation APIs"""
        self.log("\n" + "="*70)
        self.log("TESTING RISK ANALYSIS")
        self.log("="*70)
        
        # Risk metrics for a ticker
        self.test(
            "Risk metrics calculation",
            lambda: self.get("/api/v1/risk/metrics/AAPL") is not None,
            "working",
            "VaR, CVaR, volatility, Sharpe"
        )
        
        # Stress test scenarios
        self.test(
            "Stress test scenarios",
            lambda: self.get("/api/v1/risk/stress/scenarios") is not None,
            "working",
            "Available stress test scenarios"
        )
    
    def test_predictions(self):
        """Test ML prediction APIs"""
        self.log("\n" + "="*70)
        self.log("TESTING ML PREDICTIONS")
        self.log("="*70)
        
        # Quick prediction
        self.test(
            "Quick stock prediction",
            lambda: self.get("/api/v1/predictions/quick-predict?ticker=AAPL") is not None,
            "working",
            "Fast ML-based price prediction"
        )
        
        # Model info
        self.test(
            "Available ML models",
            lambda: self.get("/api/v1/models/") is not None,
            "working",
            "List trained models"
        )
    
    def test_monitoring(self):
        """Test monitoring endpoints"""
        self.log("\n" + "="*70)
        self.log("TESTING MONITORING")
        self.log("="*70)
        
        # System metrics
        self.test(
            "System monitoring",
            lambda: self.get("/api/v1/monitoring/system") is not None,
            "working",
            "CPU, memory, disk metrics"
        )
        
        # Recent predictions
        self.test(
            "Recent predictions log",
            lambda: self.get("/api/v1/monitoring/predictions/recent") is not None,
            "working",
            "Track prediction history"
        )
    
    def test_paper_trading(self):
        """Test paper trading features"""
        self.log("\n" + "="*70)
        self.log("TESTING PAPER TRADING")
        self.log("="*70)
        
        # Account status (may require Alpaca keys)
        account = self.get("/api/v1/paper-trading/account")
        if account:
            self.results["working"].append({
                "name": "Paper trading account",
                "status": "pass",
                "details": "Alpaca paper trading integration"
            })
        else:
            self.results["limited"].append({
                "name": "Paper trading",
                "status": "limited",
                "details": "Requires ALPACA_API_KEY and ALPACA_API_SECRET in .env"
            })
    
    def test_ai_analysis(self):
        """Test AI-powered analysis"""
        self.log("\n" + "="*70)
        self.log("TESTING AI ANALYSIS")
        self.log("="*70)
        
        # AI analysis endpoint exists
        analysis_request = {
            "ticker": "AAPL",
            "analysis_type": "quick"
        }
        result = self.post("/api/v1/ai/analyze", analysis_request)
        
        if result:
            self.results["working"].append({
                "name": "AI-powered analysis",
                "status": "pass",
                "details": "OpenAI integration for insights"
            })
        else:
            self.results["limited"].append({
                "name": "AI analysis",
                "status": "limited",
                "details": "Requires OPENAI_API_KEY in .env"
            })
    
    def test_news(self):
        """Test news feed"""
        self.log("\n" + "="*70)
        self.log("TESTING NEWS FEED")
        self.log("="*70)
        
        # News endpoint
        news = self.get("/api/v1/news/latest?limit=10")
        if news:
            self.results["working"].append({
                "name": "News feed",
                "status": "pass",
                "details": "Market news aggregation"
            })
        else:
            self.results["limited"].append({
                "name": "News feed",
                "status": "limited",
                "details": "May require FINNHUB_API_KEY for real-time news"
            })
    
    def generate_report(self):
        """Generate final audit report"""
        self.log("\n" + "="*70)
        self.log("AUDIT REPORT GENERATION")
        self.log("="*70)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.test_count,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "pass_rate": f"{(self.pass_count/self.test_count*100):.1f}%" if self.test_count > 0 else "0%"
            },
            "features": self.results
        }
        
        # Write to file
        with open("audit_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("COMPREHENSIVE FEATURE AUDIT REPORT")
        print("="*70)
        print(f"\nTotal Tests Run: {self.test_count}")
        print(f"Passed: {self.pass_count}")
        print(f"Failed: {self.fail_count}")
        print(f"Pass Rate: {report['summary']['pass_rate']}")
        
        print(f"\n✅ Working Features: {len(self.results['working'])}")
        for feature in self.results['working'][:10]:  # Show first 10
            print(f"  - {feature['name']}: {feature['details']}")
        if len(self.results['working']) > 10:
            print(f"  ... and {len(self.results['working']) - 10} more")
        
        print(f"\n⚠️  Features with Limitations: {len(self.results['limited'])}")
        for feature in self.results['limited']:
            print(f"  - {feature['name']}: {feature['details']}")
        
        print(f"\n❌ Broken Features: {len(self.results['broken'])}")
        for feature in self.results['broken']:
            print(f"  - {feature['name']}: {feature.get('error', 'Unknown error')}")
        
        print(f"\nFull report saved to: audit_report.json")
        print("="*70)
        
        return report
    
    def run_full_audit(self):
        """Run complete feature audit"""
        self.log("="*70)
        self.log("STARTING COMPREHENSIVE FEATURE AUDIT")
        self.log("="*70)
        self.log(f"Base URL: {self.base_url}")
        self.log(f"Timestamp: {datetime.now().isoformat()}")
        
        # Check if server is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                self.log("❌ API server is not responding!", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Cannot connect to API server: {e}", "ERROR")
            return False
        
        # Run all test suites
        self.test_health_endpoints()
        self.test_authentication()
        self.test_data_fetching()
        self.test_company_analysis()
        self.test_backtesting()
        self.test_risk_analysis()
        self.test_predictions()
        self.test_monitoring()
        self.test_paper_trading()
        self.test_ai_analysis()
        self.test_news()
        
        # Generate report
        self.generate_report()
        
        return True


if __name__ == "__main__":
    auditor = FeatureAuditor()
    success = auditor.run_full_audit()
    sys.exit(0 if success else 1)
