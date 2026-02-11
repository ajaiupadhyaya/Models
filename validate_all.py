#!/usr/bin/env python3
"""
Comprehensive Validation Script
Tests all modules, imports, and functionality
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results: List[Tuple[str, bool, str]] = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        self.results.append((name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def warn(self, name: str, message: str = ""):
        self.results.append((name, None, message))
        self.warnings += 1
    
    def print_summary(self):
        print(f"\n{BOLD}{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}{END}\n")
        
        for name, passed, message in self.results:
            if passed is True:
                print(f"{GREEN}✓{END} {name:50} {message}")
            elif passed is False:
                print(f"{RED}✗{END} {name:50} {message}")
            else:
                print(f"{YELLOW}⚠{END} {name:50} {message}")
        
        print(f"\n{BOLD}Results:{END} {GREEN}{self.passed} passed{END}, {RED}{self.failed} failed{END}, {YELLOW}{self.warnings} warnings{END}")
        print(f"{'='*80}\n")
        
        return self.failed == 0

def test_core_imports(result: ValidationResult):
    """Test core module imports."""
    print(f"{CYAN}Testing Core Imports...{END}")
    
    tests = [
        ("core.data_fetcher", "DataFetcher"),
        ("core.visualizations", "ChartBuilder"),
        ("core.utils", "calculate_returns"),
        ("core.data_cache", "DataCache"),
        ("core.dashboard", "create_dashboard"),
    ]
    
    for module, name in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            result.add(f"Import: {module}.{name}", True)
        except Exception as e:
            result.add(f"Import: {module}.{name}", False, str(e)[:60])

def test_model_imports(result: ValidationResult):
    """Test model module imports."""
    print(f"{CYAN}Testing Model Imports...{END}")
    
    tests = [
        ("models.valuation.dcf_model", "DCFModel"),
        ("models.options.black_scholes", "BlackScholes"),
        ("models.portfolio.optimization", "MeanVarianceOptimizer"),
        ("models.risk.var_cvar", "VaRModel"),
        ("models.trading.strategies", "SimpleMomentumStrategy"),
        ("models.ml.forecasting", "TimeSeriesForecaster"),
    ]
    
    for module, name in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            result.add(f"Import: {module}.{name}", True)
        except Exception as e:
            result.add(f"Import: {module}.{name}", False, str(e)[:60])

def test_api_imports(result: ValidationResult):
    """Test API module imports."""
    print(f"{CYAN}Testing API Imports...{END}")
    
    tests = [
        ("api.main", "app"),
        ("api.models_api", "router"),
        ("api.predictions_api", "router"),
        ("api.backtesting_api", "router"),
    ]
    
    for module, name in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            result.add(f"Import: {module}.{name}", True)
        except Exception as e:
            result.add(f"Import: {module}.{name}", False, str(e)[:60])

def test_dependencies(result: ValidationResult):
    """Test required dependencies."""
    print(f"{CYAN}Testing Dependencies...{END}")
    
    deps = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("plotly", "Visualizations"),
        ("yfinance", "Market data"),
        ("fastapi", "API framework"),
        ("dash", "Dashboard"),
        ("scipy", "Scientific computing"),
        ("sklearn", "Machine learning"),
    ]
    
    for pkg, desc in deps:
        try:
            __import__(pkg)
            result.add(f"Package: {pkg}", True, desc)
        except ImportError:
            result.add(f"Package: {pkg}", False, "NOT INSTALLED")

def test_file_structure(result: ValidationResult):
    """Test project file structure."""
    print(f"{CYAN}Testing File Structure...{END}")
    
    required_dirs = [
        "core",
        "models",
        "api",
        "notebooks",
        "data",
        "config",
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        "launch.py",
        "api/main.py",
        "core/data_fetcher.py",
        "models/valuation/dcf_model.py",
    ]
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        result.add(f"Directory: {dir_name}", path.exists(), str(path.absolute()))
    
    for file_name in required_files:
        path = Path(file_name)
        result.add(f"File: {file_name}", path.exists(), str(path.absolute()))

def test_functionality(result: ValidationResult):
    """Test basic functionality."""
    print(f"{CYAN}Testing Functionality...{END}")
    
    # Test DCF calculation
    try:
        from models.valuation.dcf_model import DCFModel
        dcf = DCFModel([100, 120, 140], terminal_growth_rate=0.03, wacc=0.10)
        ev = dcf.calculate_enterprise_value()
        result.add("DCF Calculation", True, f"EV=${ev:,.2f}")
    except Exception as e:
        result.add("DCF Calculation", False, str(e)[:60])
    
    # Test Black-Scholes
    try:
        from models.options.black_scholes import BlackScholes
        price = BlackScholes.call_price(100, 100, 0.25, 0.05, 0.20)
        result.add("Black-Scholes Calculation", True, f"Call=${price:.2f}")
    except Exception as e:
        result.add("Black-Scholes Calculation", False, str(e)[:60])
    
    # Test Data Fetcher (without API key)
    try:
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        # This should work even without API keys
        result.add("DataFetcher Initialization", True)
    except Exception as e:
        result.add("DataFetcher Initialization", False, str(e)[:60])

def test_configuration(result: ValidationResult):
    """Test configuration."""
    print(f"{CYAN}Testing Configuration...{END}")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        result.add("Configuration: .env file", True)
    else:
        result.warn("Configuration: .env file", "Not found (optional - add API keys)")
    
    if env_example.exists():
        result.add("Configuration: .env.example", True)
    else:
        result.warn("Configuration: .env.example", "Not found")

def main():
    """Run all validations."""
    print(f"\n{BOLD}{CYAN}{'='*80}")
    print("COMPREHENSIVE PROJECT VALIDATION")
    print(f"{'='*80}{END}\n")
    
    result = ValidationResult()
    
    # Run all tests
    test_file_structure(result)
    test_dependencies(result)
    test_core_imports(result)
    test_model_imports(result)
    test_api_imports(result)
    test_functionality(result)
    test_configuration(result)
    
    # Print summary
    success = result.print_summary()
    
    if success:
        print(f"{GREEN}{BOLD}✓ All critical tests passed!{END}\n")
        print(f"{CYAN}You can now run:{END}")
        print(f"  {BOLD}python launch.py{END} - Launch the unified launcher")
        print(f"  {BOLD}python quick_start.py{END} - Run quick start test")
        print(f"  {BOLD}python run_dashboard.py{END} - Launch dashboard")
        print(f"  {BOLD}./start-api.sh{END} - Start API server\n")
    else:
        print(f"{RED}{BOLD}✗ Some tests failed{END}\n")
        print(f"{YELLOW}Please install missing dependencies:{END}")
        print(f"  {BOLD}pip install -r requirements.txt{END}\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
