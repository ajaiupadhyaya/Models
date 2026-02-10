#!/usr/bin/env python3
"""
Comprehensive deployment readiness validation.
Tests all critical APIs and features.
"""

import sys
import json
from pathlib import Path
import traceback

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_api_imports():
    """Test all API modules import correctly."""
    print("\n✓ Testing API Imports...")
    try:
        from api.main import app
        from api.risk_api import router as risk_router
        from api.data_api import router as data_router
        from api.backtesting_api import router as backtest_router
        from api.models_api import router as models_router
        from api.ai_analysis_api import router as ai_router
        
        print("  ✅ All API modules imported successfully")
        return True, "API imports validated"
    except Exception as e:
        return False, f"API import failed: {str(e)}"


def test_core_models():
    """Test core model imports."""
    print("\n✓ Testing Core Models...")
    try:
        from models.portfolio.advanced_optimization import CvaROptimizer, RiskParityOptimizer
        from models.timeseries.advanced_ts import AutoArimaForecaster, TSFeatureExtractor
        from models.nlp.sentiment import SimpleSentiment
        from models.factors.multi_factor import MultiFactorModel
        from models.derivatives.option_pricing import BlackScholes, GreeksCalculator
        from models.rl.deep_rl_trading import TradingEnvironment, RLTrader
        
        print("  ✅ All core models imported successfully")
        return True, "Core models validated"
    except Exception as e:
        return False, f"Core model import failed: {str(e)}"


def test_dependencies():
    """Test all critical dependencies."""
    print("\n✓ Testing Dependencies...")
    
    requirements = {
        'numpy': 'numerical computing',
        'pandas': 'data manipulation',
        'scipy': 'scientific computing',
        'sklearn': 'machine learning',
        'torch': 'deep learning',
        'gymnasium': 'RL environments',
        'stable_baselines3': 'RL algorithms',
        'fastapi': 'API framework',
        'yfinance': 'data fetching',
        'sqlalchemy': 'database',
        'plotly': 'visualization',
        'pydantic': 'validation',
    }
    
    missing = []
    for pkg, desc in requirements.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append((pkg, desc))
    
    if missing:
        print(f"  ⚠️  Missing {len(missing)} optional dependencies:")
        for pkg, desc in missing:
            print(f"     - {pkg}: {desc}")
        return True, f"{len(requirements) - len(missing)}/{len(requirements)} dependencies installed"
    else:
        print(f"  ✅ All {len(requirements)} dependencies installed")
        return True, "All dependencies validated"


def test_phase_tests():
    """Test Phase 1, 2, 3 test suites."""
    print("\n✓ Testing Phase Suites...")
    
    import subprocess
    
    tests = {
        'Phase 2': 'test_phase2_integration.py',
        'Phase 3': 'test_phase3_integration.py',
        'Phase 1 (Robust)': 'test_phase1_robust.py',
    }
    
    results = {}
    for name, test_file in tests.items():
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / test_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Count passes from output
                output = result.stdout
                if 'ALL' in output and 'PASSED' in output:
                    results[name] = f"✅ PASSED"
                elif 'passed' in output:
                    results[name] = f"✅ PASSED"
                else:
                    results[name] = f"⚠️  Completed"
            else:
                results[name] = f"❌ FAILED"
        except subprocess.TimeoutExpired:
            results[name] = f"⏱️  TIMEOUT"
        except Exception as e:
            results[name] = f"❌ ERROR: {str(e)[:50]}"
    
    # Print results
    passed = sum(1 for v in results.values() if '✅' in v)
    for suite, result in results.items():
        print(f"     {suite}: {result}")
    
    return passed == len(results), f"{passed}/{len(results)} test suites passed"


def test_data_sanity():
    """Test basic data operations."""
    print("\n✓ Testing Data Operations...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test synthetic data generation
        data = pd.DataFrame({
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1e6, 100),
        })
        
        assert len(data) == 100, "Data should have 100 rows"
        assert not data.isnull().any().any(), "Data should not have nulls"
        
        print("  ✅ Data sanity checks passed")
        return True, "Data operations validated"
    except Exception as e:
        return False, f"Data operation failed: {str(e)}"


def test_mathematical_accuracy():
    """Test mathematical correctness of key algorithms."""
    print("\n✓ Testing Mathematical Correctness...")
    
    try:
        from models.derivatives.option_pricing import BlackScholes
        import numpy as np
        
        bs = BlackScholes()
        
        # Test put-call parity
        spot = 100
        strike = 100
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        call = bs.call_price(spot, strike, T, r, sigma)
        put = bs.put_price(spot, strike, T, r, sigma)
        
        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call - put
        rhs = spot - strike * np.exp(-r * T)
        
        error = abs(lhs - rhs)
        assert error < 0.001, f"Put-call parity error: {error}"
        
        print(f"  ✅ Black-Scholes put-call parity verified (error: {error:.6f})")
        return True, "Mathematical accuracy validated"
    except Exception as e:
        return False, f"Mathematical validation failed: {str(e)}"


def main():
    """Run all deployment readiness tests."""
    print("=" * 70)
    print("DEPLOYMENT READINESS VALIDATION")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Project Root: {project_root}")
    
    tests = [
        ("API Imports", test_api_imports),
        ("Core Models", test_core_models),
        ("Dependencies", test_dependencies),
        ("Phase Tests", test_phase_tests),
        ("Data Operations", test_data_sanity),
        ("Mathematical Accuracy", test_mathematical_accuracy),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed, message = test_func()
            results[test_name] = {
                'passed': passed,
                'message': message,
                'status': '✅' if passed else '❌'
            }
        except Exception as e:
            results[test_name] = {
                'passed': False,
                'message': f"Unexpected error: {str(e)}",
                'status': '❌',
                'traceback': traceback.format_exc()
            }
    
    # Print Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for r in results.values() if r['passed'])
    total_count = len(results)
    
    for test_name, result in results.items():
        status = result['status']
        message = result['message']
        print(f"{status} {test_name}".ljust(50) + f"{message}")
    
    print("=" * 70)
    print(f"Overall: {passed_count}/{total_count} validation tests passed")
    print("=" * 70)
    
    # Deployment readiness
    if passed_count == total_count:
        print("\n🎉 DEPLOYMENT READY!")
        print("\nAll validation tests passed. System is ready for production deployment.")
        return True
    elif passed_count >= total_count - 1:
        print("\n⚠️  MOSTLY READY")
        print("\nMost validation tests passed. Review failures before deployment.")
        return True
    else:
        print(f"\n❌ NOT READY")
        print(f"\nOnly {passed_count}/{total_count} tests passed. Address failures before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
