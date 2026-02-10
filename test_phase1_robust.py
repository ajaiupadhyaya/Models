#!/usr/bin/env python3
"""
Robust Phase 1 tests with synthetic data.
Run: python test_phase1_robust.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_trading_calendar():
    """Test trading calendar functionality."""
    print("\n✓ Testing TradingCalendar...")
    
    try:
        from core.trading_calendar import TradingCalendar
        
        # Initialize calendar
        calendar = TradingCalendar()
        
        # Test is_trading_day
        trading_day = '2024-01-02'
        weekend = '2024-01-06'
        
        assert calendar.is_trading_day(trading_day), "2024-01-02 should be trading day"
        assert not calendar.is_trading_day(weekend), "2024-01-06 is weekend"
        
        print(f"  ✅ Trading day checks passed")
        print(f"  ✅ Next trading day after 2024-01-01: {calendar.next_trading_day('2024-01-01')}")
        
        # Test trading days in range
        trading_days = calendar.trading_days('2024-01-01', '2024-01-31')
        print(f"  ✅ Trading days in Jan 2024: {len(trading_days)}")
        
        # Test business days count
        bdays_count = calendar.business_days_count('2024-01-01', '2024-01-31')
        print(f"  ✅ Business days count: {bdays_count}")
        
        return True, "Trading calendar test PASSED"
    
    except Exception as e:
        return False, f"Trading calendar test FAILED: {str(e)}"


def test_cvар_portfolio():
    """Test CVaR portfolio optimization with synthetic data."""
    print("\n✓ Testing CvaROptimizer...")
    
    try:
        from models.portfolio.advanced_optimization import CvaROptimizer
        
        # Generate synthetic returns (4 assets, 252 trading days)
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, 4) * 0.02 + 0.0005,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4']
        )
        
        # Initialize and optimize
        cvar_opt = CvaROptimizer(returns)
        result = cvar_opt.optimize_cvar()
        
        # Verify results
        assert 'weights' in result, "Result should contain weights"
        assert 'expected_return' in result, "Result should contain expected_return"
        
        # Check weights keys
        weight_keys = list(result['weights'].keys())
        
        print(f"  ✅ CVaR Optimizer:")
        print(f"     Weights: {weight_keys}")
        print(f"     Expected return: {result['expected_return']*100:.2f}%")
        if 'sharpe_ratio' in result:
            print(f"     Sharpe ratio: {result['sharpe_ratio']:.2f}")
        
        return True, "CVaR portfolio optimization test PASSED"
    
    except Exception as e:
        return False, f"CVaR optimization test FAILED: {str(e)}"


def test_feature_extraction():
    """Test feature extraction with synthetic data."""
    print("\n✓ Testing TSFeatureExtractor...")
    
    try:
        from models.timeseries.advanced_ts import TSFeatureExtractor
        
        # Generate synthetic price data
        np.random.seed(42)
        df = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(252) * 0.5)
        }, index=pd.date_range('2023-01-01', periods=252))
        
        # Extract features using static method
        features = TSFeatureExtractor.extract_relevant_features(
            df,
            column='price',
            kind='minimal',
            max_features=20
        )
        
        # Verify output
        assert isinstance(features, pd.DataFrame), "Features should be DataFrame"
        assert len(features.columns) > 0, "Features should not be empty"
        
        print(f"  ✅ Feature Extraction:")
        print(f"     Features extracted: {len(features.columns)}")
        if len(features.columns) > 0:
            print(f"     Sample features: {list(features.columns)[:3]}...")
        
        return True, "Feature extraction test PASSED"
    
    except Exception as e:
        return False, f"Feature extraction test FAILED: {str(e)}"


def test_imports():
    """Test all Phase 1 imports."""
    print("\n✓ Testing Phase 1 Imports...")
    
    try:
        # Import all Phase 1 modules
        from core.trading_calendar import TradingCalendar
        from models.portfolio.advanced_optimization import CvaROptimizer
        from models.timeseries.advanced_ts import TSFeatureExtractor, AutoArimaForecaster
        
        print("  ✅ All Phase 1 modules imported successfully")
        return True, "Imports test PASSED"
    
    except Exception as e:
        return False, f"Imports test FAILED: {str(e)}"


def main():
    """Run all Phase 1 tests."""
    print("=" * 70)
    print("PHASE 1 AWESOME QUANT INTEGRATION - ROBUST TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Trading Calendar", test_trading_calendar),
        ("CVaR Portfolio Optimization", test_cvар_portfolio),
        ("Feature Extraction", test_feature_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed, message = test_func()
        results.append((test_name, passed, message))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    
    for test_name, passed, message in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}".ljust(50) + message.split("test ")[-1])
    
    print("=" * 70)
    print(f"✅ {passed_count}/{total_count} tests passed")
    print("=" * 70)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
