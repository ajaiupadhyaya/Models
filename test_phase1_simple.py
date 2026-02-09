#!/usr/bin/env python3
"""
Test Phase 1 implementations - Simplified version.
Run: python test_phase1_simple.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all Phase 1 libraries imported successfully."""
    print("\n✓ Testing library imports...")
    
    try:
        import pmdarima
        import riskfolio
        import alphalens
        import exchange_calendars
        import tsfresh
        
        print("  ✅ pmdarima imported")
        print("  ✅ riskfolio imported")
        print("  ✅ alphalens imported")
        print("  ✅ exchange_calendars imported")
        print("  ✅ tsfresh imported")
        print("  ✅ All imports PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Import FAILED: {str(e)}")
        return False


def test_arima_with_synthetic_data():
    """Test auto-ARIMA with synthetic data (no network)."""
    print("\n✓ Testing AutoArimaForecaster with synthetic data...")
    
    try:
        from models.timeseries.advanced_ts import AutoArimaForecaster
        
        # Create synthetic data
        np.random.seed(42)
        synthetic_returns = pd.Series(np.random.randn(500) * 0.02)
        
        # Fit with non-seasonal
        forecaster = AutoArimaForecaster(seasonal=False)
        result = forecaster.fit(synthetic_returns)
        
        print(f"  ✅ Model fitted with order: {result['order']}")
        print(f"  ✅ AIC: {result['aic']:.2f}")
        
        # Forecast
        forecast, ci = forecaster.forecast(steps=10)
        print(f"  ✅ 10-step forecast generated")
        
        print("  ✅ ARIMA test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ ARIMA test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cvar_with_synthetic_data():
    """Test CVaR optimizer with synthetic data (no network)."""
    print("\n✓ Testing CvaROptimizer with synthetic data...")
    
    try:
        from models.portfolio.advanced_optimization import (
            CvaROptimizer, 
            RiskParityOptimizer,
            EnhancedPortfolioMetrics
        )
        
        # Create synthetic returns for 4 assets
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252)
        returns = pd.DataFrame(
            np.random.randn(252, 4) * 0.02,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4'],
            index=dates
        )
        
        # CVaR Optimization
        cvar_opt = CvaROptimizer(returns)
        cvar_result = cvar_opt.optimize_cvar()
        
        print(f"  ✅ CVaR Optimizer:")
        print(f"     Weights: {list(cvar_result['weights'].keys())}")
        print(f"     Expected return: {cvar_result['expected_return']*100:.2f}%")
        print(f"     Sharpe ratio: {cvar_result['sharpe_ratio']:.2f}")
        
        # Risk Parity
        rp_opt = RiskParityOptimizer(returns)
        rp_result = rp_opt.optimize_risk_parity()
        print(f"  ✅ Risk Parity Optimizer: Sharpe={rp_result['sharpe_ratio']:.2f}")
        
        # Enhanced Metrics
        portfolio_returns = (returns * pd.Series(cvar_result['weights'])).sum(axis=1)
        metrics = EnhancedPortfolioMetrics.calculate_metrics(portfolio_returns)
        print(f"  ✅ Enhanced Metrics:")
        print(f"     Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"     Sortino: {metrics['sortino_ratio']:.2f}")
        
        # VaR/CVaR
        var_metrics = EnhancedPortfolioMetrics.calculate_var_cvar(portfolio_returns)
        print(f"  ✅ VaR/CVaR: VaR_95={var_metrics['var']:.4f}, CVaR_95={var_metrics['cvar']:.4f}")
        
        print("  ✅ CVaR optimization test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ CVaR optimization test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_trading_calendar():
    """Test trading calendar."""
    print("\n✓ Testing TradingCalendar...")
    
    try:
        from core.trading_calendar import TradingCalendar
        
        cal = TradingCalendar('NYSE')
        
        # Test trading days in Jan 2024
        days = cal.trading_days('2024-01-01', '2024-01-31')
        print(f"  ✅ Trading days in Jan 2024: {len(days)}")
        
        # Test business days count
        count = cal.business_days_count('2024-01-01', '2024-01-31')
        print(f"  ✅ Business days count: {count}")
        
        # Test is_trading_day
        new_years = pd.Timestamp('2024-01-01')
        is_ny = new_years in cal.calendar.sessions
        print(f"  ✅ 2024-01-01 is trading day: {is_ny} (expected: False)")
        
        # Test next trading day
        next_day = cal.next_trading_day('2024-01-01')
        print(f"  ✅ Next trading day after 2024-01-01: {next_day}")
        
        print("  ✅ Trading calendar test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Trading calendar test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_with_synthetic_data():
    """Test time-series feature extraction with synthetic data."""
    print("\n✓ Testing TSFeatureExtractor with synthetic data...")
    
    try:
        from models.timeseries.advanced_ts import TSFeatureExtractor
        
        # Create synthetic time-series
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5))
        df = pd.DataFrame({'returns': prices.pct_change().dropna()})
        
        # Extract features
        features = TSFeatureExtractor.extract_relevant_features(
            df, 'returns', kind='minimal', max_features=10
        )
        
        print(f"  ✅ Extracted {len(features.columns)} features")
        print(f"  ✅ Feature shape: {features.shape}")
        
        print("  ✅ Feature extraction test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Feature extraction test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Phase 1 - Awesome Quant Integration Tests (Synthetic Data)")
    print("=" * 70)
    
    results = []
    
    results.append(("Library Imports", test_imports()))
    results.append(("Auto-ARIMA Forecasting", test_arima_with_synthetic_data()))
    results.append(("CVaR Portfolio Optimization", test_cvar_with_synthetic_data()))
    results.append(("Trading Calendar", test_trading_calendar()))
    results.append(("Feature Extraction", test_feature_with_synthetic_data()))
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print("=" * 70)
    if passed_count == total_count:
        print(f"✅ All {total_count} tests PASSED!")
        print("\n📦 Phase 1 libraries are ready to use!")
        print("📖 Next: Follow PHASE1_IMPLEMENTATION_GUIDE.md to add API endpoints")
        sys.exit(0)
    else:
        print(f"⚠️  {passed_count}/{total_count} tests passed, {total_count - passed_count} failed")
        sys.exit(1)
