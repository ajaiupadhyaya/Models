#!/usr/bin/env python3
"""
Test Phase 1 implementations.
Run: python test_phase1_integration.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_arima_forecast():
    """Test auto-ARIMA forecasting."""
    print("\n✓ Testing AutoArimaForecaster...")
    
    try:
        import yfinance as yf
        from models.timeseries.advanced_ts import AutoArimaForecaster, UnivariateForecaster
        
        # Get data
        data = yf.download('AAPL', period='1y', progress=False)
        returns = data['Close'].pct_change().dropna()
        
        # Forecast
        forecaster = AutoArimaForecaster()
        result = forecaster.fit(returns)
        
        print(f"  ✅ Model fitted with order: {result['order']}")
        print(f"  ✅ AIC: {result['aic']:.2f}")
        
        forecast, ci = forecaster.forecast(steps=20)
        print(f"  ✅ 20-step forecast generated, shape: {forecast.shape}")
        
        # Test UnivariateForecaster
        uv = UnivariateForecaster(returns)
        uv_result = uv.fit_and_forecast(steps=10)
        print(f"  ✅ UnivariateForecaster: Forecast steps={len(uv_result['forecast'])}")
        print(f"  ✅ Model fitted with order: {result['order']}")
        print(f"  ✅ AIC: {result['aic']:.2f}")
        
        forecast, ci = forecaster.forecast(steps=20)
        print(f"  ✅ 20-step forecast generated, shape: {forecast.shape}")
        
        # Test without seasonality on small data
        forecaster_noseason = AutoArimaForecaster(seasonal=False)
        result2 = forecaster_noseason.fit(returns)
        print(f"  ✅ Non-seasonal model order: {result2['order']}")
        
        print("  ✅ ARIMA test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ ARIMA test FAILED: {str(e)}")
        return False


def test_cvar_optimization():
    """Test CVaR portfolio optimization."""
    print("\n✓ Testing CvaROptimizer...")
    
    try:
        import yfinance as yf
        from models.portfolio.advanced_optimization import (
            CvaROptimizer, 
            RiskParityOptimizer,
            EnhancedPortfolioMetrics
        )
        
        # Get data
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        data = yf.download(tickers, period='1y', progress=False)['Adj Close']
        returns = data.pct_change().dropna()
        
        # CVaR Optimization
        cvar_opt = CvaROptimizer(returns)
        cvar_result = cvar_opt.optimize_cvar()
        
            # Validate and clean returns
            returns = returns.dropna()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
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
        print(f"     Calmar: {metrics['calmar_ratio']:.2f}")
        
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
        
        # Test trading day check
        assert not cal.is_trading_day('2024-01-01'), "2024-01-01 should not be trading day"
        assert cal.is_trading_day('2024-01-02'), "2024-01-02 should be trading day"
        print("  ✅ Trading day checks passed")
        
        # Test next trading day
        next_day = cal.next_trading_day('2024-01-01')
        print(f"  ✅ Next trading day after 2024-01-01: {next_day}")
        
        # Test trading days range
        days = cal.trading_days('2024-01-01', '2024-01-31')
        print(f"  ✅ Trading days in Jan 2024: {len(days)}")
        
        # Test business days count
        count = cal.business_days_count('2024-01-01', '2024-01-31')
        print(f"  ✅ Business days count: {count}")
        
        print("  ✅ Trading calendar test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Trading calendar test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test time-series feature extraction."""
    print("\n✓ Testing TSFeatureExtractor...")
    
    try:
        import yfinance as yf
        from models.timeseries.advanced_ts import TSFeatureExtractor
        
        # Get data
        data = yf.download('AAPL', period='6m', progress=False)
        returns = data['Close'].pct_change().dropna()
        df = pd.DataFrame({'returns': returns})
        
        # Extract features
        features = TSFeatureExtractor.extract_relevant_features(df, 'returns', kind='minimal', max_features=25)
        
        print(f"  ✅ Extracted {len(features.columns)} features")
        print(f"  ✅ Feature names sample: {list(features.columns[:5])}")
        
        print("  ✅ Feature extraction test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Feature extraction test FAILED: {str(e)}")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Phase 1 - Awesome Quant Integration Tests")
    print("=" * 70)
    
    results = []
    
    results.append(("Auto-ARIMA Forecasting", test_arima_forecast()))
    results.append(("CVaR Portfolio Optimization", test_cvar_optimization()))
    results.append(("Trading Calendar", test_trading_calendar()))
    results.append(("Feature Extraction", test_feature_extraction()))
    
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
        sys.exit(0)
    else:
        print(f"❌ {passed_count}/{total_count} tests passed, {total_count - passed_count} failed")
        sys.exit(1)
