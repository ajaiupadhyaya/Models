"""
Quick start script to test the financial models framework.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from core.data_fetcher import DataFetcher
        from core.visualizations import ChartBuilder
        from core.utils import calculate_returns, calculate_sharpe_ratio
        from models.valuation.dcf_model import DCFModel
        from models.options.black_scholes import BlackScholes
        from models.portfolio.optimization import MeanVarianceOptimizer
        from models.risk.var_cvar import VaRModel
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_fetcher():
    """Test data fetcher."""
    print("\nTesting data fetcher...")
    try:
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        
        # Test stock data (no API key needed)
        stock_data = fetcher.get_stock_data('AAPL', period='6mo')
        print(f"✓ Fetched {len(stock_data)} days of AAPL data")
        
        # Test economic data (requires API key)
        try:
            unemployment = fetcher.get_unemployment_rate()
            if len(unemployment) > 0:
                print(f"✓ Fetched unemployment data (latest: {unemployment.iloc[-1]:.2f}%)")
            else:
                print("⚠ FRED API key not configured (optional)")
        except:
            print("⚠ FRED API key not configured (optional)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_models():
    """Test model calculations."""
    print("\nTesting models...")
    try:
        # Test DCF
        from models.valuation.dcf_model import DCFModel
        dcf = DCFModel([100.0, 120.0, 140.0], terminal_growth_rate=0.03, wacc=0.10)
        ev = dcf.calculate_enterprise_value()
        print(f"✓ DCF model: Enterprise Value = ${ev:,.2f}")
        
        # Test Black-Scholes
        from models.options.black_scholes import BlackScholes
        call_price = BlackScholes.call_price(100, 100, 0.25, 0.05, 0.20)
        print(f"✓ Black-Scholes: Call price = ${call_price:.2f}")
        
        # Test VaR
        import pandas as pd
        import numpy as np
        from models.risk.var_cvar import VaRModel
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        var = VaRModel.calculate_var(returns, method='historical')
        print(f"✓ VaR calculation: {var:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Financial Models Framework - Quick Start Test")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Data Fetcher", test_data_fetcher()))
    results.append(("Models", test_models()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Set up API keys in .env file (optional but recommended)")
        print("2. Run: jupyter lab")
        print("3. Open notebooks/ for examples")
    else:
        print("\n⚠ Some tests failed. Please check error messages above.")
        print("You may need to install dependencies: pip install -r requirements.txt")

if __name__ == '__main__':
    main()
