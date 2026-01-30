"""
Validate Institutional-Grade Models
Quick verification that all institutional models are properly implemented
"""

import sys
import traceback

def test_imports():
    """Test that all institutional models can be imported."""
    print("=" * 60)
    print("Testing Institutional Model Imports")
    print("=" * 60)
    
    tests = []
    
    # Core institutional models
    try:
        from models.quant.institutional_grade import (
            FamaFrenchFactorModel, GARCHModel, HestonStochasticVolatility,
            TransactionCostModel, AdvancedRiskMetrics, StatisticalValidation,
            BlackLittermanOptimizer, RobustPortfolioOptimizer
        )
        tests.append(("✓ Core Institutional Models", True))
    except Exception as e:
        tests.append(("✗ Core Institutional Models", False))
        print(f"  Error: {e}")
    
    # Advanced econometrics
    try:
        from models.quant.advanced_econometrics import (
            VectorAutoregression, ARIMAGARCH, RegimeSwitchingModel,
            CointegrationAnalysis, KalmanFilter
        )
        tests.append(("✓ Advanced Econometrics", True))
    except Exception as e:
        tests.append(("✗ Advanced Econometrics", False))
        print(f"  Error: {e}")
    
    # Factor models
    try:
        from models.quant.factor_models_institutional import (
            APTModel, StyleFactorModel, RiskFactorModel
        )
        tests.append(("✓ Factor Models", True))
    except Exception as e:
        tests.append(("✗ Factor Models", False))
        print(f"  Error: {e}")
    
    # Advanced options pricing
    try:
        from models.options.advanced_pricing import (
            BinomialTree, SABRModel, FiniteDifferencePricing
        )
        tests.append(("✓ Advanced Options Pricing", True))
    except Exception as e:
        tests.append(("✗ Advanced Options Pricing", False))
        print(f"  Error: {e}")
    
    # Institutional DCF
    try:
        from models.valuation.institutional_dcf import InstitutionalDCF
        tests.append(("✓ Institutional DCF", True))
    except Exception as e:
        tests.append(("✗ Institutional DCF", False))
        print(f"  Error: {e}")
    
    # Institutional backtesting
    try:
        from core.institutional_backtesting import InstitutionalBacktestEngine
        tests.append(("✓ Institutional Backtesting", True))
    except Exception as e:
        tests.append(("✗ Institutional Backtesting", False))
        print(f"  Error: {e}")
    
    # Institutional integration
    try:
        from core.integration_institutional import InstitutionalIntegration
        tests.append(("✓ Institutional Integration", True))
    except Exception as e:
        tests.append(("✗ Institutional Integration", False))
        print(f"  Error: {e}")
    
    # Print results
    print("\nResults:")
    for name, passed in tests:
        print(f"  {name}")
    
    all_passed = all(passed for _, passed in tests)
    return all_passed


def test_basic_functionality():
    """Test basic functionality of key models."""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    import numpy as np
    import pandas as pd
    
    tests = []
    
    # Test Advanced Risk Metrics
    try:
        from models.quant.institutional_grade import AdvancedRiskMetrics
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        es = AdvancedRiskMetrics.expected_shortfall(returns, 0.05)
        sortino = AdvancedRiskMetrics.sortino_ratio(returns)
        tests.append(("✓ Advanced Risk Metrics", True))
    except Exception as e:
        tests.append(("✗ Advanced Risk Metrics", False))
        print(f"  Error: {e}")
    
    # Test Transaction Cost Model
    try:
        from models.quant.institutional_grade import TransactionCostModel
        cost = TransactionCostModel.calculate_total_cost(
            trade_size=1000,
            price=100,
            daily_volume=1000000,
            volatility=0.2
        )
        tests.append(("✓ Transaction Cost Model", True))
    except Exception as e:
        tests.append(("✗ Transaction Cost Model", False))
        print(f"  Error: {e}")
    
    # Test Statistical Validation
    try:
        from models.quant.institutional_grade import StatisticalValidation
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        norm_test = StatisticalValidation.normality_test(returns)
        tests.append(("✓ Statistical Validation", True))
    except Exception as e:
        tests.append(("✗ Statistical Validation", False))
        print(f"  Error: {e}")
    
    # Test Binomial Tree
    try:
        from models.options.advanced_pricing import BinomialTree
        tree = BinomialTree(n_steps=50)
        price = tree.call_price(S=100, K=100, T=0.25, r=0.02, sigma=0.2)
        tests.append(("✓ Binomial Tree", True))
    except Exception as e:
        tests.append(("✗ Binomial Tree", False))
        print(f"  Error: {e}")
    
    # Print results
    print("\nResults:")
    for name, passed in tests:
        print(f"  {name}")
    
    all_passed = all(passed for _, passed in tests)
    return all_passed


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("INSTITUTIONAL-GRADE MODEL VALIDATION")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Functionality: {'✓ PASS' if functionality_ok else '✗ FAIL'}")
    
    if imports_ok and functionality_ok:
        print("\n✅ ALL TESTS PASSED - Institutional models ready!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
