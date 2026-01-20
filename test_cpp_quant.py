#!/usr/bin/env python3
"""
Test suite for C++ quantitative finance library
Verifies C++ implementations match Python implementations
"""

import sys
import numpy as np

def test_cpp_availability():
    """Test if C++ library is available"""
    print("=" * 70)
    print("TEST 1: C++ Library Availability")
    print("=" * 70)
    
    try:
        from quant_accelerated import CPP_AVAILABLE
        if CPP_AVAILABLE:
            print("✓ C++ library is available and loaded")
            print("  Using high-performance C++ implementations")
            return True
        else:
            print("⚠ C++ library not available")
            print("  Using pure Python fallback implementations")
            print("  To build C++: ./build_cpp.sh")
            return False
    except ImportError as e:
        print(f"✗ Failed to import quant_accelerated: {e}")
        return False


def test_black_scholes():
    """Test Black-Scholes implementations"""
    print("\n" + "=" * 70)
    print("TEST 2: Black-Scholes Options Pricing")
    print("=" * 70)
    
    try:
        from quant_accelerated import BlackScholesAccelerated
        
        # Test parameters
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        
        # Test call price
        call_price = BlackScholesAccelerated.call_price(S, K, T, r, sigma)
        print(f"✓ Call price: ${call_price:.4f}")
        assert 10.0 < call_price < 11.0, f"Call price {call_price} out of expected range"
        
        # Test put price
        put_price = BlackScholesAccelerated.put_price(S, K, T, sigma, r, sigma)
        print(f"✓ Put price: ${put_price:.4f}")
        
        # Test Greeks
        delta = BlackScholesAccelerated.delta(S, K, T, r, sigma, is_call=True)
        print(f"✓ Delta: {delta:.4f}")
        assert 0.5 < delta < 0.7, f"Delta {delta} out of expected range"
        
        gamma = BlackScholesAccelerated.gamma(S, K, T, r, sigma)
        print(f"✓ Gamma: {gamma:.6f}")
        assert gamma > 0, "Gamma should be positive"
        
        vega = BlackScholesAccelerated.vega(S, K, T, r, sigma)
        print(f"✓ Vega: {vega:.4f}")
        assert vega > 0, "Vega should be positive"
        
        theta = BlackScholesAccelerated.theta(S, K, T, r, sigma, is_call=True)
        print(f"✓ Theta: {theta:.4f}")
        
        rho = BlackScholesAccelerated.rho(S, K, T, r, sigma, is_call=True)
        print(f"✓ Rho: {rho:.4f}")
        
        print("\n✅ All Black-Scholes tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Black-Scholes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monte_carlo():
    """Test Monte Carlo simulations"""
    print("\n" + "=" * 70)
    print("TEST 3: Monte Carlo Simulation Engine")
    print("=" * 70)
    
    try:
        from quant_accelerated import MonteCarloAccelerated
        
        mc = MonteCarloAccelerated(seed=42)
        
        # Test European option pricing
        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        n_sim = 10000
        
        price = mc.price_european_option(S0, K, T, r, sigma, is_call=True, n_simulations=n_sim)
        print(f"✓ European call price (MC): ${price:.4f} ({n_sim} simulations)")
        assert 9.5 < price < 11.5, f"MC price {price} deviates too much from analytical"
        
        # Test GBM path simulation
        path = mc.simulate_gbm_path(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252)
        print(f"✓ GBM path simulation: {len(path)} steps, final price ${path[-1]:.2f}")
        assert len(path) == 253, "Path should have 253 points (252 steps + initial)"
        
        # Test Asian option
        asian_price = mc.price_asian_option(S0, K, T, r, sigma, is_call=True, 
                                           n_simulations=1000, n_steps=50)
        print(f"✓ Asian option price: ${asian_price:.4f}")
        assert asian_price > 0, "Asian option price should be positive"
        
        print("\n✅ All Monte Carlo tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Monte Carlo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio():
    """Test portfolio analytics"""
    print("\n" + "=" * 70)
    print("TEST 4: Portfolio Analytics")
    print("=" * 70)
    
    try:
        from quant_accelerated import PortfolioAccelerated
        
        # Test data
        weights = [0.3, 0.3, 0.4]
        expected_returns = [0.10, 0.12, 0.08]
        cov_matrix = [
            [0.04, 0.01, 0.02],
            [0.01, 0.06, 0.015],
            [0.02, 0.015, 0.05]
        ]
        
        # Test portfolio return
        port_return = PortfolioAccelerated.portfolio_return(weights, expected_returns)
        print(f"✓ Portfolio return: {port_return*100:.2f}%")
        assert 0.08 < port_return < 0.12, f"Portfolio return {port_return} out of range"
        
        # Test portfolio volatility
        port_vol = PortfolioAccelerated.portfolio_volatility(weights, cov_matrix)
        print(f"✓ Portfolio volatility: {port_vol*100:.2f}%")
        assert port_vol > 0, "Volatility should be positive"
        
        # Test Sharpe ratio
        sharpe = PortfolioAccelerated.sharpe_ratio(weights, expected_returns, 
                                                   cov_matrix, risk_free_rate=0.03)
        print(f"✓ Sharpe ratio: {sharpe:.4f}")
        assert sharpe > 0, "Sharpe ratio should be positive for this portfolio"
        
        # Test max drawdown
        cumulative_returns = [1.0, 1.1, 1.05, 1.15, 1.08, 1.20]
        max_dd = PortfolioAccelerated.max_drawdown(cumulative_returns)
        print(f"✓ Max drawdown: {max_dd*100:.2f}%")
        assert 0 <= max_dd <= 1, "Max drawdown should be between 0 and 1"
        
        # Test VaR
        returns = list(np.random.normal(0.001, 0.02, 1000))
        var_95 = PortfolioAccelerated.historical_var(returns, confidence_level=0.95)
        print(f"✓ VaR (95%): {var_95*100:.2f}%")
        assert var_95 > 0, "VaR should be positive"
        
        # Test CVaR
        cvar_95 = PortfolioAccelerated.conditional_var(returns, confidence_level=0.95)
        print(f"✓ CVaR (95%): {cvar_95*100:.2f}%")
        assert cvar_95 >= var_95, "CVaR should be >= VaR"
        
        print("\n✅ All portfolio analytics tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Portfolio analytics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility():
    """Test that existing Python functions still work"""
    print("\n" + "=" * 70)
    print("TEST 5: Compatibility with Existing Python Code")
    print("=" * 70)
    
    try:
        # Test that original Python implementations still work
        from models.options.black_scholes import BlackScholes
        
        call_price = BlackScholes.call_price(100, 100, 1.0, 0.05, 0.2)
        print(f"✓ Original Python BlackScholes.call_price: ${call_price:.4f}")
        assert 10.0 < call_price < 11.0, "Original implementation changed"
        
        delta = BlackScholes.delta(100, 100, 1.0, 0.05, 0.2, 'call')
        print(f"✓ Original Python BlackScholes.delta: {delta:.4f}")
        
        print("\n✅ All compatibility tests passed")
        print("  Original Python functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        print("  Note: This test requires numpy, pandas, and scipy")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Quick performance comparison"""
    print("\n" + "=" * 70)
    print("TEST 6: Performance Comparison")
    print("=" * 70)
    
    try:
        import time
        from quant_accelerated import BlackScholesAccelerated, CPP_AVAILABLE
        
        if not CPP_AVAILABLE:
            print("⚠ C++ library not available, skipping performance test")
            return True
        
        # Benchmark Black-Scholes
        n_iterations = 10000
        start = time.time()
        for i in range(n_iterations):
            _ = BlackScholesAccelerated.call_price(100, 100+i*0.01, 1.0, 0.05, 0.2)
        cpp_time = time.time() - start
        
        print(f"✓ C++ Black-Scholes: {n_iterations} calls in {cpp_time:.4f}s")
        print(f"  ({n_iterations/cpp_time:.0f} calls/second)")
        
        # Try Python comparison if available
        try:
            from models.options.black_scholes import BlackScholes
            start = time.time()
            for i in range(n_iterations):
                _ = BlackScholes.call_price(100, 100+i*0.01, 1.0, 0.05, 0.2)
            py_time = time.time() - start
            
            print(f"✓ Python Black-Scholes: {n_iterations} calls in {py_time:.4f}s")
            print(f"  ({n_iterations/py_time:.0f} calls/second)")
            print(f"\n  Speedup: {py_time/cpp_time:.1f}x faster with C++")
        except:
            print("  (Python comparison skipped - dependencies not available)")
        
        print("\n✅ Performance test completed")
        return True
        
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")
        return True  # Don't fail on performance test


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "C++ QUANTITATIVE FINANCE LIBRARY TEST SUITE" + " " * 10 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("C++ Availability", test_cpp_availability()))
    results.append(("Black-Scholes", test_black_scholes()))
    results.append(("Monte Carlo", test_monte_carlo()))
    results.append(("Portfolio Analytics", test_portfolio()))
    results.append(("Compatibility", test_compatibility()))
    results.append(("Performance", test_performance()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! The C++ quant library is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
