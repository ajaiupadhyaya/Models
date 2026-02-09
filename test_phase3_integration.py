"""
Phase 3 Integration Tests - Awesome Quant Integration
Tests: Options Pricing, Black-Scholes, Greeks, RL Trading Environment
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 3 modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Validating Phase 3 Imports")
    print("="*60)
    
    try:
        from models.derivatives.option_pricing import BlackScholes, GreeksCalculator, ImpliedVolatility, OptionAnalyzer
        print("✓ Options pricing modules imported successfully")
        
        from models.rl.deep_rl_trading import TradingEnvironment, RLTrader
        print("✓ Deep RL trading modules imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_black_scholes():
    """Test Black-Scholes option pricing."""
    print("\n" + "="*60)
    print("TEST 2: Black-Scholes Option Pricing")
    print("="*60)
    
    try:
        from models.derivatives.option_pricing import BlackScholes
        
        # Parameters
        S = 100.0  # Spot price
        K = 100.0  # Strike (ATM)
        T = 0.25   # 3 months
        r = 0.05   # 5% risk-free
        sigma = 0.2  # 20% vol
        
        # Price call and put
        call_price = BlackScholes.call_price(S, K, T, r, sigma)
        put_price = BlackScholes.put_price(S, K, T, r, sigma)
        
        print(f"ATM Call Price: ${call_price:.4f}")
        print(f"ATM Put Price: ${put_price:.4f}")
        
        # Put-call parity check: C - P = S - K*e^(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = S - K * np.exp(-r * T)
        parity_diff = abs(parity_lhs - parity_rhs)
        
        print(f"\nPut-Call Parity Check:")
        print(f"  C - P = {parity_lhs:.4f}")
        print(f"  S - K*e^(-rT) = {parity_rhs:.4f}")
        print(f"  Difference: {parity_diff:.6f}")
        
        # Validate
        assert call_price > 0, "Call price must be positive"
        assert put_price > 0, "Put price must be positive"
        assert parity_diff < 0.0001, f"Put-call parity violated: {parity_diff}"
        
        # Test ITM/OTM
        call_itm = BlackScholes.call_price(110, 100, T, r, sigma)  # ITM
        call_otm = BlackScholes.call_price(90, 100, T, r, sigma)   # OTM
        
        print(f"\nITM Call (S=110, K=100): ${call_itm:.4f}")
        print(f"OTM Call (S=90, K=100): ${call_otm:.4f}")
        
        assert call_itm > call_price, "ITM call should be more expensive"
        assert call_otm < call_price, "OTM call should be cheaper"
        
        print("✓ Black-Scholes test passed")
        return True
    except Exception as e:
        print(f"✗ Black-Scholes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_greeks():
    """Test Greeks calculations."""
    print("\n" + "="*60)
    print("TEST 3: Greeks Calculations")
    print("="*60)
    
    try:
        from models.derivatives.option_pricing import GreeksCalculator
        
        # Parameters
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        sigma = 0.2
        
        # Calculate Greeks
        delta_call = GreeksCalculator.call_delta(S, K, T, r, sigma)
        delta_put = GreeksCalculator.put_delta(S, K, T, r, sigma)
        gamma = GreeksCalculator.gamma(S, K, T, r, sigma)
        vega = GreeksCalculator.vega(S, K, T, r, sigma)
        theta_call = GreeksCalculator.call_theta(S, K, T, r, sigma)
        rho_call = GreeksCalculator.call_rho(S, K, T, r, sigma)
        
        print(f"Delta (Call): {delta_call:.4f}")
        print(f"Delta (Put): {delta_put:.4f}")
        print(f"Gamma: {gamma:.6f}")
        print(f"Vega: {vega:.4f}")
        print(f"Theta (Call): {theta_call:.4f} per day")
        print(f"Rho (Call): {rho_call:.4f}")
        
        # Validate ranges
        assert 0 <= delta_call <= 1, "Call delta should be in [0, 1]"
        assert -1 <= delta_put <= 0, "Put delta should be in [-1, 0]"
        assert gamma >= 0, "Gamma should be non-negative"
        assert vega >= 0, "Vega should be non-negative"
        assert theta_call < 0, "Call theta should be negative (time decay)"
        
        # Delta-gamma relationship: put delta = call delta - 1
        delta_diff = abs(delta_put - (delta_call - 1))
        print(f"\nPut-Call Delta relationship: |put_delta - (call_delta - 1)| = {delta_diff:.6f}")
        assert delta_diff < 0.01, "Put-call delta relationship violated"
        
        print("✓ Greeks test passed")
        return True
    except Exception as e:
        print(f"✗ Greeks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_implied_volatility():
    """Test implied volatility calculation."""
    print("\n" + "="*60)
    print("TEST 4: Implied Volatility")
    print("="*60)
    
    try:
        from models.derivatives.option_pricing import BlackScholes, ImpliedVolatility
        
        # Parameters
        S = 100.0
        K = 100.0
        T = 0.25
        r = 0.05
        true_sigma = 0.25  # 25% volatility
        
        # Calculate option price with known volatility
        market_price = BlackScholes.call_price(S, K, T, r, true_sigma)
        print(f"Market price (with σ={true_sigma}): ${market_price:.4f}")
        
        # Back out implied volatility
        implied_vol = ImpliedVolatility.call_iv(market_price, S, K, T, r)
        
        print(f"True volatility: {true_sigma:.4f}")
        print(f"Implied volatility: {implied_vol:.4f}")
        print(f"Difference: {abs(implied_vol - true_sigma):.6f}")
        
        # Validate
        assert implied_vol is not None, "IV calculation failed"
        assert abs(implied_vol - true_sigma) < 0.0001, f"IV mismatch: {abs(implied_vol - true_sigma)}"
        
        print("✓ Implied volatility test passed")
        return True
    except Exception as e:
        print(f"✗ Implied volatility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option_analyzer():
    """Test comprehensive option analyzer."""
    print("\n" + "="*60)
    print("TEST 5: Option Analyzer")
    print("="*60)
    
    try:
        from models.derivatives.option_pricing import OptionAnalyzer
        
        # Analyze ATM call
        analysis = OptionAnalyzer.analyze_option(
            option_type="call",
            S=100.0,
            K=100.0,
            T=0.25,
            r=0.05,
            sigma=0.20
        )
        
        print("ATM Call Analysis:")
        print(f"  Price: ${analysis['price']}")
        print(f"  Delta: {analysis['delta']}")
        print(f"  Gamma: {analysis['gamma']}")
        print(f"  Vega: {analysis['vega']}")
        print(f"  Theta: {analysis['theta']}")
        print(f"  Intrinsic: ${analysis['intrinsic_value']}")
        print(f"  Time Value: ${analysis['time_value']}")
        print(f"  Moneyness: {analysis['moneyness_type']}")
        
        # Validate
        assert 'price' in analysis, "Missing price"
        assert 'delta' in analysis, "Missing delta"
        assert analysis['moneyness_type'] == 'ATM', "Should be ATM"
        assert analysis['time_value'] > 0, "Time value should be positive"
        
        print("✓ Option analyzer test passed")
        return True
    except Exception as e:
        print(f"✗ Option analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trading_environment():
    """Test RL trading environment."""
    print("\n" + "="*60)
    print("TEST 6: RL Trading Environment")
    print("="*60)
    
    try:
        from models.rl.deep_rl_trading import TradingEnvironment
        
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200)))
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
            'High': prices * (1 + np.random.uniform(0, 0.02, 200)),
            'Low': prices * (1 + np.random.uniform(-0.02, 0, 200)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        # Create environment
        env = TradingEnvironment(df, initial_balance=10000.0)
        
        print(f"Environment created successfully")
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.n} actions")
        
        # Reset environment
        obs, info = env.reset()
        print(f"\nInitial observation shape: {obs.shape}")
        print(f"Portfolio value: ${env.portfolio_value:.2f}")
        
        # Take random actions
        actions_taken = []
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            actions_taken.append(info['action_taken'])
            
            if terminated or truncated:
                break
        
        print(f"\nTook {len(actions_taken)} steps")
        print(f"Actions: {actions_taken}")
        print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Return: {(info['portfolio_value'] - 10000) / 10000 * 100:.2f}%")
        
        # Validate
        assert obs.shape[0] == env.observation_space.shape[0], "Observation shape mismatch"
        assert env.action_space.n == 3, "Should have 3 actions (sell, hold, buy)"
        
        print("✓ Trading environment test passed")
        return True
    except Exception as e:
        print(f"✗ Trading environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_trader_initialization():
    """Test RL trader initialization (without training)."""
    print("\n" + "="*60)
    print("TEST 7: RL Trader Initialization")
    print("="*60)
    
    try:
        from models.rl.deep_rl_trading import RLTrader, TradingEnvironment
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Initialize trader
        trader = RLTrader(algorithm="PPO", learning_rate=3e-4)
        print("✓ RLTrader initialized successfully")
        
        # Create environment
        env = trader.create_environment(df, initial_balance=10000.0)
        print("✓ Trading environment created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n}")
        
        print("✓ RL trader initialization test passed")
        print("  (Note: Training test skipped - requires ~1-2 min)")
        return True
    except Exception as e:
        print(f"✗ RL trader initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 tests."""
    print("\n" + "="*60)
    print("PHASE 3 AWESOME QUANT INTEGRATION - TEST SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("Imports", test_imports),
        ("Black-Scholes", test_black_scholes),
        ("Greeks", test_greeks),
        ("Implied Volatility", test_implied_volatility),
        ("Option Analyzer", test_option_analyzer),
        ("Trading Environment", test_trading_environment),
        ("RL Trader Init", test_rl_trader_initialization)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
