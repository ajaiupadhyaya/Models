"""
Pytest tests for C++ quantitative finance library.
Skips gracefully if C++ extension is not built (e.g. CI without build step).
"""

import numpy as np
import pytest

try:
    from quant_accelerated import CPP_AVAILABLE, BlackScholesAccelerated
except ImportError:
    CPP_AVAILABLE = False
    BlackScholesAccelerated = None


def _cpp_available():
    try:
        from quant_accelerated import CPP_AVAILABLE
        return CPP_AVAILABLE
    except ImportError:
        return False


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_cpp_availability():
    """C++ library is available when built."""
    from quant_accelerated import CPP_AVAILABLE
    assert CPP_AVAILABLE is True


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_black_scholes_call_price():
    """Black-Scholes call price in expected range for ATM option."""
    from quant_accelerated import BlackScholesAccelerated
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call_price = BlackScholesAccelerated.call_price(S, K, T, r, sigma)
    assert 10.0 < call_price < 11.0, f"Call price {call_price} out of expected range"


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_black_scholes_put_price():
    """Black-Scholes put price is positive."""
    from quant_accelerated import BlackScholesAccelerated
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    put_price = BlackScholesAccelerated.put_price(S, K, T, r, sigma)
    assert put_price > 0


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_black_scholes_delta():
    """Delta for ATM call in (0.5, 0.7)."""
    from quant_accelerated import BlackScholesAccelerated
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    delta = BlackScholesAccelerated.delta(S, K, T, r, sigma, is_call=True)
    assert 0.5 < delta < 0.7


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_black_scholes_gamma_vega_positive():
    """Gamma and vega are positive."""
    from quant_accelerated import BlackScholesAccelerated
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    gamma = BlackScholesAccelerated.gamma(S, K, T, r, sigma)
    vega = BlackScholesAccelerated.vega(S, K, T, r, sigma)
    assert gamma > 0
    assert vega > 0


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_monte_carlo_european_option():
    """Monte Carlo European call price near analytical."""
    from quant_accelerated import MonteCarloAccelerated
    mc = MonteCarloAccelerated(seed=42)
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    price = mc.price_european_option(S0, K, T, r, sigma, is_call=True, n_simulations=5000)
    assert 9.0 < price < 12.0


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_monte_carlo_gbm_path_length():
    """GBM path has steps+1 points."""
    from quant_accelerated import MonteCarloAccelerated
    mc = MonteCarloAccelerated(seed=42)
    path = mc.simulate_gbm_path(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252)
    assert len(path) == 253


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_portfolio_return_volatility():
    """Portfolio return and volatility are in expected range."""
    from quant_accelerated import PortfolioAccelerated
    weights = [0.3, 0.3, 0.4]
    expected_returns = [0.10, 0.12, 0.08]
    cov_matrix = [[0.04, 0.01, 0.02], [0.01, 0.06, 0.015], [0.02, 0.015, 0.05]]
    port_return = PortfolioAccelerated.portfolio_return(weights, expected_returns)
    port_vol = PortfolioAccelerated.portfolio_volatility(weights, cov_matrix)
    assert 0.08 < port_return < 0.12
    assert port_vol > 0


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_portfolio_sharpe_positive():
    """Sharpe ratio positive for positive expected return portfolio."""
    from quant_accelerated import PortfolioAccelerated
    weights = [0.3, 0.3, 0.4]
    expected_returns = [0.10, 0.12, 0.08]
    cov_matrix = [[0.04, 0.01, 0.02], [0.01, 0.06, 0.015], [0.02, 0.015, 0.05]]
    sharpe = PortfolioAccelerated.sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.03)
    assert sharpe > 0


@pytest.mark.skipif(not _cpp_available(), reason="C++ quant extension not built")
def test_portfolio_max_drawdown_bounds():
    """Max drawdown from cumulative returns in [0, 1]."""
    from quant_accelerated import PortfolioAccelerated
    cumulative_returns = [1.0, 1.1, 1.05, 1.15, 1.08, 1.20]
    max_dd = PortfolioAccelerated.max_drawdown(cumulative_returns)
    assert 0 <= max_dd <= 1


def test_python_black_scholes_still_works():
    """Original Python Black-Scholes implementation still works (compatibility)."""
    from models.options.black_scholes import BlackScholes
    call_price = BlackScholes.call_price(100, 100, 1.0, 0.05, 0.2)
    assert 10.0 < call_price < 11.0
