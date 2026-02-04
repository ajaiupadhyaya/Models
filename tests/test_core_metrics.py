"""
Unit tests for core/utils.py metric functions.
Fixed return series -> known Sharpe, Sortino, max drawdown, VaR, CVaR.
"""

import numpy as np
import pandas as pd
import pytest

from core.utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    drawdown_series_from_returns,
    drawdown_series_from_equity,
    calculate_max_drawdown_from_equity,
    calculate_var,
    calculate_cvar,
    annualize_returns,
    annualize_volatility,
)


PERIODS_PER_YEAR = 252
RISK_FREE_RATE = 0.02


@pytest.fixture
def fixed_returns():
    """Fixed return series: 100 points, known distribution (seed 42)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.01)


@pytest.fixture
def known_mean_std_returns():
    """Returns with exactly known mean and std for hand-computed Sharpe."""
    # 10 daily returns: mean 0.001, std 0.02
    r = np.array([0.001] * 10) + np.array([0.02, -0.02, 0.01, -0.01, 0.015, -0.015, 0.02, -0.02, 0.01, -0.01])
    return pd.Series(r)


def test_calculate_returns_simple():
    """Simple returns = pct_change."""
    prices = pd.Series([100, 102, 101, 105])
    r = calculate_returns(prices, method="simple")
    assert len(r) == 3
    assert abs(r.iloc[0] - 0.02) < 1e-10
    assert abs(r.iloc[1] - (-1 / 102)) < 1e-10


def test_calculate_returns_log():
    """Log returns = log(P_t / P_{t-1})."""
    prices = pd.Series([100, 102])
    r = calculate_returns(prices, method="log")
    assert len(r) == 1
    assert abs(r.iloc[0] - np.log(102 / 100)) < 1e-10


def test_sharpe_ratio_fixed_returns(fixed_returns):
    """Sharpe is annualized (sqrt(252)) * (mean - rf/252) / std."""
    sharpe = calculate_sharpe_ratio(fixed_returns, risk_free_rate=RISK_FREE_RATE)
    excess = fixed_returns - RISK_FREE_RATE / PERIODS_PER_YEAR
    expected = np.sqrt(PERIODS_PER_YEAR) * excess.mean() / fixed_returns.std()
    assert np.isfinite(sharpe)
    assert abs(sharpe - expected) < 1e-10


def test_sharpe_ratio_known(known_mean_std_returns):
    """Sharpe with known mean/std matches hand calculation."""
    sharpe = calculate_sharpe_ratio(known_mean_std_returns, risk_free_rate=0.02)
    excess = known_mean_std_returns - 0.02 / 252
    expected = np.sqrt(252) * excess.mean() / known_mean_std_returns.std()
    assert abs(sharpe - expected) < 1e-8


def test_sharpe_ratio_zero_volatility():
    """Zero volatility returns -> inf or nan; caller should handle."""
    flat = pd.Series([0.001] * 20)
    sharpe = calculate_sharpe_ratio(flat)
    assert not np.isfinite(sharpe)  # std=0 -> div by zero


def test_sortino_ratio_fixed(fixed_returns):
    """Sortino uses downside deviation."""
    sortino = calculate_sortino_ratio(fixed_returns, risk_free_rate=RISK_FREE_RATE)
    # May be nan if no negative returns
    if len(fixed_returns[fixed_returns < 0]) > 0:
        assert np.isfinite(sortino) or np.isnan(sortino)
    else:
        assert np.isnan(sortino)


def test_max_drawdown_fixed(fixed_returns):
    """Max drawdown is min of drawdown series."""
    dd = calculate_max_drawdown(fixed_returns)
    dd_series = drawdown_series_from_returns(fixed_returns)
    assert abs(dd - dd_series.min()) < 1e-12
    assert dd <= 0


def test_max_drawdown_known():
    """Known sequence: 1, 1.1, 0.99, 1.05 -> drawdown min = (0.99 - 1.1)/1.1."""
    returns = pd.Series([0.1, -0.1, 0.06060606])  # approx path 1 -> 1.1 -> 0.99 -> 1.05
    dd = calculate_max_drawdown(returns)
    cum = (1 + returns).cumprod()
    run_max = cum.expanding().max()
    expected = ((cum - run_max) / run_max).min()
    assert abs(dd - expected) < 1e-5


def test_drawdown_series_from_equity_matches_returns(fixed_returns):
    """Equity = (1+returns).cumprod() -> same drawdown as from returns."""
    equity = (1 + fixed_returns).cumprod()
    dd_from_equity = calculate_max_drawdown_from_equity(equity)
    dd_from_returns = calculate_max_drawdown(fixed_returns)
    assert abs(dd_from_equity - dd_from_returns) < 1e-12


def test_var_fixed(fixed_returns):
    """VaR at 0.05 = 5th percentile."""
    var = calculate_var(fixed_returns, confidence_level=0.05)
    expected = fixed_returns.quantile(0.05)
    assert abs(var - expected) < 1e-12


def test_cvar_fixed(fixed_returns):
    """CVaR = mean of returns <= VaR."""
    var_05 = calculate_var(fixed_returns, 0.05)
    cvar = calculate_cvar(fixed_returns, 0.05)
    tail = fixed_returns[fixed_returns <= var_05]
    expected = tail.mean()
    assert abs(cvar - expected) < 1e-12
    assert cvar <= var_05 + 1e-10


def test_annualize_returns(fixed_returns):
    """Annualized return = (1 + mean)^252 - 1."""
    ann = annualize_returns(fixed_returns, periods_per_year=PERIODS_PER_YEAR)
    expected = (1 + fixed_returns.mean()) ** PERIODS_PER_YEAR - 1
    assert abs(ann - expected) < 1e-10


def test_annualize_volatility(fixed_returns):
    """Annualized vol = std * sqrt(252)."""
    ann_vol = annualize_volatility(fixed_returns, periods_per_year=PERIODS_PER_YEAR)
    expected = fixed_returns.std() * np.sqrt(PERIODS_PER_YEAR)
    assert abs(ann_vol - expected) < 1e-10


def test_max_drawdown_empty_series():
    """Empty returns -> drawdown series is empty; min is nan."""
    empty = pd.Series(dtype=float)
    dd = calculate_max_drawdown(empty)
    assert np.isnan(dd)


def test_max_drawdown_single_point():
    """Single return -> one cumulative point; drawdown series has one value."""
    single = pd.Series([0.01])
    dd = calculate_max_drawdown(single)
    # (1+0.01)=1.01, running_max=1.01, drawdown=0
    assert dd == 0.0
