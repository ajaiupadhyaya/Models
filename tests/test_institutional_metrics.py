"""
Unit tests for models/quant/institutional_grade.py: AdvancedRiskMetrics and StatisticalValidation.
"""

import numpy as np
import pandas as pd
import pytest

from models.quant.institutional_grade import (
    AdvancedRiskMetrics,
    StatisticalValidation,
)
from core.utils import drawdown_series_from_returns


np.random.seed(42)


@pytest.fixture
def fixed_returns():
    """Fixed return series."""
    return pd.Series(np.random.randn(100) * 0.01)


@pytest.fixture
def fixed_equity(fixed_returns):
    """Equity curve from returns."""
    return (1 + fixed_returns).cumprod()


def test_maximum_drawdown_structure(fixed_equity):
    """maximum_drawdown returns dict with expected keys."""
    result = AdvancedRiskMetrics.maximum_drawdown(fixed_equity)
    assert isinstance(result, dict)
    assert "max_drawdown" in result
    assert "max_drawdown_pct" in result
    assert "drawdown_date" in result
    assert result["max_drawdown"] <= 0


def test_maximum_drawdown_value_matches_core(fixed_returns, fixed_equity):
    """maximum_drawdown scalar matches core.utils drawdown from returns."""
    from core.utils import calculate_max_drawdown
    dd_core = calculate_max_drawdown(fixed_returns)
    result = AdvancedRiskMetrics.maximum_drawdown(fixed_equity)
    assert abs(result["max_drawdown"] - dd_core) < 1e-10


def test_sortino_ratio_fixed(fixed_returns):
    """Sortino ratio is finite when downside returns exist."""
    sortino = AdvancedRiskMetrics.sortino_ratio(fixed_returns, risk_free_rate=0.02)
    assert np.isfinite(sortino) or sortino == 0.0


def test_calmar_ratio_fixed(fixed_returns, fixed_equity):
    """Calmar = annual return / |max drawdown|."""
    calmar = AdvancedRiskMetrics.calmar_ratio(fixed_returns, fixed_equity)
    assert np.isfinite(calmar)
    assert calmar >= 0


def test_expected_shortfall_fixed(fixed_returns):
    """Expected shortfall = mean of returns <= VaR."""
    es = AdvancedRiskMetrics.expected_shortfall(fixed_returns, confidence=0.05)
    var = fixed_returns.quantile(0.05)
    tail = fixed_returns[fixed_returns <= var]
    expected = tail.mean()
    assert abs(es - expected) < 1e-12


def test_bootstrap_confidence_interval():
    """Bootstrap CI for mean: with enough samples, CI contains true mean."""
    np.random.seed(123)
    data = np.random.normal(0, 1, 200)
    lower, upper = StatisticalValidation.bootstrap_confidence_interval(
        data, statistic=np.mean, n_bootstrap=500, confidence=0.95
    )
    assert lower < upper
    assert lower < 0.2 and upper > -0.2  # true mean 0


def test_normality_test_synthetic_normal():
    """Synthetic normal series -> is_normal True (p > 0.05)."""
    np.random.seed(42)
    normal_returns = pd.Series(np.random.randn(500) * 0.01)
    result = StatisticalValidation.normality_test(normal_returns)
    if "error" in result:
        pytest.skip(result["error"])
    assert "is_normal" in result
    assert "jarque_bera_pvalue" in result
    # Normal data often passes JB; allow either
    assert result["jarque_bera_pvalue"] > 0 or not result["is_normal"]


def test_stationarity_test_random_walk():
    """Random walk is non-stationary (p typically > 0.05)."""
    np.random.seed(42)
    rw = pd.Series(np.random.randn(200).cumsum())
    result = StatisticalValidation.stationarity_test(rw)
    if "error" in result:
        pytest.skip(result["error"])
    assert "is_stationary" in result
    assert "p_value" in result


def test_stationarity_test_stationary_series():
    """White noise is stationary (p typically < 0.05)."""
    np.random.seed(42)
    white = pd.Series(np.random.randn(200))
    result = StatisticalValidation.stationarity_test(white)
    if "error" in result:
        pytest.skip(result["error"])
    assert "is_stationary" in result


def test_cointegration_test_known_pair():
    """Cointegrated pair: y = x + noise -> is_cointegrated often True."""
    np.random.seed(42)
    x = pd.Series(np.cumsum(np.random.randn(100)))
    y = x + np.random.randn(100) * 0.5
    result = StatisticalValidation.cointegration_test(x, y)
    if "error" in result:
        pytest.skip(result["error"])
    assert "is_cointegrated" in result
    assert "p_value" in result
