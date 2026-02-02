"""
Unit tests for models/risk/var_cvar.py.

Fixed return series -> known percentile (historical VaR 95% = 5th percentile)
and tail expectation (CVaR).
"""

import numpy as np
import pandas as pd
import pytest
from models.risk.var_cvar import VaRModel, CVaRModel


@pytest.fixture
def fixed_returns():
    """Fixed return series: 100 points, known distribution for VaR/CVaR."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.01)


def test_var_historical_95_is_5th_percentile(fixed_returns):
    """Historical VaR at 95% (confidence_level=0.05) equals 5th percentile of returns."""
    var = VaRModel.calculate_var(fixed_returns, method="historical", confidence_level=0.05)
    expected = fixed_returns.quantile(0.05)
    assert abs(var - expected) < 1e-10


def test_var_historical_99_is_1st_percentile(fixed_returns):
    """Historical VaR at 99% (confidence_level=0.01) equals 1st percentile."""
    var = VaRModel.calculate_var(fixed_returns, method="historical", confidence_level=0.01)
    expected = fixed_returns.quantile(0.01)
    assert abs(var - expected) < 1e-10


def test_cvar_historical_is_tail_mean(fixed_returns):
    """Historical CVaR at 95% is mean of returns <= VaR."""
    var_95 = VaRModel.historical_var(fixed_returns, confidence_level=0.05)
    cvar = CVaRModel.calculate_cvar(fixed_returns, method="historical", confidence_level=0.05)
    tail = fixed_returns[fixed_returns <= var_95]
    expected = tail.mean() if len(tail) > 0 else var_95
    assert abs(cvar - expected) < 1e-10


def test_cvar_historical_less_than_or_equal_var(fixed_returns):
    """CVaR (expected shortfall) should be <= VaR for left tail."""
    var_95 = VaRModel.calculate_var(fixed_returns, method="historical", confidence_level=0.05)
    cvar_95 = CVaRModel.calculate_cvar(fixed_returns, method="historical", confidence_level=0.05)
    assert cvar_95 <= var_95 + 1e-10
