"""
Unit tests for fundamental metrics: Altman Z-Score, Piotroski F-Score, key ratios.
Fixed inputs -> expected outputs from literature or hand calculation.
"""

import numpy as np
import pandas as pd
import pytest

from models.fundamental.company_analyzer import FundamentalMetrics
from models.fundamental.ratios import FinancialRatios


def test_altman_z_score_known_inputs():
    """Altman Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 with known financials."""
    financials = {
        "working_capital": 100,
        "retained_earnings": 200,
        "ebit": 50,
        "market_cap": 500,
        "total_liabilities": 200,
        "revenue": 800,
        "total_assets": 1000,
    }
    z = FundamentalMetrics.altman_z_score(financials)
    x1 = 100 / 1000
    x2 = 200 / 1000
    x3 = 50 / 1000
    x4 = 500 / 200
    x5 = 800 / 1000
    expected = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    assert abs(z - expected) < 1e-10
    assert z > 2.0  # safe zone > 2.99; this is in gray area


def test_altman_z_score_distress_zone():
    """Low Z-Score indicates distress (< 1.81)."""
    financials = {
        "working_capital": -50,
        "retained_earnings": -100,
        "ebit": -20,
        "market_cap": 50,
        "total_liabilities": 500,
        "revenue": 200,
        "total_assets": 400,
    }
    z = FundamentalMetrics.altman_z_score(financials)
    assert z < 2.0


def test_piotroski_f_score_all_positive():
    """All 9 criteria positive -> F-Score = 9."""
    financials = {
        "net_income": 1,
        "roa": 0.01,
        "operating_cash_flow": 10,
        "debt_to_equity_change": -0.1,
        "current_ratio_change": 0.1,
        "shares_outstanding_change": 0,
        "gross_margin_change": 0.01,
        "asset_turnover_change": 0.01,
    }
    # operating_cash_flow > net_income -> +1
    score = FundamentalMetrics.piotroski_f_score(financials)
    assert score >= 7  # at least 7 with all positive


def test_piotroski_f_score_all_negative():
    """All criteria negative -> F-Score = 0."""
    financials = {
        "net_income": -1,
        "roa": -0.01,
        "operating_cash_flow": -1,
        "debt_to_equity_change": 1,
        "current_ratio_change": -0.1,
        "shares_outstanding_change": 1,
        "gross_margin_change": -0.01,
        "asset_turnover_change": -0.01,
    }
    score = FundamentalMetrics.piotroski_f_score(financials)
    assert score == 0


def test_piotroski_f_score_bounds():
    """F-Score is always 0-9."""
    for _ in range(5):
        financials = {k: np.random.randn() for k in [
            "net_income", "roa", "operating_cash_flow",
            "debt_to_equity_change", "current_ratio_change",
            "shares_outstanding_change", "gross_margin_change", "asset_turnover_change"
        ]}
        score = FundamentalMetrics.piotroski_f_score(financials)
        assert 0 <= score <= 9


def test_liquidity_ratios_structure():
    """FinancialRatios.liquidity_ratios returns dict with current_ratio, quick_ratio."""
    # Rows = line items, columns = periods (latest = first column)
    bs = pd.DataFrame(
        {"Q1": [200, 100, 50, 80]},
        index=["Current Assets", "Current Liabilities", "Inventory", "Cash And Cash Equivalents"],
    )
    result = FinancialRatios.liquidity_ratios(bs)
    assert isinstance(result, dict)
    assert "current_ratio" in result
    assert "quick_ratio" in result


def test_liquidity_ratios_current_ratio():
    """Current ratio = current_assets / current_liabilities."""
    bs = pd.DataFrame(
        {"Q1": [1000, 500, 200, 100]},
        index=["Current Assets", "Current Liabilities", "Inventory", "Cash And Cash Equivalents"],
    )
    result = FinancialRatios.liquidity_ratios(bs)
    assert result["current_ratio"] == 1000 / 500
    assert result["quick_ratio"] == (1000 - 200) / 500
