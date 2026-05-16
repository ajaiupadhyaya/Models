"""
Portfolio optimization using real historical returns from DB.
Mean-variance optimization, efficient frontier.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_returns(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch aligned daily returns through canonical market data facade."""
    from core.market_data_facade import fetch_returns_matrix

    return fetch_returns_matrix(tickers, start_date, end_date, min_series=2)


def run_optimization(
    tickers: List[str],
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Mean-variance optimization from real returns.
    Returns optimal weights, expected return, volatility, Sharpe, efficient frontier points.
    """
    from models.portfolio.optimization import MeanVarianceOptimizer

    tickers = [t.upper() for t in tickers]
    if len(tickers) < 2:
        return {"error": "Provide at least 2 tickers"}

    returns = _fetch_returns(tickers, start_date, end_date)
    if returns.empty or len(returns) < 20:
        return {"error": "Insufficient return data from DB"}

    returns = returns.reindex(columns=[s for s in tickers if s in returns.columns]).dropna(axis=1, how="all")
    if returns.shape[1] < 2:
        return {"error": "Insufficient series after alignment"}

    exp_ret = returns.mean()
    cov = returns.cov()
    opt = MeanVarianceOptimizer(exp_ret, cov, risk_free_rate)

    sharpe_result = opt.optimize_sharpe()
    min_vol_result = opt.optimize_min_volatility()

    weights = {k: round(float(v), 4) for k, v in sharpe_result["weights"].items()}
    exp_ret_val = float(sharpe_result["expected_return"])
    vol_val = float(sharpe_result["volatility"])
    sharpe_val = float(sharpe_result["sharpe_ratio"])

    # Efficient frontier: sweep target returns from min vol to max return
    min_ret = float(exp_ret.min())
    max_ret = float(exp_ret.max())
    n_pts = 25
    frontier_ret = np.linspace(min_ret, max_ret, n_pts)
    frontier_vol = []
    frontier_sharpe = []
    for r in frontier_ret:
        try:
            res = opt.optimize_target_return(r)
            v = float(res["volatility"])
            frontier_vol.append(v)
            s = (r - risk_free_rate) / v if v > 0 else 0
            frontier_sharpe.append(s)
        except Exception:
            frontier_vol.append(np.nan)
            frontier_sharpe.append(np.nan)

    frontier = [
        {"return": round(float(r), 6), "volatility": round(float(v), 6), "sharpe": round(float(s), 4)}
        for r, v, s in zip(frontier_ret, frontier_vol, frontier_sharpe)
        if not np.isnan(v)
    ]

    return {
        "tickers": list(weights.keys()),
        "weights": weights,
        "expected_return": round(exp_ret_val, 4),
        "volatility": round(vol_val, 4),
        "sharpe_ratio": round(sharpe_val, 4),
        "efficient_frontier": frontier,
        "period": {"start": start_date, "end": end_date},
    }
