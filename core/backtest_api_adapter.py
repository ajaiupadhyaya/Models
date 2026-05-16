"""
Shared API adapter for backtest endpoints.

Provides one normalization path so different API routes expose a compatible
backtest response contract while using the same core service implementation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def run_backtest_contract(
    *,
    symbol: str,
    strategy: str,
    start_date: str,
    end_date: Optional[str],
    initial_capital: float,
    commission: float,
    model_name: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    equity_points_limit: Optional[int] = None,
) -> Dict[str, Any]:
    from core.backtest_service import run_backtest

    resolved_end = end_date or datetime.now().strftime("%Y-%m-%d")
    params = strategy_params or {}
    result = run_backtest(
        symbol=symbol.upper(),
        strategy=strategy,
        start_date=start_date,
        end_date=resolved_end,
        initial_capital=initial_capital,
        commission=commission,
        **params,
    )
    if "error" in result:
        raise ValueError(result["error"])

    equity_curve = result.get("equity_curve", [])
    if equity_points_limit is not None:
        equity_curve = equity_curve[-equity_points_limit:]

    return {
        "model_name": model_name or strategy,
        "symbol": result.get("symbol", symbol.upper()),
        "strategy": result.get("strategy", strategy),
        "start_date": start_date,
        "end_date": resolved_end,
        "period": {"start": start_date, "end": resolved_end},
        "metrics": result.get("metrics", {}),
        "equity_curve": equity_curve,
        "trades": result.get("trades", []),
        "status": "success",
    }
