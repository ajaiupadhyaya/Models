"""
Shared request contracts and mapping helpers for backtest APIs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LegacyModelBacktestRequest(BaseModel):
    """Legacy /api/v1/backtest/run request model."""

    model_name: str = Field(description="Name of the model to backtest")
    symbol: str = Field(description="Stock symbol", example="SPY")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100000.0, description="Starting capital")
    commission: float = Field(default=0.001, description="Commission rate")
    position_size: float = Field(default=1.0, description="Position size (0-1)")
    use_institutional: Optional[bool] = Field(
        default=None,
        description="Deprecated; kept for compatibility",
    )


class StrategyBacktestRequest(BaseModel):
    """Strategy-driven request model used by /api/v1/quant/backtest."""

    symbol: str = "AAPL"
    strategy: str = Field("sma_cross", description="sma_cross, rsi_mean_reversion, factor_momentum")
    start_date: str = "2019-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 100000.0
    commission: float = 0.001
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    rsi_period: Optional[int] = None
    oversold: Optional[int] = None
    overbought: Optional[int] = None
    lookback: Optional[int] = None


def strategy_from_model_name(model_name: str) -> str:
    """Map legacy model names to supported strategy backtests."""
    name = (model_name or "").lower()
    if "rsi" in name:
        return "rsi_mean_reversion"
    if "factor" in name or "momentum" in name:
        return "factor_momentum"
    return "sma_cross"


def strategy_params_from_request(req: StrategyBacktestRequest) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if req.fast_period is not None:
        params["fast_period"] = req.fast_period
    if req.slow_period is not None:
        params["slow_period"] = req.slow_period
    if req.rsi_period is not None:
        params["rsi_period"] = req.rsi_period
    if req.oversold is not None:
        params["oversold"] = req.oversold
    if req.overbought is not None:
        params["overbought"] = req.overbought
    if req.lookback is not None:
        params["lookback"] = req.lookback
    return params
