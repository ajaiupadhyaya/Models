"""
Risk API

Endpoints for VaR, CVaR, volatility, and stress metrics per symbol.
Used by the terminal Portfolio panel for risk analytics.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics/{ticker}")
async def get_risk_metrics(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Get risk metrics for a symbol: VaR (95%, 99%), CVaR, volatility, max drawdown.
    Used by the terminal Portfolio / risk view.
    """
    try:
        from core.data_fetcher import DataFetcher
        from models.risk.var_cvar import VaRModel, CVaRModel
        import pandas as pd
        import numpy as np

        fetcher = DataFetcher()
        data = fetcher.get_stock_data(ticker.upper(), period=period)
        if data is None or data.empty or "Close" not in data.columns:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        returns = data["Close"].pct_change().dropna()
        if len(returns) < 20:
            raise HTTPException(status_code=400, detail="Insufficient data for risk metrics")

        # VaR/CVaR: use left tail (confidence_level = tail probability)
        # 95% VaR = 5th percentile; 99% VaR = 1st percentile
        var_95 = float(VaRModel.calculate_var(returns, confidence_level=0.05))
        var_99 = float(VaRModel.calculate_var(returns, confidence_level=0.01))
        cvar_95 = float(CVaRModel.calculate_cvar(returns, confidence_level=0.05))
        cvar_99 = float(CVaRModel.calculate_cvar(returns, confidence_level=0.01))

        volatility_daily = float(returns.std())
        volatility_annual = float(volatility_daily * (252 ** 0.5))
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())

        rf = 0.02
        sharpe = float((returns.mean() * 252 - rf) / volatility_annual) if volatility_annual > 0 else 0.0

        return {
            "ticker": ticker.upper(),
            "period": period,
            "var_95_pct": round(var_95 * 100, 4),
            "var_99_pct": round(var_99 * 100, 4),
            "cvar_95_pct": round(cvar_95 * 100, 4),
            "cvar_99_pct": round(cvar_99 * 100, 4),
            "volatility_daily_pct": round(volatility_daily * 100, 4),
            "volatility_annual_pct": round(volatility_annual * 100, 4),
            "max_drawdown_pct": round(max_drawdown * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Risk metrics failed for %s: %s", ticker, e)
        return {
            "ticker": ticker.upper(),
            "period": period,
            "var_95_pct": 0.0,
            "var_99_pct": 0.0,
            "cvar_95_pct": 0.0,
            "cvar_99_pct": 0.0,
            "volatility_daily_pct": 0.0,
            "volatility_annual_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "error": str(e),
        }


@router.get("/stress/scenarios")
async def list_stress_scenarios() -> Dict[str, Any]:
    """
    List available historical stress scenarios (Black Monday, COVID crash, etc.).
    Used by the terminal Portfolio / stress section.
    """
    try:
        from models.risk.stress_testing import HistoricalScenarioAnalyzer
        scenarios = HistoricalScenarioAnalyzer.list_scenarios()
        return {"scenarios": scenarios, "count": len(scenarios)}
    except Exception as e:
        logger.warning("Stress scenarios list failed: %s", e)
        return {"scenarios": [], "count": 0, "error": str(e)}


@router.get("/stress")
async def run_stress_ticker(
    ticker: str = Query(..., description="Stock symbol"),
) -> Dict[str, Any]:
    """
    Estimate stress impact for a single ticker across historical scenarios.
    Uses SPY/QQQ shock as proxy for equity tickers when symbol not in scenario.
    """
    try:
        from models.risk.stress_testing import HistoricalScenarioAnalyzer
        ticker = ticker.upper()
        results: List[Dict[str, Any]] = []
        for key, data in HistoricalScenarioAnalyzer.SCENARIOS.items():
            shocks = data.get("shocks", {})
            shock = shocks.get(ticker)
            if shock is None:
                shock = shocks.get("SPY") or shocks.get("QQQ") or 0.0
            results.append({
                "scenario_id": key,
                "name": data.get("name", key),
                "estimated_return_pct": round(float(shock) * 100, 2),
            })
        return {"ticker": ticker, "scenarios": results}
    except Exception as e:
        logger.warning("Stress run failed for %s: %s", ticker, e)
        return {"ticker": ticker.upper(), "scenarios": [], "error": str(e)}
