"""
Risk API

Endpoints for VaR, CVaR, volatility, and stress metrics per symbol.
Used by the terminal Portfolio panel for risk analytics.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import logging
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import pandas as pd

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
        import numpy as np

        fetcher = DataFetcher()
        data = fetcher.get_stock_data(ticker.upper(), period=period)
        if data is None or data.empty or "Close" not in data.columns:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        from core.utils import (
            calculate_returns,
            calculate_sharpe_ratio,
            calculate_max_drawdown,
            annualize_volatility,
        )
        returns = calculate_returns(data["Close"])
        if len(returns) < 20:
            raise HTTPException(status_code=400, detail="Insufficient data for risk metrics")

        # VaR/CVaR: use left tail (confidence_level = tail probability)
        # 95% VaR = 5th percentile; 99% VaR = 1st percentile
        var_95 = float(VaRModel.calculate_var(returns, confidence_level=0.05))
        var_99 = float(VaRModel.calculate_var(returns, confidence_level=0.01))
        cvar_95 = float(CVaRModel.calculate_cvar(returns, confidence_level=0.05))
        cvar_99 = float(CVaRModel.calculate_cvar(returns, confidence_level=0.01))

        volatility_daily = float(returns.std())
        volatility_annual = float(annualize_volatility(returns))
        max_drawdown = float(calculate_max_drawdown(returns))
        sharpe = float(calculate_sharpe_ratio(returns))
        if not np.isfinite(sharpe):
            sharpe = 0.0

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


@router.get("/optimize")
async def portfolio_optimize(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN,TSLA", description="Comma-separated symbols for portfolio"),
    period: str = Query("1y", description="History period for returns"),
    method: str = Query("sharpe", description="Optimization method: sharpe, min_vol, risk_parity"),
) -> Dict[str, Any]:
    """
    Mean-variance (or risk-parity) portfolio optimization from historical returns.
    Used by the terminal Portfolio panel for rebalancing suggestions.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from models.portfolio.optimization import optimize_portfolio_from_returns

        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
        if len(sym_list) < 2:
            return {"weights": {}, "error": "Provide at least 2 symbols"}
        data = yf.download(sym_list, period=period, progress=False, auto_adjust=True, group_by="ticker", threads=False)
        if data.empty:
            return {"weights": {}, "error": "No price data"}
        if isinstance(data.columns, pd.MultiIndex):
            # group_by='ticker' -> (Ticker, OHLCV); level 1 is Close
            try:
                closes = data.xs("Close", axis=1, level=1)
            except (KeyError, TypeError):
                level0 = data.columns.get_level_values(0).unique()
                closes = pd.DataFrame({s: data[s]["Close"] for s in sym_list if s in level0})
            if closes.empty or not hasattr(closes, "columns"):
                closes = pd.DataFrame()
            else:
                closes = closes.reindex(columns=[s for s in sym_list if s in closes.columns]).dropna(axis=1, how="all").dropna(axis=0, how="all")
        else:
            if "Close" in data.columns:
                closes = data[["Close"]].copy()
                closes.columns = sym_list[:1]
            else:
                return {"weights": {}, "error": "No close prices"}
        closes = closes.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if closes.shape[1] < 2:
            return {"weights": {}, "error": "Insufficient series for optimization"}
        returns = closes.pct_change().dropna()
        if len(returns) < 20:
            return {"weights": {}, "error": "Insufficient data points"}
        result = optimize_portfolio_from_returns(returns, method=method)
        weights_series = result.get("weights")
        if weights_series is not None and hasattr(weights_series, "to_dict"):
            weights_dict = {k: round(float(v), 4) for k, v in weights_series.to_dict().items()}
        else:
            weights_dict = {}
        return {
            "symbols": list(weights_dict.keys()),
            "weights": weights_dict,
            "expected_return": round(float(result.get("expected_return", 0)), 4),
            "volatility": round(float(result.get("volatility", 0)), 4),
            "sharpe_ratio": round(float(result.get("sharpe_ratio", 0)), 4) if result.get("sharpe_ratio") is not None else None,
        }
    except ValueError as e:
        return {"weights": {}, "error": str(e)}
    except Exception as e:
        logger.warning("Portfolio optimize failed: %s", e)
        return {"weights": {}, "error": str(e)}


@router.get("/portfolio/optimize-cvar")
async def portfolio_optimize_cvar(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN", description="Comma-separated symbols"),
    period: str = Query("1y", description="History period for returns"),
    method: str = Query("sharpe", description="Optimization method: sharpe, min_cvar"),
    risk_free_rate: float = Query(0.02, description="Risk-free rate for Sharpe")
) -> Dict[str, Any]:
    """
    CVaR-based portfolio optimization for tail-risk management.
    More sophisticated than mean-variance for downside risk.
    
    Phase 1 Awesome Quant Integration - riskfolio-lib
    """
    try:
        import yfinance as yf
        import pandas as pd
        from models.portfolio.advanced_optimization import CvaROptimizer, RiskParityOptimizer
        
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
        if len(sym_list) < 2:
            return {"weights": {}, "error": "Provide at least 2 symbols"}
        
        # Fetch data
        data = yf.download(sym_list, period=period, progress=False, auto_adjust=True, group_by="ticker", threads=False)
        if data.empty:
            return {"weights": {}, "error": "No price data"}
        
        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            try:
                closes = data.xs("Close", axis=1, level=1)
            except (KeyError, TypeError):
                level0 = data.columns.get_level_values(0).unique()
                closes = pd.DataFrame({s: data[s]["Close"] for s in sym_list if s in level0})
        else:
            if "Close" in data.columns:
                closes = data[["Close"]].copy()
                closes.columns = sym_list[:1]
            else:
                return {"weights": {}, "error": "No close prices"}
        
        closes = closes.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if closes.shape[1] < 2:
            return {"weights": {}, "error": "Insufficient series for optimization"}
        
        # Calculate returns
        returns = closes.pct_change().dropna()
        if len(returns) < 30:
            return {"weights": {}, "error": "Insufficient data points (need 30+)"}
        
        # Optimize
        if method == "risk_parity":
            optimizer = RiskParityOptimizer(returns)
            result = optimizer.optimize_risk_parity()
        else:
            optimizer = CvaROptimizer(returns)
            if method == "min_cvar":
                result = optimizer.optimize_min_cvar()
            else:
                result = optimizer.optimize_cvar(risk_free_rate=risk_free_rate)
        
        weights_dict = {k: round(float(v), 4) for k, v in result["weights"].items()}
        
        return {
            "method": method,
            "symbols": list(weights_dict.keys()),
            "weights": weights_dict,
            "expected_return": round(float(result.get("expected_return", 0)), 4),
            "volatility": round(float(result.get("volatility", 0)), 4),
            "sharpe_ratio": round(float(result.get("sharpe_ratio", 0)), 4) if result.get("sharpe_ratio") is not None else None,
            "cvar_95": round(float(result.get("cvar_95", 0)), 4) if result.get("cvar_95") is not None else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        return {"weights": {}, "error": str(e)}
    except Exception as e:
        logger.warning("CVaR portfolio optimize failed: %s", e)
        import traceback
        traceback.print_exc()
        return {"weights": {}, "error": str(e)}


@router.get("/portfolio/enhanced-metrics")
async def portfolio_enhanced_metrics(
    symbols: str = Query("AAPL,MSFT,GOOGL", description="Comma-separated symbols"),
    weights: str = Query("0.33,0.33,0.34", description="Comma-separated weights (must sum to 1)"),
    period: str = Query("1y", description="History period for returns"),
    risk_free_rate: float = Query(0.02, description="Risk-free rate")
) -> Dict[str, Any]:
    """
    Enhanced portfolio metrics: Sharpe, Sortino, Calmar, Max Drawdown, VaR/CVaR.
    
    Phase 1 Awesome Quant Integration - riskfolio-lib
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from models.portfolio.advanced_optimization import EnhancedPortfolioMetrics
        from datetime import datetime
        
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
        weight_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
        
        if len(sym_list) != len(weight_list):
            return {"error": "Number of symbols and weights must match"}
        
        if abs(sum(weight_list) - 1.0) > 0.01:
            return {"error": "Weights must sum to 1.0"}
        
        # Fetch data
        data = yf.download(sym_list, period=period, progress=False, auto_adjust=True, group_by="ticker", threads=False)
        if data.empty:
            return {"error": "No price data"}
        
        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            try:
                closes = data.xs("Close", axis=1, level=1)
            except (KeyError, TypeError):
                level0 = data.columns.get_level_values(0).unique()
                closes = pd.DataFrame({s: data[s]["Close"] for s in sym_list if s in level0})
        else:
            if "Close" in data.columns:
                closes = data[["Close"]].copy()
                closes.columns = sym_list[:1]
            else:
                return {"error": "No close prices"}
        
        closes = closes.dropna(axis=1, how="all").dropna(axis=0, how="all")
        returns = closes.pct_change().dropna()
        
        if len(returns) < 20:
            return {"error": "Insufficient data points (need 20+)"}
        
        # Calculate portfolio returns
        weights_series = pd.Series(weight_list, index=closes.columns)
        portfolio_returns = (returns * weights_series).sum(axis=1)
        
        # Calculate metrics
        metrics = EnhancedPortfolioMetrics.calculate_metrics(portfolio_returns, risk_free_rate)
        var_cvar = EnhancedPortfolioMetrics.calculate_var_cvar(portfolio_returns, confidence=0.95)
        
        return {
            "symbols": sym_list,
            "weights": {s: round(w, 4) for s, w in zip(sym_list, weight_list)},
            "period": period,
            "metrics": {
                "total_return_pct": round(metrics["total_return"] * 100, 2),
                "annual_return_pct": round(metrics["annual_return"] * 100, 2),
                "annual_volatility_pct": round(metrics["annual_volatility"] * 100, 2),
                "sharpe_ratio": round(metrics["sharpe_ratio"], 3),
                "sortino_ratio": round(metrics["sortino_ratio"], 3),
                "calmar_ratio": round(metrics["calmar_ratio"], 3),
                "max_drawdown_pct": round(metrics["max_drawdown"] * 100, 2),
                "var_95_pct": round(var_cvar["var"] * 100, 4),
                "cvar_95_pct": round(var_cvar["cvar"] * 100, 4),
            },
            "num_periods": metrics["num_periods"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning("Enhanced metrics failed: %s", e)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============ Phase 2: Multi-Factor Analysis Endpoints ============

@router.get("/multi-factor/{ticker}", tags=["Risk", "Factors"])
async def analyze_multi_factor(
    ticker: str,
    period: str = Query("1y", description="Data period (1y, 2y, 5y)"),
    factor_symbols: str = Query("SPY,IWM,EFA", description="Factor proxy symbols (comma-separated)")
) -> Dict[str, Any]:
    """
    Multi-factor model analysis using Fama-French style decomposition.
    
    Phase 2 Awesome Quant Integration - Multi-Factor Models
    
    Decomposes asset returns into:
    - Alpha (excess return)
    - Factor exposures (betas)
    - Factor attribution
    - Residual risk
    """
    try:
        from models.factors.multi_factor import MultiFactorModel
        import yfinance as yf
        
        # Download asset data
        asset = yf.download(ticker.upper(), period=period, progress=False)
        if asset.empty or "Close" not in asset.columns:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        
        asset_returns = asset["Close"].pct_change().dropna()
        
        # Download factor proxies
        factor_list = [f.strip().upper() for f in factor_symbols.split(',')]
        factor_data = yf.download(factor_list, period=period, progress=False)
        
        if factor_data.empty:
            raise HTTPException(status_code=404, detail="No factor data available")
        
        # Extract Close prices for factors
        if len(factor_list) == 1:
            factor_closes = factor_data["Close"].to_frame(factor_list[0])
        else:
            factor_closes = factor_data["Close"]
        
        factor_returns = factor_closes.pct_change().dropna()
        factor_returns.columns = [f"Factor_{i+1}" for i in range(len(factor_list))]
        
        # Fit multi-factor model
        model = MultiFactorModel(asset_returns, factor_returns)
        model.fit()
        
        # Get results
        summary = model.get_summary()
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "factor_proxies": factor_list,
            "alpha_pct_annual": round(summary["alpha"] * 252 * 100, 3),
            "alpha_pvalue": round(summary["alpha_pvalue"], 4),
            "alpha_significant": summary["alpha_significant"],
            "factor_exposures": {k: round(v, 4) for k, v in summary["factor_exposures"].items()},
            "factor_attribution_pct": {k: round(v * 100, 3) for k, v in summary["factor_attribution"].items()},
            "r_squared": round(summary["r_squared"], 4),
            "adj_r_squared": round(summary["adj_r_squared"], 4),
            "residual_volatility_pct": round(summary["residual_analysis"]["residual_std"] * np.sqrt(252) * 100, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-factor analysis failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/factor-ic", tags=["Risk", "Factors"])
async def calculate_factor_ic(
    factor_ticker: str,
    universe_symbols: str = Query("AAPL,MSFT,GOOGL,AMZN,TSLA", description="Universe symbols"),
    period: str = Query("1y", description="Data period"),
    forward_days: int = Query(5, description="Forward return period")
) -> Dict[str, Any]:
    """
    Calculate Information Coefficient (IC) for a factor across a universe.
    
    Phase 2 Awesome Quant Integration - Factor Analysis
    
    IC measures the correlation between factor values and forward returns.
    High IC indicates predictive power.
    """
    try:
        from models.factors.factor_analysis import SimpleFactorAnalysis
        import yfinance as yf
        
        # Get universe data
        universe_list = [s.strip().upper() for s in universe_symbols.split(',')]
        data = yf.download(universe_list, period=period, progress=False)
        
        if data.empty or "Close" not in data.columns:
            raise HTTPException(status_code=404, detail="No data for universe")
        
        # Get factor data
        factor_data = yf.download(factor_ticker.upper(), period=period, progress=False)
        if factor_data.empty or "Close" not in factor_data.columns:
            raise HTTPException(status_code=404, detail=f"No factor data for {factor_ticker}")
        
        # Extract closes
        if len(universe_list) == 1:
            universe_closes = data["Close"].to_frame(universe_list[0])
        else:
            universe_closes = data["Close"]
        
        factor_closes = factor_data["Close"]
        
        # Calculate returns
        universe_returns = universe_closes.pct_change()
        factor_returns = factor_closes.pct_change()
        
        # Forward returns
        forward_returns = universe_returns.shift(-forward_days)
        
        # Broadcast factor returns to match universe shape
        factor_values = pd.DataFrame({col: factor_returns for col in universe_returns.columns})
        
        # Calculate IC
        ic_series = SimpleFactorAnalysis.calculate_ic(factor_values, forward_returns, method='spearman')
        
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        return {
            "factor_ticker": factor_ticker.upper(),
            "universe": universe_list,
            "period": period,
            "forward_days": forward_days,
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "ic_information_ratio": round(ic_ir, 4),
            "num_observations": int(ic_series.notna().sum()),
            "interpretation": "Strong" if abs(ic_mean) > 0.05 else "Moderate" if abs(ic_mean) > 0.02 else "Weak",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Factor IC calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============ Phase 3: Options Pricing Endpoints ============

@router.get("/options/price", tags=["Risk", "Options"])
async def price_option(
    option_type: str = Query(..., description="'call' or 'put'"),
    spot_price: float = Query(..., description="Current stock price"),
    strike_price: float = Query(..., description="Strike price"),
    days_to_expiry: int = Query(..., description="Days until expiration"),
    volatility: float = Query(..., description="Annual volatility (e.g., 0.25 for 25%)"),
    risk_free_rate: float = Query(0.05, description="Risk-free rate (e.g., 0.05 for 5%)"),
    dividend_yield: float = Query(0.0, description="Continuous dividend yield")
) -> Dict[str, Any]:
    """
    Price European option using Black-Scholes model.
    
    Phase 3 Awesome Quant Integration - Options Pricing
    
    Returns option price and implied volatility calculations.
    """
    try:
        from models.derivatives.option_pricing import BlackScholes
        
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            raise HTTPException(status_code=400, detail="Days to expiry must be positive")
        
        if volatility <= 0:
            raise HTTPException(status_code=400, detail="Volatility must be positive")
        
        if option_type.lower() not in ["call", "put"]:
            raise HTTPException(status_code=400, detail="Option type must be 'call' or 'put'")
        
        # Calculate price
        if option_type.lower() == "call":
            price = BlackScholes.call_price(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield
            )
        else:
            price = BlackScholes.put_price(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield
            )
        
        # Moneyness
        moneyness = spot_price / strike_price
        intrinsic = max(spot_price - strike_price, 0) if option_type.lower() == "call" else max(strike_price - spot_price, 0)
        extrinsic = price - intrinsic
        
        return {
            "option_type": option_type.lower(),
            "spot_price": spot_price,
            "strike_price": strike_price,
            "days_to_expiry": days_to_expiry,
            "time_to_expiry_years": round(time_to_expiry, 4),
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "option_price": round(price, 4),
            "intrinsic_value": round(intrinsic, 4),
            "time_value": round(extrinsic, 4),
            "moneyness": round(moneyness, 4),
            "moneyness_type": "ITM" if (option_type.lower() == "call" and spot_price > strike_price) or (option_type.lower() == "put" and spot_price < strike_price) else "OTM" if (option_type.lower() == "call" and spot_price < strike_price) or (option_type.lower() == "put" and spot_price > strike_price) else "ATM",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Option pricing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/greeks", tags=["Risk", "Options"])
async def calculate_greeks(
    option_type: str = Query(..., description="'call' or 'put'"),
    spot_price: float = Query(..., description="Current stock price"),
    strike_price: float = Query(..., description="Strike price"),
    days_to_expiry: int = Query(..., description="Days until expiration"),
    volatility: float = Query(..., description="Annual volatility"),
    risk_free_rate: float = Query(0.05, description="Risk-free rate"),
    dividend_yield: float = Query(0.0, description="Dividend yield")
) -> Dict[str, Any]:
    """
    Calculate option Greeks (delta, gamma, vega, theta, rho).
    
    Phase 3 Awesome Quant Integration - Options Greeks
    
    Greeks measure option price sensitivity to various parameters.
    """
    try:
        from models.derivatives.option_pricing import OptionAnalyzer
        
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            raise HTTPException(status_code=400, detail="Days to expiry must be positive")
        
        if option_type.lower() not in ["call", "put"]:
            raise HTTPException(status_code=400, detail="Option type must be 'call' or 'put'")
        
        # Get complete analysis
        analysis = OptionAnalyzer.analyze_option(
            option_type.lower(),
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield
        )
        
        return {
            "option_type": option_type.lower(),
            "spot_price": spot_price,
            "strike_price": strike_price,
            "days_to_expiry": days_to_expiry,
            "price": analysis["price"],
            "greeks": {
                "delta": analysis["delta"],
                "gamma": analysis["gamma"],
                "vega": analysis["vega"],
                "theta": analysis["theta"],
                "rho": analysis["rho"]
            },
            "values": {
                "intrinsic": analysis["intrinsic_value"],
                "time_value": analysis["time_value"]
            },
            "moneyness": {
                "ratio": analysis["moneyness"],
                "type": analysis["moneyness_type"]
            },
            "interpretations": {
                "delta": f"Option price moves ${abs(analysis['delta']):.2f} for $1 move in underlying",
                "gamma": f"Delta changes by {abs(analysis['gamma']):.4f} for $1 move",
                "vega": f"Option price changes ${abs(analysis['vega']):.2f} for 1% volatility increase",
                "theta": f"Option loses ${abs(analysis['theta']):.2f} per day from time decay",
                "rho": f"Option price changes ${abs(analysis['rho']):.2f} for 1% rate change"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Greeks calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/implied-volatility", tags=["Risk", "Options"])
async def calculate_implied_volatility(
    option_type: str = Query(..., description="'call' or 'put'"),
    market_price: float = Query(..., description="Observed market price of option"),
    spot_price: float = Query(..., description="Current stock price"),
    strike_price: float = Query(..., description="Strike price"),
    days_to_expiry: int = Query(..., description="Days until expiration"),
    risk_free_rate: float = Query(0.05, description="Risk-free rate"),
    dividend_yield: float = Query(0.0, description="Dividend yield")
) -> Dict[str, Any]:
    """
    Calculate implied volatility from market option price.
    
    Phase 3 Awesome Quant Integration - Implied Volatility
    
    Backs out the volatility implied by the market price.
    """
    try:
        from models.derivatives.option_pricing import ImpliedVolatility
        
        # Convert days to years
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            raise HTTPException(status_code=400, detail="Days to expiry must be positive")
        
        if option_type.lower() not in ["call", "put"]:
            raise HTTPException(status_code=400, detail="Option type must be 'call' or 'put'")
        
        # Calculate implied volatility
        if option_type.lower() == "call":
            iv = ImpliedVolatility.call_iv(
                market_price, spot_price, strike_price,
                time_to_expiry, risk_free_rate, dividend_yield
            )
        else:
            iv = ImpliedVolatility.put_iv(
                market_price, spot_price, strike_price,
                time_to_expiry, risk_free_rate, dividend_yield
            )
        
        if iv is None:
            raise HTTPException(status_code=400, detail="Could not calculate implied volatility (check inputs)")
        
        return {
            "option_type": option_type.lower(),
            "market_price": market_price,
            "spot_price": spot_price,
            "strike_price": strike_price,
            "days_to_expiry": days_to_expiry,
            "implied_volatility": round(iv, 4),
            "implied_volatility_pct": round(iv * 100, 2),
            "interpretation": f"Market implies {iv*100:.1f}% annual volatility",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Implied volatility calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
