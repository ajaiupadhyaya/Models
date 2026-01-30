"""
Backtesting API

Endpoints for running backtests through the API:
- Run single strategy backtest
- Compare multiple strategies
- Walk-forward analysis
- Parameter optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly to avoid cascade
import importlib.util
spec = importlib.util.spec_from_file_location(
    "backtesting",
    project_root / "core" / "backtesting.py"
)
backtesting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtesting_module)

BacktestEngine = backtesting_module.BacktestEngine
BacktestSignal = backtesting_module.BacktestSignal
WalkForwardAnalysis = backtesting_module.WalkForwardAnalysis

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class BacktestRequest(BaseModel):
    """Request for backtesting."""
    model_name: str = Field(description="Name of the model to backtest")
    symbol: str = Field(description="Stock symbol", example="SPY")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100000.0, description="Starting capital")
    commission: float = Field(default=0.001, description="Commission rate")
    position_size: float = Field(default=1.0, description="Position size (0-1)")


class CompareStrategiesRequest(BaseModel):
    """Request to compare multiple strategies."""
    model_names: List[str] = Field(description="List of models to compare")
    symbol: str = Field(description="Stock symbol")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100000.0)


class WalkForwardRequest(BaseModel):
    """Request for walk-forward analysis."""
    model_name: str = Field(description="Name of the model")
    symbol: str = Field(description="Stock symbol")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    train_window: int = Field(default=252, description="Training window in days")
    test_window: int = Field(default=63, description="Testing window in days")


class BacktestResponse(BaseModel):
    """Response with backtest results."""
    model_name: str
    symbol: str
    period: Dict[str, str]
    metrics: Dict[str, float]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    status: str


class CompareStrategiesResponse(BaseModel):
    """Response comparing strategies."""
    symbol: str
    period: Dict[str, str]
    strategies: List[Dict[str, Any]]
    best_strategy: Dict[str, str]


# Helper functions
def get_app_state() -> Dict[str, Any]:
    """Get global app state."""
    from api.main import get_app_state
    return get_app_state()


@router.get("/sample-data")
async def get_sample_data(
    symbol: str = "AAPL",
    period: str = "3mo"
) -> Dict[str, Any]:
    """
    Get OHLCV sample data for charting (used by frontend Primary Instrument).
    Returns candles in format expected by the terminal candlestick chart.
    """
    try:
        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if data.empty or len(data) == 0:
            return {"candles": [], "symbol": symbol, "error": "No data found"}
        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = data.columns.get_level_values(0)
        close_col = "Close" if "Close" in data.columns else "Adj Close"
        vol_col = "Volume" if "Volume" in data.columns else None
        candles = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(data["Open"].loc[idx]),
                "high": float(data["High"].loc[idx]),
                "low": float(data["Low"].loc[idx]),
                "close": float(data[close_col].loc[idx]),
                "volume": int(data[vol_col].loc[idx]) if vol_col else 0,
            }
            for idx in data.index
        ]
        return {"candles": candles, "symbol": symbol, "period": period}
    except Exception as e:
        logger.warning(f"Sample data fetch failed for {symbol}: {e}")
        return {"candles": [], "symbol": symbol, "error": str(e)}


# API Endpoints

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest for a model.
    
    Args:
        request: Backtest configuration
        
    Returns:
        BacktestResponse: Backtest results and metrics
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Get model
        if request.model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found"
            )
        
        model_data = models[request.model_name]
        model = model_data["model"]
        metadata = model_data["metadata"]
        
        # Download data
        logger.info(f"Downloading data for {request.symbol}")
        data = yf.download(
            request.symbol,
            start=request.start_date,
            end=request.end_date,
            progress=False
        )
        
        if data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for {request.symbol}"
            )
        
        # Generate signals
        model_type = metadata.get("type", "unknown")
        signals = []
        
        logger.info(f"Generating signals using {model_type} model")
        
        if model_type == "simple":
            # Simple predictor
            signal_obj = model.predict(data)
            # Expand to all dates
            for date in data.index:
                signals.append(BacktestSignal(
                    date=date,
                    signal=signal_obj.signal,
                    confidence=signal_obj.confidence
                ))
        
        elif model_type == "ensemble":
            # Ensemble predictor
            features = model.calculate_features(data)
            features = features.dropna()
            
            for idx in features.index:
                signal_val = model.predict(features.loc[[idx]])
                signals.append(BacktestSignal(
                    date=idx,
                    signal=float(signal_val),
                    confidence=0.7
                ))
        
        elif model_type == "lstm":
            # LSTM predictor
            X, _ = model.prepare_data(data)
            predictions = model.predict(X)
            
            # Match predictions to dates (account for lookback window)
            lookback = model.lookback_window
            for i, pred in enumerate(predictions):
                if i + lookback < len(data):
                    signals.append(BacktestSignal(
                        date=data.index[i + lookback],
                        signal=float(pred),
                        confidence=0.8
                    ))
        
        # Run backtest
        logger.info("Running backtest...")
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission=request.commission,
            position_size=request.position_size
        )
        
        results = engine.run_backtest(data, signals)
        
        # Format response
        equity_curve = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "equity": float(equity)
            }
            for date, equity in zip(
                results["equity_curve"].index,
                results["equity_curve"].values
            )
        ]
        
        trades = [
            {
                "entry_date": trade.entry_date.strftime("%Y-%m-%d"),
                "exit_date": trade.exit_date.strftime("%Y-%m-%d") if trade.exit_date else None,
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price) if trade.exit_price else None,
                "quantity": float(trade.quantity),
                "pnl": float(trade.pnl) if trade.pnl else None,
                "return_pct": float(trade.return_pct) if trade.return_pct else None
            }
            for trade in results["trades"]
        ]
        
        metrics = {
            key: float(value) if value is not None else None
            for key, value in results["metrics"].items()
        }
        
        return BacktestResponse(
            model_name=request.model_name,
            symbol=request.symbol,
            period={
                "start": request.start_date,
                "end": request.end_date or "now"
            },
            metrics=metrics,
            equity_curve=equity_curve[-100:],  # Last 100 points
            trades=trades,
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareStrategiesResponse)
async def compare_strategies(request: CompareStrategiesRequest) -> CompareStrategiesResponse:
    """
    Compare multiple strategies.
    
    Args:
        request: Comparison configuration
        
    Returns:
        CompareStrategiesResponse: Comparison results
    """
    try:
        strategies = []
        
        # Run backtest for each model
        for model_name in request.model_names:
            try:
                backtest_request = BacktestRequest(
                    model_name=model_name,
                    symbol=request.symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital
                )
                
                result = await run_backtest(backtest_request)
                
                strategies.append({
                    "model_name": model_name,
                    "metrics": result.metrics,
                    "final_equity": result.equity_curve[-1]["equity"] if result.equity_curve else 0,
                    "num_trades": len(result.trades)
                })
                
            except Exception as e:
                logger.warning(f"Failed to backtest {model_name}: {e}")
                strategies.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        # Find best strategy (by Sharpe ratio)
        valid_strategies = [s for s in strategies if "error" not in s]
        
        if valid_strategies:
            best = max(
                valid_strategies,
                key=lambda x: x["metrics"].get("sharpe_ratio", -999)
            )
            best_strategy = {
                "model_name": best["model_name"],
                "sharpe_ratio": best["metrics"].get("sharpe_ratio"),
                "total_return": best["metrics"].get("total_return")
            }
        else:
            best_strategy = {"model_name": "none", "reason": "all strategies failed"}
        
        return CompareStrategiesResponse(
            symbol=request.symbol,
            period={
                "start": request.start_date,
                "end": request.end_date or "now"
            },
            strategies=strategies,
            best_strategy=best_strategy
        )
    
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/walk-forward", response_model=Dict[str, Any])
async def walk_forward_analysis(request: WalkForwardRequest) -> Dict[str, Any]:
    """
    Run walk-forward analysis.
    
    Args:
        request: Walk-forward configuration
        
    Returns:
        dict: Walk-forward analysis results
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Get model
        if request.model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found"
            )
        
        model_data = models[request.model_name]
        model = model_data["model"]
        
        # Download data
        logger.info(f"Downloading data for {request.symbol}")
        data = yf.download(
            request.symbol,
            start=request.start_date,
            end=request.end_date,
            progress=False
        )
        
        if data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for {request.symbol}"
            )
        
        # Run walk-forward analysis
        logger.info("Running walk-forward analysis...")
        wfa = WalkForwardAnalysis(
            train_window=request.train_window,
            test_window=request.test_window
        )
        
        results = wfa.run(model, data)
        
        # Format results
        windows = [
            {
                "window": i + 1,
                "train_start": w["train_start"].strftime("%Y-%m-%d"),
                "train_end": w["train_end"].strftime("%Y-%m-%d"),
                "test_start": w["test_start"].strftime("%Y-%m-%d"),
                "test_end": w["test_end"].strftime("%Y-%m-%d"),
                "metrics": {
                    key: float(value) if value is not None else None
                    for key, value in w["metrics"].items()
                }
            }
            for i, w in enumerate(results["windows"])
        ]
        
        summary = {
            key: float(value) if value is not None else None
            for key, value in results["summary"].items()
        }
        
        return {
            "model_name": request.model_name,
            "symbol": request.symbol,
            "train_window": request.train_window,
            "test_window": request.test_window,
            "num_windows": len(windows),
            "windows": windows,
            "summary": summary,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_available_metrics() -> Dict[str, List[str]]:
    """
    Get list of available backtest metrics.
    
    Returns:
        dict: Available metrics with descriptions
    """
    return {
        "performance_metrics": [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio"
        ],
        "risk_metrics": [
            "max_drawdown",
            "volatility",
            "downside_deviation",
            "var_95",
            "cvar_95"
        ],
        "trade_metrics": [
            "num_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "avg_trade_duration"
        ]
    }
