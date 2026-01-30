"""
Automated Trading Orchestrator API

This API exposes a single, enhanced orchestrator that coordinates
AI/ML/DL/RL models for automated trading. It is the primary engine
behind the Bloomberg-style terminal.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from core.enhanced_orchestrator import EnhancedOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/orchestrator", tags=["Orchestrator"])

# Global orchestrator instance (enhanced engine)
_orchestrator: Optional[EnhancedOrchestrator] = None


def get_orchestrator() -> EnhancedOrchestrator:
    """Get or create the global enhanced orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        # Default symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        _orchestrator = EnhancedOrchestrator(
            symbols=symbols,
            use_rl=True,
            use_lstm=True,
            use_ensemble=True
        )
    return _orchestrator


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator."""
    symbols: List[str]
    initial_capital: float = 100000
    retrain_frequency: str = "daily"
    use_rl: bool = True
    use_lstm: bool = True
    use_ensemble: bool = True
    risk_limit: float = 0.02


@router.post("/initialize")
async def initialize_orchestrator(config: OrchestratorConfig):
    """
    Initialize the automated trading orchestrator.
    
    Args:
        config: Orchestrator configuration
    
    Returns:
        Initialization status
    """
    try:
        global _orchestrator
        _orchestrator = EnhancedOrchestrator(
            symbols=config.symbols,
            initial_capital=config.initial_capital,
            retrain_frequency=config.retrain_frequency,
            use_rl=config.use_rl,
            use_lstm=config.use_lstm,
            use_ensemble=config.use_ensemble,
            risk_limit=config.risk_limit
        )
        
        # Initialize models
        _orchestrator.initialize_models()
        
        return {
            "status": "initialized",
            "symbols": config.symbols,
            "models_initialized": _orchestrator.get_status()["models_initialized"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-cycle")
async def run_trading_cycle(
    execute_trades: bool = Query(False, description="Execute trades"),
    background_tasks: BackgroundTasks = None
):
    """
    Run one complete trading cycle.
    
    Args:
        execute_trades: Whether to execute trades
        background_tasks: Background tasks
    
    Returns:
        Cycle results
    """
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run_cycle(execute_trades=execute_trades)
        return result
    except Exception as e:
        logger.error(f"Cycle failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-automated")
async def start_automated_trading(
    interval_minutes: int = Query(60, description="Interval between cycles (minutes)"),
    execute: bool = Query(False, description="Execute trades"),
    background_tasks: BackgroundTasks = None
):
    """
    Start continuous automated trading.
    
    Args:
        interval_minutes: Minutes between cycles
        execute: Whether to execute trades
        background_tasks: Background tasks
    
    Returns:
        Start status
    """
    try:
        orchestrator = get_orchestrator()
        orchestrator.start_automated_trading(
            interval_minutes=interval_minutes,
            execute=execute
        )
        return {
            "status": "started",
            "interval_minutes": interval_minutes,
            "execute_trades": execute,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain_models(
    symbol: Optional[str] = Query(None, description="Specific symbol to retrain")
):
    """
    Manually trigger model retraining.
    
    Args:
        symbol: Specific symbol to retrain, or None for all
    
    Returns:
        Retraining status
    """
    try:
        orchestrator = get_orchestrator()
        orchestrator.retrain_models(symbol=symbol)
        return {
            "status": "retraining",
            "symbol": symbol or "all",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_orchestrator_status():
    """
    Get orchestrator status.
    
    Returns:
        Status information
    """
    try:
        orchestrator = get_orchestrator()
        # Enhanced orchestrator exposes richer status information, including
        # model counts, symbols, and market regime diagnostics.
        return orchestrator.get_enhanced_status()
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
async def get_latest_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(10, description="Number of signals to return")
):
    """
    Get latest trading signals.
    
    Args:
        symbol: Filter by symbol
        limit: Number of signals
    
    Returns:
        List of signals
    """
    try:
        orchestrator = get_orchestrator()
        signals = orchestrator.signals_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        # Sort by timestamp (newest first)
        signals = sorted(signals, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return {
            "signals": [
                {
                    "symbol": s.symbol,
                    "action": s.action,
                    "confidence": s.confidence,
                    "price": s.price,
                    "target_price": s.target_price,
                    "stop_loss": s.stop_loss,
                    "reasoning": s.reasoning,
                    "model_source": s.model_source,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in signals
            ],
            "count": len(signals)
        }
    except Exception as e:
        logger.error(f"Signal retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, description="Number of trades to return")
):
    """
    Get trade history.
    
    Args:
        symbol: Filter by symbol
        limit: Number of trades
    
    Returns:
        List of trades
    """
    try:
        orchestrator = get_orchestrator()
        trades = orchestrator.trade_history
        
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        
        trades = trades[-limit:]  # Most recent
        
        return {
            "trades": trades,
            "count": len(trades)
        }
    except Exception as e:
        logger.error(f"Trade retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-signals/{symbol}")
async def generate_signals_for_symbol(symbol: str):
    """
    Generate trading signals for a specific symbol.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Generated signals
    """
    try:
        orchestrator = get_orchestrator()
        signals = orchestrator.generate_signals(symbol)
        
        return {
            "symbol": symbol,
            "signals": [
                {
                    "action": s.action,
                    "confidence": s.confidence,
                    "price": s.price,
                    "target_price": s.target_price,
                    "stop_loss": s.stop_loss,
                    "reasoning": s.reasoning,
                    "model_source": s.model_source,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in signals
            ],
            "count": len(signals)
        }
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
