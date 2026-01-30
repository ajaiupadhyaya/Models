"""
Institutional-Grade API Endpoints
Jane Street / Citadel Level Analysis
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query

from core.integration_institutional import InstitutionalIntegration
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/institutional", tags=["Institutional"])

# Global institutional integration instance
_institutional: Optional[InstitutionalIntegration] = None


def get_institutional() -> InstitutionalIntegration:
    """Get or create global institutional integration instance."""
    global _institutional
    if _institutional is None:
        _institutional = InstitutionalIntegration()
        _institutional.initialize_all_components()
    return _institutional


@router.post("/initialize")
async def initialize_institutional(symbols: List[str] = Query(["AAPL", "MSFT", "GOOGL"])):
    """
    Initialize institutional-grade integration.
    
    Args:
        symbols: List of symbols to analyze
    
    Returns:
        Initialization status
    """
    try:
        global _institutional
        _institutional = InstitutionalIntegration(symbols=symbols)
        _institutional.initialize_all_components()
        
        return {
            "status": "initialized",
            "symbols": symbols,
            "institutional_grade": True,
            "components": _institutional.get_institutional_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Institutional initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}")
async def institutional_analysis(symbol: str):
    """
    Run institutional-grade analysis for a symbol.
    Uses only institutional-grade models and methods.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Institutional analysis
    """
    try:
        institutional = get_institutional()
        analysis = institutional.institutional_analysis(symbol)
        return analysis
    except Exception as e:
        logger.error(f"Institutional analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_institutional_status():
    """
    Get institutional integration status.
    
    Returns:
        Status of all institutional components
    """
    try:
        institutional = get_institutional()
        return institutional.get_institutional_status()
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def institutional_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy: str = Query("momentum", description="Strategy type")
):
    """
    Run institutional-grade backtest with proper transaction costs.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: Strategy type
    
    Returns:
        Backtest results
    """
    try:
        institutional = get_institutional()
        
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period="2y")
        
        if df is None or len(df) == 0:
            raise HTTPException(status_code=404, detail="Data not found")
        
        # Filter dates
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Generate signals (simplified - would use actual strategy)
        returns = df['Close'].pct_change()
        signals = np.where(returns > returns.rolling(20).mean(), 1,
                          np.where(returns < returns.rolling(20).mean(), -1, 0))
        
        # Run backtest
        results = institutional.institutional_backtest(df, signals)
        
        return {
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "strategy": strategy,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
