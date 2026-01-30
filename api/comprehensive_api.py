"""
Comprehensive Integration API
Exposes comprehensive analysis endpoints that integrate ALL components
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from core.comprehensive_integration import ComprehensiveIntegration

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/comprehensive", tags=["Comprehensive"])

# Global integration instance
_integration: Optional[ComprehensiveIntegration] = None


def get_integration() -> ComprehensiveIntegration:
    """Get or create global integration instance."""
    global _integration
    if _integration is None:
        _integration = ComprehensiveIntegration()
        _integration.initialize_all_components()
    return _integration


@router.post("/initialize")
async def initialize_comprehensive(symbols: List[str] = Query(["AAPL", "MSFT", "GOOGL"])):
    """
    Initialize comprehensive integration.
    
    Args:
        symbols: List of symbols to analyze
    
    Returns:
        Initialization status
    """
    try:
        global _integration
        _integration = ComprehensiveIntegration(symbols=symbols)
        _integration.initialize_all_components()
        
        return {
            "status": "initialized",
            "symbols": symbols,
            "components": _integration.get_integration_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}")
async def comprehensive_analysis(symbol: str):
    """
    Run comprehensive analysis for a symbol.
    Integrates ALL components: ML/DL/RL, Risk, Portfolio, Valuation, Options, AI.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Comprehensive analysis
    """
    try:
        integration = get_integration()
        analysis = integration.comprehensive_analysis(symbol)
        return analysis
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/daily-analysis")
async def run_daily_analysis(background_tasks: BackgroundTasks):
    """
    Run automated daily analysis for all symbols.
    
    Args:
        background_tasks: Background tasks
    
    Returns:
        Daily analysis results
    """
    try:
        integration = get_integration()
        
        # Run in background
        def run_analysis():
            return integration.automated_daily_analysis()
        
        background_tasks.add_task(run_analysis)
        
        return {
            "status": "started",
            "message": "Daily analysis running in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Daily analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_integration_status():
    """
    Get comprehensive integration status.
    
    Returns:
        Status of all integrated components
    """
    try:
        integration = get_integration()
        return integration.get_integration_status()
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, description="Number of alerts")
):
    """
    Get alerts from comprehensive analysis.
    
    Args:
        severity: Filter by severity (info, warning, critical)
        limit: Number of alerts
    
    Returns:
        List of alerts
    """
    try:
        integration = get_integration()
        
        from core.alerting_system import AlertSeverity
        severity_enum = None
        if severity:
            severity_enum = AlertSeverity(severity)
        
        alerts = integration.alerting.get_alerts(
            severity=severity_enum,
            limit=limit
        )
        
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Alert retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
