"""
FastAPI Main Application

Production-ready API server for ML trading models with:
- Real-time predictions
- Model management
- Backtesting
- WebSocket streaming
- Performance monitoring
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Lazy load routers to avoid import cascade issues
def get_routers():
    """Lazy load all API routers."""
    from api.models_api import router as models_router
    from api.predictions_api import router as predictions_router
    from api.backtesting_api import router as backtesting_router
    from api.websocket_api import router as websocket_router
    from api.monitoring import router as monitoring_router
    from api.paper_trading_api import router as paper_trading_router
    from api.investor_reports_api import router as investor_reports_router
    from api.company_analysis_api import router as company_analysis_router
    from api.ai_analysis_api import router as ai_analysis_router
    from api.automation_api import router as automation_router
    return models_router, predictions_router, backtesting_router, websocket_router, monitoring_router, paper_trading_router, investor_reports_router, company_analysis_router, ai_analysis_router, automation_router

def get_managers():
    """Lazy load connection and metrics managers."""
    from api.websocket_api import ConnectionManager
    from api.monitoring import MetricsCollector
    return ConnectionManager, MetricsCollector

ConnectionManager = None
MetricsCollector = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state: Dict[str, Any] = {
    "models": {},
    "connection_manager": None,
    "metrics_collector": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Initialize connection manager
    - Load pre-trained models
    - Start metrics collection
    - Cleanup resources on shutdown
    """
    # Startup
    logger.info("Starting API server...")
    
    # Load managers
    try:
        ConnectionManager_cls, MetricsCollector_cls = get_managers()
        app_state["connection_manager"] = ConnectionManager_cls()
        app_state["metrics_collector"] = MetricsCollector_cls()
        logger.info("Connection manager and metrics collector initialized")
    except Exception as e:
        logger.warning(f"Could not initialize managers: {e}")
    
    # Load models (if any saved models exist)
    try:
        from api.models_api import load_saved_models
        loaded_models = await load_saved_models()
        app_state["models"].update(loaded_models)
        logger.info(f"Loaded {len(loaded_models)} pre-trained models")
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
    
    logger.info("API server ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    
    # Close WebSocket connections
    if app_state["connection_manager"]:
        try:
            await app_state["connection_manager"].disconnect_all()
            logger.info("All WebSocket connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    # Save metrics
    if app_state["metrics_collector"]:
        try:
            app_state["metrics_collector"].save_metrics()
            logger.info("Metrics saved")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    logger.info("API server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Trading ML API",
    description="Production-grade API for ML trading models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """
    Root endpoint - API health check.
    
    Returns:
        dict: API status and version
    """
    return {
        "status": "online",
        "message": "Trading ML API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint with system status.
    
    Returns:
        dict: System health metrics
    """
    try:
        num_models = len(app_state["models"])
        num_connections = (
            len(app_state["connection_manager"].active_connections)
            if app_state["connection_manager"]
            else 0
        )
        
        return {
            "status": "healthy",
            "models_loaded": num_models,
            "active_connections": num_connections,
            "metrics_collector": app_state["metrics_collector"] is not None,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Health"])
async def system_info() -> Dict[str, Any]:
    """
    Get detailed system information.
    
    Returns:
        dict: Detailed system status
    """
    try:
        return {
            "api_version": "1.0.0",
            "models": {
                "loaded": list(app_state["models"].keys()),
                "count": len(app_state["models"])
            },
            "websocket": {
                "enabled": app_state["connection_manager"] is not None,
                "connections": (
                    len(app_state["connection_manager"].active_connections)
                    if app_state["connection_manager"]
                    else 0
                )
            },
            "monitoring": {
                "enabled": app_state["metrics_collector"] is not None,
            }
        }
    except Exception as e:
        logger.error(f"System info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include routers
try:
    models_router, predictions_router, backtesting_router, websocket_router, monitoring_router, paper_trading_router, investor_reports_router, company_analysis_router, ai_analysis_router, automation_router = get_routers()
    app.include_router(models_router, prefix="/api/v1/models", tags=["Models"])
    app.include_router(predictions_router, prefix="/api/v1/predictions", tags=["Predictions"])
    app.include_router(backtesting_router, prefix="/api/v1/backtest", tags=["Backtesting"])
    app.include_router(websocket_router, prefix="/api/v1/ws", tags=["WebSocket"])
    app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(paper_trading_router, prefix="/api/v1/paper-trading", tags=["Paper Trading"])
    app.include_router(investor_reports_router, tags=["Investor Reports"])
    app.include_router(company_analysis_router, prefix="/api/v1/company", tags=["Company Analysis"])
    app.include_router(ai_analysis_router, tags=["AI Analysis"])
    app.include_router(automation_router, tags=["Automation"])
    logger.info("All routers loaded successfully (including automation)")
except Exception as e:
    logger.warning(f"Failed to load some routers: {e}")
    # Continue anyway - some routers can fail


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Make app_state accessible to routers
def get_app_state() -> Dict[str, Any]:
    """Get global application state."""
    return app_state


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
