"""
FastAPI Main Application

Production-ready API server for ML trading models with:
- Real-time predictions
- Model management
- Backtesting
- WebSocket streaming
- Performance monitoring
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Disable curl_cffi in yfinance BEFORE any yfinance imports
# This must be imported first to prevent Yahoo Finance blocking
from core import yfinance_session

# Custom JSON encoder to handle numpy, pandas, and NaN values
import json
import numpy as np
import pandas as pd
from decimal import Decimal

class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types and NaN values"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/Inf to None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, Decimal):
            return float(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

# Configure FastAPI to use custom JSON encoder
from fastapi.encoders import jsonable_encoder

original_jsonable_encoder = jsonable_encoder

def patched_jsonable_encoder(obj, *args, **kwargs):
    """Patched encoder that handles NaN/Inf values"""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: patched_jsonable_encoder(v, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [patched_jsonable_encoder(item, *args, **kwargs) for item in obj]
    return original_jsonable_encoder(obj, *args, **kwargs)

# Monkey patch the encoder
import fastapi.encoders
fastapi.encoders.jsonable_encoder = patched_jsonable_encoder

# Add a response middleware to clean NaN/Inf values
class NaNInfCleanupMiddleware(BaseHTTPMiddleware):
    """Middleware to clean NaN and Inf values from responses before JSON serialization"""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # For JSON responses, we need to clean the data before serialization
        if "application/json" in response.headers.get("content-type", ""):
            try:
                # This is handled by the patched encoder above
                pass
            except:
                pass
        return response

# Built frontend (Docker/production); when present, serve SPA from same origin
frontend_dist = project_root / "frontend" / "dist"
serve_spa = (frontend_dist / "index.html").exists()

# Lazy load routers to avoid import cascade issues; each router loaded in try/except
# so one failure (e.g. missing optional dep) does not prevent others from loading.
def get_routers():
    """
    Lazy load API routers for the core domains. Each router is loaded in try/except
    so deploy (e.g. Render) works even if one module fails (missing dep, config, etc.).
    """
    routers: Dict[str, Any] = {}

    def _add(name: str, load_fn):
        try:
            router = load_fn()
            if router is not None:
                routers[name] = router
        except Exception as e:
            logger.warning("Router %s not available: %s", name, e)

    # Core domain routers (charts, AI, ML, data, risk, etc.)
    _add("models", lambda: __import__("api.models_api", fromlist=["router"]).router)
    _add("predictions", lambda: __import__("api.predictions_api", fromlist=["router"]).router)
    _add("backtesting", lambda: __import__("api.backtesting_api", fromlist=["router"]).router)
    _add("websocket", lambda: __import__("api.websocket_api", fromlist=["router"]).router)
    _add("monitoring", lambda: __import__("api.monitoring", fromlist=["router"]).router)
    _add("paper_trading", lambda: __import__("api.paper_trading_api", fromlist=["router"]).router)
    _add("investor_reports", lambda: __import__("api.investor_reports_api", fromlist=["router"]).router)
    _add("company", lambda: __import__("api.company_analysis_api", fromlist=["router"]).router)
    _add("ai", lambda: __import__("api.ai_analysis_api", fromlist=["router"]).router)
    _add("data", lambda: __import__("api.data_api", fromlist=["router"]).router)
    _add("risk", lambda: __import__("api.risk_api", fromlist=["router"]).router)

    # Optional routers
    _add("automation", lambda: __import__("api.automation_api", fromlist=["router"]).router)
    _add("orchestrator", lambda: __import__("api.orchestrator_api", fromlist=["router"]).router)
    _add("screener", lambda: __import__("api.screener_api", fromlist=["router"]).router)
    _add("comprehensive", lambda: __import__("api.comprehensive_api", fromlist=["router"]).router)
    _add("institutional", lambda: __import__("api.institutional_api", fromlist=["router"]).router)

    return routers

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
        logger.info("Loaded %s pre-trained models", len(loaded_models))
    except Exception as e:
        logger.warning("Could not load models: %s", e)

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

# Request logging middleware: log method, path, status, duration
import time

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Return 429 when IP exceeds rate limit for /api/* (skips /health, /docs, etc.)."""

    async def dispatch(self, request: Request, call_next):
        from api.rate_limit import check_rate_limit
        allowed, retry_after = check_rate_limit(request)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


app.add_middleware(RequestLoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit (runs first on request; skip /health, /docs, etc.)
app.add_middleware(RateLimitMiddleware)


# Root endpoint (only when not serving built SPA)
if not serve_spa:
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
    Lightweight health check for load balancers and Render.
    Always returns 200 when the process is up; no DB or heavy work.
    """
    num_models = len(app_state.get("models") or {})
    conn_mgr = app_state.get("connection_manager")
    num_connections = len(conn_mgr.active_connections) if conn_mgr else 0
    return {
        "status": "healthy",
        "models_loaded": num_models,
        "active_connections": num_connections,
        "metrics_collector": app_state.get("metrics_collector") is not None,
    }


@app.get("/info", tags=["Health"])
async def system_info() -> Dict[str, Any]:
    """
    Get detailed system information including loaded routers and capabilities.
    
    Returns:
        dict: Detailed system status (AI, ML, RL, DL, WebSocket, etc.)
    """
    try:
        routers_loaded = app_state.get("routers_loaded", [])
        capabilities = []
        if "ai" in routers_loaded:
            capabilities.append("ai")
        if "predictions" in routers_loaded:
            capabilities.append("ml")
        if "orchestrator" in routers_loaded or "automation" in routers_loaded:
            capabilities.append("rl")
        if "predictions" in routers_loaded:
            capabilities.append("dl")
        return {
            "api_version": "1.0.0",
            "routers_loaded": routers_loaded,
            "capabilities": capabilities,
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


# Include routers — register auth first so POST /api/auth/login is never shadowed
try:
    from api.auth_api import router as auth_router
    app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
except Exception as e:
    logger.warning("Auth router not available: %s", e)

routers = {}
try:
    routers = get_routers()
    app_state["routers_loaded"] = list(routers.keys())
    logger.info("Routers loaded: %s", app_state["routers_loaded"])
except Exception as e:
    logger.warning("Failed to load routers: %s", e)
    app_state["routers_loaded"] = []

# Core domains (only include if loaded)
try:
    if "models" in routers:
        app.include_router(routers["models"], prefix="/api/v1/models", tags=["Models"])
    if "predictions" in routers:
        app.include_router(routers["predictions"], prefix="/api/v1/predictions", tags=["Predictions"])
    if "backtesting" in routers:
        app.include_router(routers["backtesting"], prefix="/api/v1/backtest", tags=["Backtesting"])
    if "websocket" in routers:
        app.include_router(routers["websocket"], prefix="/api/v1/ws", tags=["WebSocket"])
    if "monitoring" in routers:
        app.include_router(routers["monitoring"], prefix="/api/v1/monitoring", tags=["Monitoring"])
    if "paper_trading" in routers:
        app.include_router(routers["paper_trading"], prefix="/api/v1/paper-trading", tags=["Paper Trading"])
    if "investor_reports" in routers:
        app.include_router(routers["investor_reports"], prefix="/api/v1/reports", tags=["Investor Reports"])
    if "company" in routers:
        app.include_router(routers["company"], prefix="/api/v1/company", tags=["Company Analysis"])
    if "ai" in routers:
        app.include_router(routers["ai"], prefix="/api/v1/ai", tags=["AI"])
    if "data" in routers:
        app.include_router(routers["data"], prefix="/api/v1/data", tags=["Data"])
    try:
        from api.news_api import router as news_router
        app.include_router(news_router, prefix="/api/v1/data", tags=["News"])
        app_state["routers_loaded"].append("news")
    except Exception as e:
        logger.info("News router not available: %s", e)
    if "risk" in routers:
        app.include_router(routers["risk"], prefix="/api/v1/risk", tags=["Risk"])

    if "automation" in routers:
        app.include_router(routers["automation"], prefix="/api/v1/automation", tags=["Automation"])
    if "orchestrator" in routers:
        app.include_router(routers["orchestrator"], prefix="/api/v1/orchestrator", tags=["Orchestrator"])
    if "screener" in routers:
        app.include_router(routers["screener"], prefix="/api/v1/screener", tags=["Screener"])

    if "comprehensive" in routers:
        app.include_router(routers["comprehensive"], prefix="/api/v1/comprehensive", tags=["Comprehensive"])
    if "institutional" in routers:
        app.include_router(routers["institutional"], prefix="/api/v1/institutional", tags=["Institutional"])

    logger.info("Routers registered successfully")
except Exception as e:
    logger.warning("Failed to register routers: %s", e)
    app_state["routers_loaded"] = app_state.get("routers_loaded") or []

# Serve built SPA when frontend/dist exists (Docker/production)
# - Mount StaticFiles only at /assets (never at "/" — that would catch POST and return 405).
# - SPA fallback: middleware only. Do NOT add a catch-all GET route: it would match path /api/auth/login
#   and Starlette would return 405 for POST (path matched, method didn't). Middleware runs after
#   routing, so POST /api/auth/login hits the auth route; only 404 GET gets index.html.
if serve_spa:
    _spa_index = frontend_dist / "index.html"
    _spa_assets = frontend_dist / "assets"
    _api_path_prefixes = ("/api", "/docs", "/redoc", "/openapi.json")
    _api_exact = ("/health", "/info")

    if _spa_assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_spa_assets)), name="spa_assets")

    class SPAFallbackMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            if response.status_code != 404:
                return response
            path = request.url.path
            if request.method != "GET":
                return response
            if any(path.startswith(p) for p in _api_path_prefixes) or path in _api_exact:
                return response
            return FileResponse(str(_spa_index), media_type="text/html")

    app.add_middleware(SPAFallbackMiddleware)

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
