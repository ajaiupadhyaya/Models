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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Built frontend (Docker/production); when present, serve SPA from same origin
frontend_dist = project_root / "frontend" / "dist"
serve_spa = (frontend_dist / "index.html").exists()

# Lazy load routers to avoid import cascade issues
def get_routers():
    """
    Lazy load API routers for the core domains.

    The goal is to expose a clean, focused surface for the
    Bloomberg-style terminal while keeping advanced/institutional
    endpoints available but out of the default path.
    """
    # Core domain routers
    from api.models_api import router as models_router
    from api.predictions_api import router as predictions_router
    from api.backtesting_api import router as backtesting_router
    from api.websocket_api import router as websocket_router
    from api.monitoring import router as monitoring_router
    from api.paper_trading_api import router as paper_trading_router
    from api.investor_reports_api import router as investor_reports_router
    from api.company_analysis_api import router as company_analysis_router
    from api.ai_analysis_api import router as ai_router
    from api.data_api import router as data_router
    from api.risk_api import router as risk_router

    routers = {
        "models": models_router,
        "predictions": predictions_router,
        "backtesting": backtesting_router,
        "websocket": websocket_router,
        "monitoring": monitoring_router,
        "paper_trading": paper_trading_router,
        "investor_reports": investor_reports_router,
        "company": company_analysis_router,
        "ai": ai_router,
        "data": data_router,
        "risk": risk_router,
    }

    try:
        from api.automation_api import router as automation_router
        routers["automation"] = automation_router
    except Exception as e:
        logger.info(f"Automation router not available: {e}")
    try:
        from api.orchestrator_api import router as orchestrator_router
        routers["orchestrator"] = orchestrator_router
    except Exception as e:
        logger.info(f"Orchestrator router not available: {e}")

    # Advanced / institutional-grade routers stay importable but are not
    # part of the default core surface. They can still be mounted
    # manually from a custom main if desired.
    try:
        from api.comprehensive_api import router as comprehensive_router  # type: ignore
        routers["comprehensive"] = comprehensive_router
    except Exception as e:
        logger.info(f"Comprehensive router not available: {e}")

    try:
        from api.institutional_api import router as institutional_router  # type: ignore
        routers["institutional"] = institutional_router
    except Exception as e:
        logger.info(f"Institutional router not available: {e}")

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

# Request logging middleware: log method, path, status, duration
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
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

    routers = get_routers()
    app_state["routers_loaded"] = list(routers.keys())

    # Core domains
    app.include_router(routers["models"], prefix="/api/v1/models", tags=["Models"])
    app.include_router(routers["predictions"], prefix="/api/v1/predictions", tags=["Predictions"])
    app.include_router(routers["backtesting"], prefix="/api/v1/backtest", tags=["Backtesting"])
    app.include_router(routers["websocket"], prefix="/api/v1/ws", tags=["WebSocket"])
    app.include_router(routers["monitoring"], prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(routers["paper_trading"], prefix="/api/v1/paper-trading", tags=["Paper Trading"])
    app.include_router(routers["investor_reports"], tags=["Investor Reports"])
    app.include_router(routers["company"], prefix="/api/v1/company", tags=["Company Analysis"])
    app.include_router(routers["ai"], tags=["AI"])
    app.include_router(routers["data"], prefix="/api/v1/data", tags=["Data"])
    app.include_router(routers["risk"], prefix="/api/v1/risk", tags=["Risk"])

    if "automation" in routers:
        app.include_router(routers["automation"], tags=["Automation"])
    if "orchestrator" in routers:
        app.include_router(routers["orchestrator"], tags=["Orchestrator"])

    # Advanced / optional domains
    if "comprehensive" in routers:
        app.include_router(routers["comprehensive"], tags=["Comprehensive"])
    if "institutional" in routers:
        app.include_router(routers["institutional"], tags=["Institutional"])

    logger.info("Routers loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load some routers: {e}")
    # Continue anyway - some routers can fail

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
