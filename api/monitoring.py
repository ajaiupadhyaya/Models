"""
Monitoring and Metrics API

Endpoints for monitoring system performance:
- Prediction metrics
- Model performance tracking
- System health metrics
- Real-time monitoring dashboard data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict, deque
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()

# Metrics storage directory
METRICS_DIR = Path(__file__).parent.parent / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class MetricsCollector:
    """Collect and track system metrics."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of records to keep in memory
        """
        self.max_history = max_history
        self.predictions: deque = deque(maxlen=max_history)
        self.model_performance: Dict[str, List[Dict]] = defaultdict(list)
        self.api_calls: deque = deque(maxlen=max_history)
        self.errors: deque = deque(maxlen=max_history)
    
    def record_prediction(
        self,
        model_name: str,
        symbol: str,
        signal: float,
        confidence: float
    ) -> None:
        """
        Record a prediction.
        
        Args:
            model_name: Name of the model
            symbol: Stock symbol
            signal: Prediction signal
            confidence: Prediction confidence
        """
        self.predictions.append({
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence
        })
    
    def record_api_call(
        self,
        endpoint: str,
        duration: float,
        status_code: int
    ) -> None:
        """
        Record an API call.
        
        Args:
            endpoint: API endpoint called
            duration: Request duration in seconds
            status_code: HTTP status code
        """
        self.api_calls.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "duration": duration,
            "status_code": status_code
        })
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """
        Record an error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        })
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            dict: Model statistics
        """
        model_predictions = [
            p for p in self.predictions
            if p["model_name"] == model_name
        ]
        
        if not model_predictions:
            return {
                "total_predictions": 0,
                "avg_confidence": 0,
                "avg_signal": 0
            }
        
        signals = [p["signal"] for p in model_predictions]
        confidences = [p["confidence"] for p in model_predictions]
        
        return {
            "total_predictions": len(model_predictions),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_signal": sum(signals) / len(signals),
            "buy_signals": sum(1 for s in signals if s > 0.2),
            "sell_signals": sum(1 for s in signals if s < -0.2),
            "hold_signals": sum(1 for s in signals if -0.2 <= s <= 0.2)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get overall system statistics.
        
        Returns:
            dict: System statistics
        """
        # Calculate time windows
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Filter recent API calls
        recent_calls = [
            c for c in self.api_calls
            if datetime.fromisoformat(c["timestamp"]) > last_hour
        ]
        
        # Filter recent errors
        recent_errors = [
            e for e in self.errors
            if datetime.fromisoformat(e["timestamp"]) > last_day
        ]
        
        # Calculate average response time
        if recent_calls:
            avg_response_time = sum(c["duration"] for c in recent_calls) / len(recent_calls)
        else:
            avg_response_time = 0
        
        return {
            "total_predictions": len(self.predictions),
            "total_api_calls": len(self.api_calls),
            "total_errors": len(self.errors),
            "recent_api_calls_1h": len(recent_calls),
            "recent_errors_24h": len(recent_errors),
            "avg_response_time": avg_response_time,
            "uptime": "N/A"  # Would need startup time tracking
        }
    
    def save_metrics(self, filename: str = None) -> None:
        """
        Save metrics to disk.
        
        Args:
            filename: Optional filename, defaults to timestamp
        """
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = METRICS_DIR / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "predictions": list(self.predictions),
            "api_calls": list(self.api_calls),
            "errors": list(self.errors)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filename: str) -> None:
        """
        Load metrics from disk.
        
        Args:
            filename: Filename to load
        """
        filepath = METRICS_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filename}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.predictions.extend(data.get("predictions", []))
        self.api_calls.extend(data.get("api_calls", []))
        self.errors.extend(data.get("errors", []))
        
        logger.info(f"Metrics loaded from {filepath}")


def get_app_state() -> Dict[str, Any]:
    """Get global app state."""
    from api.main import get_app_state
    return get_app_state()


# Response Models
class SystemMetrics(BaseModel):
    """System-level metrics."""
    total_predictions: int
    total_api_calls: int
    total_errors: int
    recent_api_calls_1h: int
    recent_errors_24h: int
    avg_response_time: float
    uptime: str


class ModelMetrics(BaseModel):
    """Model-specific metrics."""
    model_name: str
    total_predictions: int
    avg_confidence: float
    avg_signal: float
    buy_signals: int
    sell_signals: int
    hold_signals: int


# API Endpoints

@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics() -> SystemMetrics:
    """
    Get system-level metrics.
    
    Returns:
        SystemMetrics: Overall system statistics
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        
        if not collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available"
            )
        
        stats = collector.get_system_stats()
        return SystemMetrics(**stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}", response_model=ModelMetrics)
async def get_model_metrics(model_name: str) -> ModelMetrics:
    """
    Get metrics for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelMetrics: Model statistics
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        
        if not collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available"
            )
        
        stats = collector.get_model_stats(model_name)
        return ModelMetrics(model_name=model_name, **stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/recent")
async def get_recent_predictions(limit: int = 100) -> Dict[str, Any]:
    """
    Get recent predictions.
    
    Args:
        limit: Maximum number of predictions to return
        
    Returns:
        dict: Recent predictions
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        
        if not collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available"
            )
        
        # Get recent predictions
        recent = list(collector.predictions)[-limit:]
        
        return {
            "count": len(recent),
            "predictions": recent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recent predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors/recent")
async def get_recent_errors(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent errors.
    
    Args:
        limit: Maximum number of errors to return
        
    Returns:
        dict: Recent errors
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        
        if not collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available"
            )
        
        # Get recent errors
        recent = list(collector.errors)[-limit:]
        
        return {
            "count": len(recent),
            "errors": recent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """
    Get comprehensive dashboard data.
    
    Returns:
        dict: Dashboard metrics and statistics
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        models = app_state.get("models", {})
        
        if not collector:
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {},
                "models": {},
                "recent_predictions": [],
                "recent_errors": [],
                "active_models": 0,
                "available_models": [],
            }
        
        # Get system stats
        system_stats = collector.get_system_stats()
        
        # Get model stats
        model_stats = {}
        for model_name in models.keys():
            model_stats[model_name] = collector.get_model_stats(model_name)
        
        # Recent activity
        recent_predictions = list(collector.predictions)[-10:]
        recent_errors = list(collector.errors)[-5:]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": system_stats,
            "models": model_stats,
            "recent_predictions": recent_predictions,
            "recent_errors": recent_errors,
            "active_models": len(models),
            "available_models": list(models.keys())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "models": {},
            "recent_predictions": [],
            "recent_errors": [],
            "active_models": 0,
            "available_models": [],
        }


@router.post("/save")
async def save_metrics() -> Dict[str, str]:
    """
    Save current metrics to disk.
    
    Returns:
        dict: Save status
    """
    try:
        app_state = get_app_state()
        collector: MetricsCollector = app_state.get("metrics_collector")
        
        if not collector:
            raise HTTPException(
                status_code=503,
                detail="Metrics collector not available"
            )
        
        collector.save_metrics()
        
        return {
            "status": "success",
            "message": "Metrics saved successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_metrics_history() -> Dict[str, Any]:
    """
    Get list of saved metrics files.
    
    Returns:
        dict: Available metrics files
    """
    try:
        files = []
        
        for filepath in METRICS_DIR.glob("metrics_*.json"):
            stat = filepath.stat()
            files.append({
                "filename": filepath.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by modified time
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "count": len(files),
            "files": files
        }
    
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
