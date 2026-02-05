"""
Real-Time Predictions API

Endpoints for generating predictions from trained models:
- Single prediction
- Batch predictions
- Live streaming predictions
- Multi-model ensemble predictions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf
import pandas as pd
import numpy as np

# Import BacktestSignal separately to avoid cascade
import importlib.util
spec = importlib.util.spec_from_file_location(
    "backtesting",
    project_root / "core" / "backtesting.py"
)
backtesting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtesting_module)

BacktestSignal = backtesting_module.BacktestSignal

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request for a prediction."""
    model_name: str = Field(description="Name of the model to use")
    symbol: str = Field(description="Stock symbol", example="SPY")
    days_lookback: int = Field(
        default=60,
        description="Number of historical days to use"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    model_name: str = Field(description="Name of the model to use")
    symbols: List[str] = Field(description="List of symbols", example=["SPY", "QQQ", "IWM"])
    days_lookback: int = Field(default=60, description="Historical days")


class EnsemblePredictionRequest(BaseModel):
    """Request for ensemble prediction from multiple models."""
    model_names: List[str] = Field(description="List of model names")
    symbol: str = Field(description="Stock symbol")
    weights: Optional[List[float]] = Field(
        None,
        description="Weights for each model (must sum to 1)"
    )
    days_lookback: int = Field(default=60, description="Historical days")


class PredictionResponse(BaseModel):
    """Response with prediction."""
    model_name: str
    symbol: str
    timestamp: str
    signal: float = Field(description="Signal strength (-1 to 1)")
    confidence: float = Field(description="Confidence (0 to 1)")
    current_price: float
    recommendation: Literal["BUY", "SELL", "HOLD"]
    metadata: Dict[str, Any]


class BatchPredictionResponse(BaseModel):
    """Response with batch predictions."""
    model_name: str
    timestamp: str
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]


class EnsemblePredictionResponse(BaseModel):
    """Response with ensemble prediction."""
    models_used: List[str]
    symbol: str
    timestamp: str
    ensemble_signal: float
    ensemble_confidence: float
    individual_predictions: List[Dict[str, Any]]
    recommendation: Literal["BUY", "SELL", "HOLD"]


# Helper functions
def get_app_state() -> Dict[str, Any]:
    """Get global app state."""
    from api.main import get_app_state
    return get_app_state()


def get_recommendation(signal: float, confidence: float) -> str:
    """
    Convert signal and confidence to recommendation.
    
    Args:
        signal: Signal strength (-1 to 1)
        confidence: Confidence (0 to 1)
        
    Returns:
        str: BUY, SELL, or HOLD
    """
    threshold = 0.2
    min_confidence = 0.5
    
    if confidence < min_confidence:
        return "HOLD"
    
    if signal > threshold:
        return "BUY"
    elif signal < -threshold:
        return "SELL"
    else:
        return "HOLD"


def fetch_recent_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Fetch recent market data.
    
    Args:
        symbol: Stock symbol
        days: Number of days to fetch
        
    Returns:
        DataFrame: OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 10)  # Extra buffer
    
    data = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False
    )
    
    if data.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Flatten multi-level column index if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data


# API Endpoints

@router.get("/quick-predict")
async def quick_predict(symbol: str = "AAPL", days_lookback: int = 60) -> Dict[str, Any]:
    """
    On-the-fly ML prediction without pre-loaded models.
    Trains EnsemblePredictor on recent data and returns signal and recommendation.
    Used by the terminal to show ML/DL capability without requiring model training first.
    """
    try:
        from models.ml.advanced_trading import EnsemblePredictor
        data = fetch_recent_data(symbol, days_lookback)
        if data.empty or len(data) < 20:
            return {"symbol": symbol, "error": "Insufficient data", "signal": 0.0, "recommendation": "HOLD"}
        model = EnsemblePredictor(lookback_window=20)
        model.train(data)
        pred = model.predict(data)
        signal_val = float(pred[-1]) if len(pred) else 0.0
        rec = get_recommendation(signal_val, 0.65)
        current_price = float(data["Close"].iloc[-1])
        return {
            "symbol": symbol,
            "signal": signal_val,
            "recommendation": rec,
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "model": "ensemble_on_the_fly",
        }
    except Exception as e:
        logger.warning(f"Quick predict failed for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e), "signal": 0.0, "recommendation": "HOLD"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate a prediction for a symbol.
    
    Args:
        request: Prediction request
        
    Returns:
        PredictionResponse: Prediction with signal and recommendation
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
        
        # Fetch data
        logger.info(f"Fetching data for {request.symbol}")
        data = fetch_recent_data(request.symbol, request.days_lookback)
        
        # Generate prediction
        model_type = metadata.get("type", "unknown")
        
        if model_type == "simple":
            # Simple predictor returns BacktestSignal
            signal_obj = model.predict(data)
            signal = signal_obj.signal
            confidence = signal_obj.confidence
            
        elif model_type == "ensemble":
            # Ensemble predictor
            features = model.calculate_features(data)
            features = features.dropna()
            
            if len(features) == 0:
                raise ValueError("No valid features calculated")
            
            # Get latest prediction
            latest_features = features.iloc[-1:]
            signal = model.predict(latest_features)
            confidence = 0.7  # Default confidence for ensemble
            
        elif model_type == "lstm":
            # LSTM predictor
            X, _ = model.prepare_data(data)
            
            if len(X) == 0:
                raise ValueError("No valid sequences for LSTM")
            
            # Get latest prediction
            latest_seq = X[-1:]
            signal = model.predict(latest_seq)
            confidence = 0.8  # Default confidence for LSTM
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get current price
        current_price = float(data['Close'].iloc[-1])
        
        # Generate recommendation
        recommendation = get_recommendation(signal, confidence)
        
        # Record metrics
        metrics_collector = app_state.get("metrics_collector")
        if metrics_collector:
            metrics_collector.record_prediction(
                model_name=request.model_name,
                symbol=request.symbol,
                signal=signal,
                confidence=confidence
            )
        
        return PredictionResponse(
            model_name=request.model_name,
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            signal=float(signal),
            confidence=float(confidence),
            current_price=current_price,
            recommendation=recommendation,
            metadata={
                "model_type": model_type,
                "data_points": len(data),
                "latest_date": data.index[-1].strftime("%Y-%m-%d")
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate predictions for multiple symbols.
    
    Args:
        request: Batch prediction request
        
    Returns:
        BatchPredictionResponse: Predictions for all symbols
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
        
        predictions = []
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
        # Generate predictions for each symbol
        for symbol in request.symbols:
            try:
                # Create individual request
                pred_request = PredictionRequest(
                    model_name=request.model_name,
                    symbol=symbol,
                    days_lookback=request.days_lookback
                )
                
                # Get prediction
                pred = await predict(pred_request)
                
                predictions.append({
                    "symbol": symbol,
                    "signal": pred.signal,
                    "confidence": pred.confidence,
                    "recommendation": pred.recommendation,
                    "current_price": pred.current_price
                })
                
                # Update counts
                if pred.recommendation == "BUY":
                    buy_signals += 1
                elif pred.recommendation == "SELL":
                    sell_signals += 1
                else:
                    hold_signals += 1
                
            except Exception as e:
                logger.warning(f"Failed to predict {symbol}: {e}")
                predictions.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        # Summary
        summary = {
            "total": len(request.symbols),
            "successful": len([p for p in predictions if "error" not in p]),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals
        }
        
        return BatchPredictionResponse(
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            summary=summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: EnsemblePredictionRequest) -> EnsemblePredictionResponse:
    """
    Generate ensemble prediction from multiple models.
    
    Args:
        request: Ensemble prediction request
        
    Returns:
        EnsemblePredictionResponse: Combined prediction
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Validate models exist
        for model_name in request.model_names:
            if model_name not in models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {model_name} not found"
                )
        
        # Set default weights
        if request.weights is None:
            weights = [1.0 / len(request.model_names)] * len(request.model_names)
        else:
            weights = request.weights
            
            # Validate weights
            if len(weights) != len(request.model_names):
                raise HTTPException(
                    status_code=400,
                    detail="Number of weights must match number of models"
                )
            
            if not np.isclose(sum(weights), 1.0):
                raise HTTPException(
                    status_code=400,
                    detail="Weights must sum to 1.0"
                )
        
        # Get predictions from each model
        individual_predictions = []
        signals = []
        confidences = []
        
        for model_name, weight in zip(request.model_names, weights):
            try:
                # Create individual request
                pred_request = PredictionRequest(
                    model_name=model_name,
                    symbol=request.symbol,
                    days_lookback=request.days_lookback
                )
                
                # Get prediction
                pred = await predict(pred_request)
                
                individual_predictions.append({
                    "model_name": model_name,
                    "signal": pred.signal,
                    "confidence": pred.confidence,
                    "weight": weight,
                    "weighted_signal": pred.signal * weight
                })
                
                signals.append(pred.signal * weight)
                confidences.append(pred.confidence * weight)
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                individual_predictions.append({
                    "model_name": model_name,
                    "error": str(e),
                    "weight": weight
                })
        
        # Calculate ensemble signal
        ensemble_signal = float(sum(signals))
        ensemble_confidence = float(sum(confidences))
        
        # Generate recommendation
        recommendation = get_recommendation(ensemble_signal, ensemble_confidence)
        
        return EnsemblePredictionResponse(
            models_used=request.model_names,
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            individual_predictions=individual_predictions,
            recommendation=recommendation
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/signals/{symbol}")
async def get_historical_signals(
    model_name: str,
    symbol: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get historical signals from a model.
    
    Args:
        model_name: Name of the model
        symbol: Stock symbol
        days: Number of days to generate signals for
        
    Returns:
        dict: Historical signals and metadata
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Get model
        if model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
        
        model_data = models[model_name]
        model = model_data["model"]
        
        # Fetch data
        data = fetch_recent_data(symbol, days + 60)  # Extra for indicators
        
        # Generate signals for each day
        signals = []
        dates = []
        prices = []
        
        # Calculate signals for recent period
        for i in range(len(data) - days, len(data)):
            historical_data = data.iloc[:i+1]
            
            try:
                # Generate signal
                model_type = model_data["metadata"].get("type")
                
                if model_type == "simple":
                    signal_obj = model.predict(historical_data)
                    signal = signal_obj.signal
                elif model_type == "ensemble":
                    features = model.calculate_features(historical_data)
                    features = features.dropna()
                    if len(features) > 0:
                        signal = model.predict(features.iloc[-1:])
                    else:
                        signal = 0.0
                elif model_type == "lstm":
                    X, _ = model.prepare_data(historical_data)
                    if len(X) > 0:
                        signal = model.predict(X[-1:])
                    else:
                        signal = 0.0
                else:
                    signal = 0.0
                
                signals.append(float(signal))
                dates.append(data.index[i].strftime("%Y-%m-%d"))
                prices.append(float(data['Close'].iloc[i]))
                
            except Exception as e:
                logger.warning(f"Failed to generate signal for {dates[-1] if dates else 'date'}: {e}")
                continue
        
        return {
            "model_name": model_name,
            "symbol": symbol,
            "period": f"{days} days",
            "data_points": len(signals),
            "dates": dates,
            "signals": signals,
            "prices": prices,
            "current_signal": signals[-1] if signals else 0.0,
            "avg_signal": float(np.mean(signals)) if signals else 0.0,
            "signal_std": float(np.std(signals)) if signals else 0.0
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical signals failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
