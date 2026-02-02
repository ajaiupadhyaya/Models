"""
Model Management API

Endpoints for managing ML models:
- List available models
- Load/unload models
- Train new models
- Get model information
- Save/load model state
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import specific modules avoiding cascade
import importlib.util

# Load advanced_trading module directly
spec = importlib.util.spec_from_file_location(
    "advanced_trading",
    project_root / "models" / "ml" / "advanced_trading.py"
)
advanced_trading = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advanced_trading)

LSTMPredictor = advanced_trading.LSTMPredictor
EnsemblePredictor = advanced_trading.EnsemblePredictor

# Load backtesting module directly
spec = importlib.util.spec_from_file_location(
    "backtesting",
    project_root / "core" / "backtesting.py"
)
backtesting_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtesting_module)

SimpleMLPredictor = backtesting_module.SimpleMLPredictor
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

# Model storage directory
MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Request/Response Models
class TrainModelRequest(BaseModel):
    """Request to train a new model."""
    model_type: Literal["simple", "ensemble", "lstm"] = Field(
        description="Type of model to train"
    )
    model_name: str = Field(description="Unique name for the model")
    symbol: str = Field(description="Stock symbol to train on", example="SPY")
    start_date: str = Field(description="Start date YYYY-MM-DD", example="2022-01-01")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Model-specific parameters"
    )


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    type: str
    symbol: str
    trained_date: Optional[str]
    parameters: Dict[str, Any]
    status: str


class ModelListResponse(BaseModel):
    """Response with list of models."""
    models: List[ModelInfo]
    count: int


class TrainResponse(BaseModel):
    """Response after training."""
    status: str
    model_name: str
    message: str
    training_time: Optional[float]


class PredictionRequest(BaseModel):
    """Request for prediction."""
    model_name: str
    symbol: str
    days: int = Field(default=1, description="Number of days to predict")


# Helper functions
def get_app_state() -> Dict[str, Any]:
    """Get global app state."""
    from api.main import get_app_state
    return get_app_state()


def save_model_metadata(model_name: str, metadata: Dict[str, Any]) -> None:
    """Save model metadata to JSON."""
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata for {model_name}")


def load_model_metadata(model_name: str) -> Optional[Dict[str, Any]]:
    """Load model metadata from JSON."""
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


async def load_saved_models() -> Dict[str, Any]:
    """Load all saved models from disk."""
    loaded = {}

    try:
        for metadata_file in MODELS_DIR.glob("*_metadata.json"):
            model_name = metadata_file.stem.replace("_metadata", "")
            metadata = load_model_metadata(model_name)
            
            if metadata:
                model_type = metadata.get("type")
                model_path = MODELS_DIR / f"{model_name}.pkl"
                
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        
                        loaded[model_name] = {
                            "model": model,
                            "metadata": metadata,
                            "loaded_at": datetime.now().isoformat()
                        }
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {e}")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    return loaded


# API Endpoints

@router.get("/", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    List all available models.
    
    Returns:
        ModelListResponse: List of models with metadata
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        model_list = []
        for name, data in models.items():
            metadata = data.get("metadata", {})
            model_list.append(ModelInfo(
                name=name,
                type=metadata.get("type", "unknown"),
                symbol=metadata.get("symbol", "unknown"),
                trained_date=metadata.get("trained_date"),
                parameters=metadata.get("parameters", {}),
                status="loaded"
            ))
        
        return ModelListResponse(models=model_list, count=len(model_list))
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return ModelListResponse(models=[], count=0)


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelInfo: Model details
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        data = models[model_name]
        metadata = data.get("metadata", {})
        
        return ModelInfo(
            name=model_name,
            type=metadata.get("type", "unknown"),
            symbol=metadata.get("symbol", "unknown"),
            trained_date=metadata.get("trained_date"),
            parameters=metadata.get("parameters", {}),
            status="loaded"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    """
    Train a new model.
    
    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        TrainResponse: Training status
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Check if model already exists
        if request.model_name in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model_name} already exists"
            )
        
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
        
        # Create and train model
        start_time = datetime.now()
        
        if request.model_type == "simple":
            model = SimpleMLPredictor()
            # Simple model doesn't need training
            
        elif request.model_type == "ensemble":
            model = EnsemblePredictor()
            # Prepare features
            features_df = model.calculate_features(data)
            features_df = features_df.dropna()
            
            # Train on 80% of data
            train_size = int(len(features_df) * 0.8)
            train_data = features_df.iloc[:train_size]
            
            model.train(train_data)
            
        elif request.model_type == "lstm":
            model = LSTMPredictor()
            # Prepare data
            X, y = model.prepare_data(data)
            
            # Train model
            model.train(X, y, epochs=50, batch_size=32)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {request.model_type}"
            )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save model
        model_path = MODELS_DIR / f"{request.model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "type": request.model_type,
            "symbol": request.symbol,
            "trained_date": datetime.now().isoformat(),
            "parameters": request.parameters or {},
            "training_time": training_time,
            "data_range": {
                "start": request.start_date,
                "end": request.end_date or "now"
            }
        }
        save_model_metadata(request.model_name, metadata)
        
        # Add to app state
        models[request.model_name] = {
            "model": model,
            "metadata": metadata,
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model {request.model_name} trained successfully")
        
        return TrainResponse(
            status="success",
            model_name=request.model_name,
            message=f"Model trained in {training_time:.2f}s",
            training_time=training_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_name}")
async def delete_model(model_name: str) -> Dict[str, str]:
    """
    Delete a model.
    
    Args:
        model_name: Name of the model to delete
        
    Returns:
        dict: Deletion status
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Remove from memory
        del models[model_name]
        
        # Delete files
        model_path = MODELS_DIR / f"{model_name}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        logger.info(f"Model {model_name} deleted")
        
        return {"status": "success", "message": f"Model {model_name} deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/reload")
async def reload_model(model_name: str) -> Dict[str, str]:
    """
    Reload a model from disk.
    
    Args:
        model_name: Name of the model to reload
        
    Returns:
        dict: Reload status
    """
    try:
        app_state = get_app_state()
        models = app_state.get("models", {})
        
        # Load metadata
        metadata = load_model_metadata(model_name)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
        
        # Load model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_name}"
            )
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Update app state
        models[model_name] = {
            "model": model,
            "metadata": metadata,
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model {model_name} reloaded")
        
        return {"status": "success", "message": f"Model {model_name} reloaded"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
