"""
WebSocket API

Real-time streaming endpoints:
- Live price updates
- Streaming predictions
- Real-time signal updates
- Portfolio monitoring
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Set
import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def disconnect_all(self) -> None:
        """Disconnect all WebSocket connections."""
        for connection in self.active_connections[:]:
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self.active_connections.clear()
        self.subscriptions.clear()
        logger.info("All WebSocket connections closed")
    
    def subscribe(self, websocket: WebSocket, symbol: str) -> None:
        """
        Subscribe a connection to a symbol.
        
        Args:
            websocket: WebSocket connection
            symbol: Stock symbol to subscribe to
        """
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(symbol)
            logger.info(f"Client subscribed to {symbol}")
    
    def unsubscribe(self, websocket: WebSocket, symbol: str) -> None:
        """
        Unsubscribe a connection from a symbol.
        
        Args:
            websocket: WebSocket connection
            symbol: Stock symbol to unsubscribe from
        """
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(symbol)
            logger.info(f"Client unsubscribed from {symbol}")
    
    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket
    ) -> None:
        """
        Send a message to a specific connection.
        
        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any], symbol: str = None) -> None:
        """
        Broadcast a message to all subscribed connections.
        
        Args:
            message: Message to broadcast
            symbol: If provided, only send to subscribers of this symbol
        """
        for connection in self.active_connections[:]:
            try:
                # If symbol specified, only send to subscribers
                if symbol:
                    if connection in self.subscriptions:
                        if symbol in self.subscriptions[connection]:
                            await connection.send_json(message)
                else:
                    await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast: {e}")
                self.disconnect(connection)


def get_app_state() -> Dict[str, Any]:
    """Get global app state."""
    from api.main import get_app_state
    return get_app_state()


async def stream_prices(
    websocket: WebSocket,
    symbol: str,
    interval: int = 5
) -> None:
    """
    Stream real-time price updates.
    
    Args:
        websocket: WebSocket connection
        symbol: Stock symbol
        interval: Update interval in seconds
    """
    try:
        while True:
            # Fetch latest price
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            # Send update
            message = {
                "type": "price_update",
                "symbol": symbol,
                "price": current_price,
                "timestamp": datetime.now().isoformat(),
                "volume": info.get('volume', 0),
                "market_cap": info.get('marketCap', 0)
            }
            
            await websocket.send_json(message)
            
            # Wait for next update
            await asyncio.sleep(interval)
    
    except WebSocketDisconnect:
        logger.info(f"Price stream disconnected for {symbol}")
    except Exception as e:
        logger.error(f"Price streaming error: {e}")


async def stream_predictions(
    websocket: WebSocket,
    model_name: str,
    symbol: str,
    interval: int = 60
) -> None:
    """
    Stream real-time predictions.
    
    Args:
        websocket: WebSocket connection
        model_name: Model to use for predictions
        symbol: Stock symbol
        interval: Update interval in seconds
    """
    try:
        from api.predictions_api import predict, PredictionRequest
        
        while True:
            try:
                # Generate prediction
                request = PredictionRequest(
                    model_name=model_name,
                    symbol=symbol,
                    days_lookback=60
                )
                
                prediction = await predict(request)
                
                # Send update
                message = {
                    "type": "prediction_update",
                    "model_name": model_name,
                    "symbol": symbol,
                    "signal": prediction.signal,
                    "confidence": prediction.confidence,
                    "recommendation": prediction.recommendation,
                    "current_price": prediction.current_price,
                    "timestamp": prediction.timestamp
                }
                
                await websocket.send_json(message)
                
            except Exception as e:
                logger.error(f"Prediction generation error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait for next update
            await asyncio.sleep(interval)
    
    except WebSocketDisconnect:
        logger.info(f"Prediction stream disconnected for {symbol}")
    except Exception as e:
        logger.error(f"Prediction streaming error: {e}")


# WebSocket Endpoints

@router.websocket("/prices/{symbol}")
async def websocket_prices(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time price streaming.
    
    Args:
        websocket: WebSocket connection
        symbol: Stock symbol to stream
    """
    app_state = get_app_state()
    manager: ConnectionManager = app_state.get("connection_manager")
    
    if not manager:
        await websocket.close(code=1011, reason="Connection manager not available")
        return
    
    await manager.connect(websocket)
    manager.subscribe(websocket, symbol)
    
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "symbol": symbol,
            "message": f"Streaming prices for {symbol}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start streaming
        await stream_prices(websocket, symbol)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/predictions/{model_name}/{symbol}")
async def websocket_predictions(
    websocket: WebSocket,
    model_name: str,
    symbol: str
):
    """
    WebSocket endpoint for real-time prediction streaming.
    
    Args:
        websocket: WebSocket connection
        model_name: Model to use for predictions
        symbol: Stock symbol
    """
    app_state = get_app_state()
    manager: ConnectionManager = app_state.get("connection_manager")
    
    if not manager:
        await websocket.close(code=1011, reason="Connection manager not available")
        return
    
    # Check if model exists
    models = app_state.get("models", {})
    if model_name not in models:
        await websocket.close(code=1008, reason=f"Model {model_name} not found")
        return
    
    await manager.connect(websocket)
    manager.subscribe(websocket, symbol)
    
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "model_name": model_name,
            "symbol": symbol,
            "message": f"Streaming predictions for {symbol}",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start streaming
        await stream_predictions(websocket, model_name, symbol)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/live")
async def websocket_live(websocket: WebSocket):
    """
    General-purpose WebSocket endpoint with command support.
    
    Supported commands:
    - {"action": "subscribe", "symbol": "SPY"}
    - {"action": "unsubscribe", "symbol": "SPY"}
    - {"action": "predict", "model": "model_name", "symbol": "SPY"}
    """
    app_state = get_app_state()
    manager: ConnectionManager = app_state.get("connection_manager")
    
    if not manager:
        await websocket.close(code=1011, reason="Connection manager not available")
        return
    
    await manager.connect(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to live feed",
            "timestamp": datetime.now().isoformat()
        })
        
        # Listen for commands
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "subscribe":
                symbol = data.get("symbol")
                if symbol:
                    manager.subscribe(websocket, symbol)
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif action == "unsubscribe":
                symbol = data.get("symbol")
                if symbol:
                    manager.unsubscribe(websocket, symbol)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif action == "predict":
                model_name = data.get("model")
                symbol = data.get("symbol")
                
                if model_name and symbol:
                    from api.predictions_api import predict, PredictionRequest
                    
                    try:
                        request = PredictionRequest(
                            model_name=model_name,
                            symbol=symbol
                        )
                        prediction = await predict(request)
                        
                        await websocket.send_json({
                            "type": "prediction",
                            "model_name": model_name,
                            "symbol": symbol,
                            "signal": prediction.signal,
                            "confidence": prediction.confidence,
                            "recommendation": prediction.recommendation,
                            "timestamp": prediction.timestamp
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.now().isoformat()
                        })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
