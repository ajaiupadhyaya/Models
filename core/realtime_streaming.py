"""
Real-time Data Streaming System
WebSocket-based live market data streaming
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from core.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class StreamMessage:
    """Message for real-time streaming."""
    type: str  # price, signal, trade, alert
    symbol: str
    data: Dict
    timestamp: datetime


class RealTimeDataStreamer:
    """
    Real-time data streaming system.
    Provides live market data updates via callbacks.
    """
    
    def __init__(self, symbols: List[str], update_interval: float = 1.0):
        """
        Initialize real-time streamer.
        
        Args:
            symbols: List of symbols to stream
            update_interval: Update interval in seconds
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.data_fetcher = DataFetcher()
        self.subscribers: List[Callable] = []
        self.is_running = False
        self._task = None
        
        # Cache for latest data
        self.latest_data = {}
        self.latest_signals = {}
    
    def subscribe(self, callback: Callable[[StreamMessage], None]):
        """
        Subscribe to stream updates.
        
        Args:
            callback: Function to call with StreamMessage
        """
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from stream."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, message: StreamMessage):
        """Notify all subscribers."""
        for callback in self.subscribers:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")
    
    async def _stream_loop(self):
        """Main streaming loop."""
        logger.info("Starting real-time data stream...")
        
        while self.is_running:
            try:
                # Fetch latest data for all symbols
                for symbol in self.symbols:
                    try:
                        # Get latest price data
                        df = self.data_fetcher.get_stock_data(symbol, period="1d")
                        if df is not None and len(df) > 0:
                            current_price = df['Close'].iloc[-1]
                            volume = df['Volume'].iloc[-1]
                            
                            # Check if price changed
                            if symbol not in self.latest_data or \
                               self.latest_data[symbol].get('price') != current_price:
                                
                                prev_price = self.latest_data.get(symbol, {}).get('price', current_price)
                                change = current_price - prev_price
                                change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                                
                                message = StreamMessage(
                                    type="price",
                                    symbol=symbol,
                                    data={
                                        'price': float(current_price),
                                        'volume': int(volume),
                                        'change': float(change),
                                        'change_pct': float(change_pct),
                                        'timestamp': datetime.now().isoformat()
                                    },
                                    timestamp=datetime.now()
                                )
                                
                                self._notify_subscribers(message)
                                self.latest_data[symbol] = {
                                    'price': current_price,
                                    'volume': volume,
                                    'timestamp': datetime.now()
                                }
                    
                    except Exception as e:
                        logger.warning(f"Error streaming {symbol}: {e}")
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def start(self):
        """Start streaming."""
        if self.is_running:
            logger.warning("Stream already running")
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info("Real-time streaming started")
    
    def stop(self):
        """Stop streaming."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info("Real-time streaming stopped")
    
    def stream_signal(self, symbol: str, signal_data: Dict):
        """
        Stream a trading signal.
        
        Args:
            symbol: Stock symbol
            signal_data: Signal data dictionary
        """
        message = StreamMessage(
            type="signal",
            symbol=symbol,
            data=signal_data,
            timestamp=datetime.now()
        )
        self._notify_subscribers(message)
        self.latest_signals[symbol] = signal_data
    
    def stream_trade(self, symbol: str, trade_data: Dict):
        """
        Stream a trade execution.
        
        Args:
            symbol: Stock symbol
            trade_data: Trade data dictionary
        """
        message = StreamMessage(
            type="trade",
            symbol=symbol,
            data=trade_data,
            timestamp=datetime.now()
        )
        self._notify_subscribers(message)
    
    def stream_alert(self, alert_type: str, message: str, severity: str = "info"):
        """
        Stream an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: info, warning, error
        """
        alert_message = StreamMessage(
            type="alert",
            symbol="SYSTEM",
            data={
                'alert_type': alert_type,
                'message': message,
                'severity': severity
            },
            timestamp=datetime.now()
        )
        self._notify_subscribers(alert_message)


class WebSocketStreamAdapter:
    """
    Adapter for WebSocket connections.
    Converts StreamMessage to WebSocket format.
    """
    
    def __init__(self, websocket):
        """
        Initialize adapter.
        
        Args:
            websocket: WebSocket connection
        """
        self.websocket = websocket
        self.streamer = None
    
    async def handle_message(self, message: StreamMessage):
        """Handle stream message and send via WebSocket."""
        try:
            data = {
                'type': message.type,
                'symbol': message.symbol,
                'data': message.data,
                'timestamp': message.timestamp.isoformat()
            }
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
    
    def connect_streamer(self, streamer: RealTimeDataStreamer):
        """Connect to streamer."""
        self.streamer = streamer
        streamer.subscribe(self.handle_message)
    
    def disconnect(self):
        """Disconnect from streamer."""
        if self.streamer:
            self.streamer.unsubscribe(self.handle_message)
