"""
Paper Trading API Endpoints

Handles paper trading operations:
- Order placement and management
- Position tracking
- Portfolio monitoring
- Trade execution based on signals
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Paper Trading"])


# Request/Response Models
class ExecuteSignalRequest(BaseModel):
    symbol: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    current_price: float
    model_name: Optional[str] = None


class PlaceOrderRequest(BaseModel):
    symbol: str
    quantity: float
    side: str  # "buy" or "sell"
    order_type: str = "market"  # "market", "limit", "stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class OrderResponse(BaseModel):
    order_id: Optional[str]
    symbol: str
    quantity: float
    side: str
    status: str
    timestamp: datetime
    message: Optional[str] = None


class PositionData(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioResponse(BaseModel):
    timestamp: datetime
    cash: float
    portfolio_value: float
    total_value: float
    positions: List[PositionData]
    total_pnl: float
    total_pnl_pct: float
    num_positions: int


class OrderHistoryResponse(BaseModel):
    order_id: str
    symbol: str
    quantity: float
    side: str
    status: str
    timestamp: datetime
    filled_price: Optional[float] = None


# Global paper trading engine (lazy loaded)
_trading_engine = None


def get_trading_engine():
    """Get or initialize paper trading engine."""
    global _trading_engine
    if _trading_engine is None:
        from core.paper_trading import AlpacaAdapter, PaperTradingEngine
        import os
        
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_API_SECRET", "")
        base_url = os.getenv("ALPACA_API_BASE", "https://paper-api.alpaca.markets")
        
        if not api_key or not api_secret:
            logger.warning("Alpaca credentials not configured for paper trading")
            return None
        
        adapter = AlpacaAdapter(api_key, api_secret, base_url)
        _trading_engine = PaperTradingEngine(adapter)
    
    return _trading_engine


@router.post("/initialize")
async def initialize_trading():
    """Initialize connection to paper trading broker."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading not configured. Set ALPACA_API_KEY and ALPACA_API_SECRET"
            )
        
        if await engine.initialize():
            return {
                "status": "initialized",
                "message": "Paper trading engine connected",
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize paper trading engine"
            )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-signal")
async def execute_signal(request: ExecuteSignalRequest):
    """Execute trade based on ML model signal."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        order_id = await engine.execute_signal(
            symbol=request.symbol,
            signal=request.signal,
            confidence=request.confidence,
            current_price=request.current_price
        )
        
        return {
            "status": "executed" if order_id else "hold",
            "order_id": order_id,
            "symbol": request.symbol,
            "signal": request.signal,
            "confidence": request.confidence,
            "model": request.model_name,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Failed to execute signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders/place")
async def place_order(request: PlaceOrderRequest) -> OrderResponse:
    """Place a manual trading order."""
    try:
        from core.paper_trading import Order
        
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        order = Order(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price
        )
        
        order_id = await engine.broker.place_order(order)
        
        return OrderResponse(
            order_id=order_id,
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            status="submitted" if order_id else "failed",
            timestamp=datetime.now(),
            message="Order placed successfully" if order_id else "Failed to place order"
        )
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an open order."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        if await engine.broker.cancel_order(order_id):
            return {
                "status": "cancelled",
                "order_id": order_id,
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to cancel order"
            )
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio() -> PortfolioResponse:
    """Get current portfolio state."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        summary = await engine.get_portfolio_summary()
        
        positions = [
            PositionData(
                symbol=symbol,
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                current_price=pos_data["current_price"],
                unrealized_pnl=pos_data["unrealized_pnl"],
                unrealized_pnl_pct=pos_data["unrealized_pnl_pct"]
            )
            for symbol, pos_data in summary.get("positions", {}).items()
        ]
        
        return PortfolioResponse(
            timestamp=datetime.fromisoformat(summary["timestamp"]),
            cash=summary["cash"],
            portfolio_value=summary["portfolio_value"],
            total_value=summary["total_value"],
            positions=positions,
            total_pnl=summary["total_pnl"],
            total_pnl_pct=summary["total_pnl_pct"],
            num_positions=summary["num_positions"]
        )
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions():
    """Get all open positions."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        positions = await engine.broker.get_positions()
        
        return {
            "timestamp": datetime.now(),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct
                }
                for symbol, pos in positions.items()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_orders(status: str = "all"):
    """Get order history."""
    try:
        engine = get_trading_engine()
        if not engine:
            raise HTTPException(
                status_code=400,
                detail="Paper trading engine not initialized"
            )
        
        orders = await engine.broker.get_orders(status)
        
        return {
            "timestamp": datetime.now(),
            "status": status,
            "orders": [
                OrderHistoryResponse(
                    order_id=order.broker_order_id or "N/A",
                    symbol=order.symbol,
                    quantity=order.quantity,
                    side=order.side,
                    status=order.status,
                    timestamp=order.timestamp
                )
                for order in orders
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shutdown")
async def shutdown_trading():
    """Shutdown paper trading engine."""
    try:
        global _trading_engine
        
        engine = get_trading_engine()
        if not engine:
            return {"status": "not_initialized"}
        
        if await engine.shutdown():
            _trading_engine = None
            return {
                "status": "shutdown",
                "message": "Paper trading engine shut down",
                "timestamp": datetime.now()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to shutdown engine"
            )
    except Exception as e:
        logger.error(f"Failed to shutdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def paper_trading_health():
    """Check paper trading engine health."""
    try:
        engine = get_trading_engine()
        
        if not engine:
            return {
                "status": "not_initialized",
                "configured": False
            }
        
        account = engine.account
        return {
            "status": "healthy",
            "configured": True,
            "connected": account is not None,
            "cash": account.cash if account else None,
            "portfolio_value": account.portfolio_value if account else None,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
