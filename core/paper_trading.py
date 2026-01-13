"""
Paper Trading Integration Framework

Supports multiple brokers for live paper trading:
- Alpaca (recommended for simplicity)
- TD Ameritrade/Schwab
- Interactive Brokers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    quantity: float
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    broker_order_id: Optional[str] = None
    status: str = "pending"  # pending, filled, cancelled, error


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_timestamp: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update position with current price."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = (price / self.entry_price - 1) * 100


@dataclass
class Account:
    """Represents trading account state."""
    cash: float
    portfolio_value: float
    buying_power: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def total_value(self) -> float:
        """Total account value."""
        positions_value = sum(p.quantity * p.current_price for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def margin_used(self) -> float:
        """Calculate margin used."""
        return max(0, self.portfolio_value - self.cash)


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        """
        Initialize broker adapter.
        
        Args:
            api_key: Broker API key
            api_secret: Broker API secret
            base_url: API base URL
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    async def get_account(self) -> Account:
        """Get current account state."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Optional[str]:
        """
        Place an order.
        
        Returns:
            Order ID if successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    async def get_orders(self, status: str = "all") -> List[Order]:
        """Get orders by status."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current quote for symbol.
        
        Returns:
            Dict with 'bid', 'ask', 'last' prices
        """
        pass


class AlpacaAdapter(BrokerAdapter):
    """Alpaca broker adapter."""
    
    async def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url
            )
            self.logger.info("Connected to Alpaca")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Alpaca."""
        try:
            if hasattr(self, 'api'):
                del self.api
            self.logger.info("Disconnected from Alpaca")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect: {e}")
            return False
    
    async def get_account(self) -> Account:
        """Get Alpaca account state."""
        try:
            account = self.api.get_account()
            
            return Account(
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                buying_power=float(account.buying_power)
            )
        except Exception as e:
            self.logger.error(f"Failed to get account: {e}")
            return None
    
    async def place_order(self, order: Order) -> Optional[str]:
        """Place order on Alpaca."""
        try:
            result = self.api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side,
                type=order.order_type,
                time_in_force="day",
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )
            
            order.broker_order_id = result.id
            order.status = "submitted"
            
            self.logger.info(f"Order placed: {result.id} for {order.symbol}")
            return result.id
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            order.status = "error"
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get positions from Alpaca."""
        try:
            positions_data = self.api.list_positions()
            positions = {}
            
            for pos in positions_data:
                positions[pos.symbol] = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_fill_price),
                    current_price=float(pos.current_price),
                    entry_timestamp=datetime.fromisoformat(pos.created_at),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc) * 100
                )
            
            return positions
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return {}
    
    async def get_orders(self, status: str = "all") -> List[Order]:
        """Get orders from Alpaca."""
        try:
            orders_data = self.api.list_orders(status=status)
            orders = []
            
            for o in orders_data:
                order = Order(
                    symbol=o.symbol,
                    quantity=float(o.qty),
                    side=o.side,
                    order_type=o.order_type,
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    stop_price=float(o.stop_price) if o.stop_price else None,
                    broker_order_id=o.id,
                    status=o.status,
                    timestamp=datetime.fromisoformat(o.created_at)
                )
                orders.append(order)
            
            return orders
        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote from Alpaca."""
        try:
            quote = self.api.get_latest_trade(symbol)
            return {
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': float(quote.price)
            }
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}


class PaperTradingEngine:
    """Main paper trading engine."""
    
    def __init__(self, broker_adapter: BrokerAdapter, risk_limit: float = 0.02):
        """
        Initialize paper trading engine.
        
        Args:
            broker_adapter: Broker adapter instance
            risk_limit: Max risk per trade (default 2% of portfolio)
        """
        self.broker = broker_adapter
        self.risk_limit = risk_limit
        self.logger = logging.getLogger(__name__)
        self.account = None
    
    async def initialize(self) -> bool:
        """Initialize connection and fetch account."""
        try:
            if not await self.broker.connect():
                return False
            
            self.account = await self.broker.get_account()
            self.logger.info("Paper trading engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    async def execute_signal(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        current_price: float
    ) -> Optional[str]:
        """
        Execute trade based on model signal.
        
        Args:
            symbol: Stock symbol
            signal: Signal value (-1 to 1)
            confidence: Prediction confidence
            current_price: Current stock price
            
        Returns:
            Order ID if executed, None otherwise
        """
        try:
            # Update account
            self.account = await self.broker.get_account()
            
            # Check if we have existing position
            existing_position = self.account.positions.get(symbol)
            
            # Determine action based on signal
            if signal > 0.3 and confidence > 0.6:  # BUY signal
                return await self._execute_buy(symbol, current_price, existing_position)
            elif signal < -0.3 and confidence > 0.6:  # SELL signal
                return await self._execute_sell(symbol, current_price, existing_position)
            else:  # HOLD
                return None
        except Exception as e:
            self.logger.error(f"Failed to execute signal: {e}")
            return None
    
    async def _execute_buy(
        self,
        symbol: str,
        price: float,
        existing_position: Optional[Position]
    ) -> Optional[str]:
        """Execute buy order."""
        try:
            # Calculate position size based on risk
            position_risk = self.account.portfolio_value * self.risk_limit
            quantity = int(position_risk / price)
            
            if quantity <= 0 or quantity * price > self.account.buying_power:
                self.logger.warning(f"Insufficient funds for {symbol}")
                return None
            
            order = Order(
                symbol=symbol,
                quantity=quantity,
                side="buy",
                order_type="market"
            )
            
            order_id = await self.broker.place_order(order)
            self.logger.info(f"BUY order placed: {symbol} x{quantity} @ ${price}")
            return order_id
        except Exception as e:
            self.logger.error(f"Failed to execute buy: {e}")
            return None
    
    async def _execute_sell(
        self,
        symbol: str,
        price: float,
        existing_position: Optional[Position]
    ) -> Optional[str]:
        """Execute sell order."""
        try:
            if not existing_position or existing_position.quantity <= 0:
                self.logger.warning(f"No position to sell for {symbol}")
                return None
            
            # Close entire position
            quantity = existing_position.quantity
            
            order = Order(
                symbol=symbol,
                quantity=quantity,
                side="sell",
                order_type="market"
            )
            
            order_id = await self.broker.place_order(order)
            self.logger.info(f"SELL order placed: {symbol} x{quantity} @ ${price}")
            return order_id
        except Exception as e:
            self.logger.error(f"Failed to execute sell: {e}")
            return None
    
    async def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        try:
            self.account = await self.broker.get_account()
            
            positions = await self.broker.get_positions()
            
            total_pnl = sum(p.unrealized_pnl for p in positions.values())
            total_pnl_pct = (total_pnl / self.account.portfolio_value * 100) if self.account.portfolio_value > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cash": self.account.cash,
                "portfolio_value": self.account.portfolio_value,
                "total_value": self.account.total_value,
                "positions": {
                    symbol: {
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": pos.unrealized_pnl_pct
                    }
                    for symbol, pos in positions.items()
                },
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "num_positions": len(positions)
            }
        except Exception as e:
            self.logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    async def shutdown(self) -> bool:
        """Shutdown trading engine."""
        try:
            result = await self.broker.disconnect()
            self.logger.info("Paper trading engine shut down")
            return result
        except Exception as e:
            self.logger.error(f"Failed to shutdown: {e}")
            return False
