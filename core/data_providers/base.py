"""
Base class and registry for data providers.

All providers inherit from DataProvider and implement the abstract methods.
DataProviderRegistry manages instantiation and fallback chains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AssetType(str, Enum):
    """Asset class types."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"


@dataclass
class OHLCV:
    """OHLCV bar data."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
        }


@dataclass
class FundamentalsData:
    """Company fundamentals."""
    symbol: str
    price: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    earnings_per_share: Optional[float] = None
    book_value_per_share: Optional[float] = None
    net_income: Optional[float] = None
    revenue: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "ps_ratio": self.ps_ratio,
            "dividend_yield": self.dividend_yield,
            "market_cap": self.market_cap,
            "earnings_per_share": self.earnings_per_share,
            "book_value_per_share": self.book_value_per_share,
            "net_income": self.net_income,
            "revenue": self.revenue,
        }


class DataProvider(ABC):
    """Abstract base class for all data providers."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        """
        Initialize provider.
        
        Args:
            name: Provider identifier (e.g., "polygon", "iex")
            api_key: API key for authentication (from env if not provided)
        """
        self.name = name
        self.api_key = api_key
        self.rate_limit = None  # Set by subclass
        self.timeout = 30  # Default timeout in seconds
    
    @abstractmethod
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """Check if provider supports asset type."""
        pass
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Asset symbol (e.g., "AAPL", "EURUSD", "BTC/USD")
            start_date: Start date
            end_date: End date
            interval: Bar interval ("1min", "5min", "1hour", "1day", "1week", "1month")
        
        Returns:
            List of OHLCV bars
        
        Raises:
            ValueError: Invalid input
            ConnectionError: API error
        """
        pass
    
    @abstractmethod
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch latest price."""
        pass
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalsData]:
        """
        Fetch company fundamentals (optional).
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
        
        Returns:
            FundamentalsData or None if not supported
        """
        return None
    
    def fetch_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch news articles (optional).
        
        Args:
            symbol: Asset symbol
            limit: Max articles to return
        
        Returns:
            List of news articles with title, source, date, url
        """
        return []
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Check if API key is valid."""
        pass
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit info (calls remaining, reset time, etc)."""
        return {"rate_limit": self.rate_limit}


class DataProviderRegistry:
    """Registry and factory for data providers."""
    
    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.fallback_chain: List[str] = []  # Order to try on failure
    
    def register(self, provider: DataProvider, is_fallback: bool = False):
        """Register a data provider."""
        self.providers[provider.name] = provider
        if is_fallback:
            self.fallback_chain.append(provider.name)
        logger.info(f"Registered provider: {provider.name}")
    
    def set_fallback_chain(self, chain: List[str]):
        """Set fallback order (primary → secondary → tertiary...)."""
        self.fallback_chain = chain
        logger.info(f"Fallback chain: {' → '.join(chain)}")
    
    def get_provider(self, name: str) -> Optional[DataProvider]:
        """Get provider by name."""
        return self.providers.get(name)
    
    def fetch_ohlcv_with_fallback(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day",
        asset_type: AssetType = AssetType.EQUITY,
    ) -> tuple[List[OHLCV], str]:
        """
        Fetch OHLCV data, trying providers in fallback order.
        
        Returns:
            (ohlcv_list, provider_used)
        """
        # Primary provider (if specified in fallback chain)
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            if not provider.supports_asset_type(asset_type):
                logger.debug(f"{provider_name} does not support {asset_type.value}")
                continue
            
            try:
                logger.info(f"Fetching {symbol} from {provider_name}...")
                ohlcv = provider.fetch_ohlcv(symbol, start_date, end_date, interval)
                logger.info(f"✓ {provider_name}: {len(ohlcv)} bars")
                return ohlcv, provider_name
            except Exception as e:
                logger.warning(f"✗ {provider_name} failed: {e}")
                continue
        
        raise ConnectionError(f"All providers failed to fetch {symbol}")
    
    def fetch_latest_price_with_fallback(
        self,
        symbol: str,
        asset_type: AssetType = AssetType.EQUITY,
    ) -> tuple[float, str]:
        """
        Fetch latest price, trying providers in fallback order.
        
        Returns:
            (price, provider_used)
        """
        for provider_name in self.fallback_chain:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            if not provider.supports_asset_type(asset_type):
                continue
            
            try:
                price = provider.fetch_latest_price(symbol)
                logger.info(f"✓ {symbol} latest price from {provider_name}: ${price:.2f}")
                return price, provider_name
            except Exception as e:
                logger.warning(f"✗ {provider_name} failed: {e}")
                continue
        
        raise ConnectionError(f"All providers failed to fetch latest price for {symbol}")
