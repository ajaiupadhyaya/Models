"""
IEX Cloud data provider.

Supports: US equities, fundamentals, news
Docs: https://iexcloud.io/docs
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import requests

from .base import DataProvider, OHLCV, FundamentalsData, AssetType

logger = logging.getLogger(__name__)


class IEXProvider(DataProvider):
    """IEX Cloud data provider."""
    
    BASE_URL = "https://cloud.iexapis.com/stable"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("iex", api_key or os.getenv("IEX_API_KEY"))
        self.rate_limit = 100  # calls per second on free tier
        if not self.api_key:
            logger.warning("IEX_API_KEY not found; IEX provider disabled")
    
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """IEX supports US equities only."""
        return asset_type == AssetType.EQUITY
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """
        Fetch OHLCV from IEX.
        
        IEX provides daily bars via /stock/{symbol}/chart/range/{range}
        Ranges: 5y, 2y, 1y, ytd, 6m, 3m, 1m, 1d, etc.
        """
        if not self.api_key:
            raise ValueError("IEX API key not configured")
        
        if interval != "1day":
            raise ValueError(f"IEX only supports daily data")
        
        # Calculate IEX range parameter
        days_diff = (end_date - start_date).days
        if days_diff <= 1:
            range_param = "1d"
        elif days_diff <= 30:
            range_param = "1m"
        elif days_diff <= 90:
            range_param = "3m"
        elif days_diff <= 180:
            range_param = "6m"
        elif days_diff <= 365:
            range_param = "1y"
        elif days_diff <= 730:
            range_param = "2y"
        else:
            range_param = "5y"
        
        url = f"{self.BASE_URL}/stock/{symbol}/chart/{range_param}"
        params = {"token": self.api_key}
        
        ohlcv_list = []
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            
            if resp.status_code == 401:
                raise ValueError("Invalid IEX API key")
            if resp.status_code == 404:
                raise ValueError(f"Symbol {symbol} not found on IEX")
            
            resp.raise_for_status()
            data = resp.json()
            
            if not isinstance(data, list):
                logger.warning(f"Unexpected IEX response for {symbol}")
                return ohlcv_list
            
            # Filter by date range
            for bar in data:
                bar_date = datetime.strptime(bar["date"], "%Y-%m-%d")
                if start_date <= bar_date <= end_date:
                    ohlcv = OHLCV(
                        date=bar_date,
                        open=float(bar.get("open", 0)),
                        high=float(bar.get("high", 0)),
                        low=float(bar.get("low", 0)),
                        close=float(bar.get("close", 0)),
                        volume=int(bar.get("volume", 0)),
                        adjusted_close=float(bar.get("close", 0)),
                    )
                    ohlcv_list.append(ohlcv)
            
            logger.info(f"IEX: fetched {len(ohlcv_list)} bars for {symbol}")
            return ohlcv_list
        
        except Exception as e:
            logger.error(f"IEX fetch error for {symbol}: {e}")
            raise
    
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch latest price."""
        if not self.api_key:
            raise ValueError("IEX API key not configured")
        
        url = f"{self.BASE_URL}/stock/{symbol}/quote"
        params = {"token": self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            price = data.get("latestPrice")
            if price:
                logger.info(f"IEX latest price for {symbol}: ${price:.2f}")
                return float(price)
            
            raise ValueError(f"No price data for {symbol}")
        
        except Exception as e:
            logger.error(f"IEX latest price error: {e}")
            raise
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalsData]:
        """Fetch fundamentals from IEX."""
        if not self.api_key:
            return None
        
        try:
            # Get quote + company info + key stats
            quote_url = f"{self.BASE_URL}/stock/{symbol}/quote"
            keystats_url = f"{self.BASE_URL}/stock/{symbol}/stats"
            
            quote_resp = requests.get(quote_url, params={"token": self.api_key}, timeout=self.timeout)
            keystats_resp = requests.get(keystats_url, params={"token": self.api_key}, timeout=self.timeout)
            
            if quote_resp.status_code != 200 or keystats_resp.status_code != 200:
                return None
            
            quote = quote_resp.json()
            stats = keystats_resp.json()
            
            price = quote.get("latestPrice", 0)
            
            return FundamentalsData(
                symbol=symbol,
                price=float(price) if price else 0.0,
                pe_ratio=quote.get("peRatio"),
                pb_ratio=stats.get("priceToBook"),
                ps_ratio=stats.get("priceToSales"),
                dividend_yield=quote.get("dividendYield"),
                market_cap=quote.get("marketCap"),
                earnings_per_share=stats.get("ttmEPS"),
                book_value_per_share=stats.get("bookValuePerShare"),
                net_income=stats.get("netIncome"),
                revenue=stats.get("revenue"),
            )
        
        except Exception as e:
            logger.warning(f"IEX fundamentals error: {e}")
            return None
    
    def fetch_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news from IEX."""
        if not self.api_key:
            return []
        
        try:
            url = f"{self.BASE_URL}/stock/{symbol}/news/last/{limit}"
            resp = requests.get(url, params={"token": self.api_key}, timeout=self.timeout)
            
            if resp.status_code != 200:
                return []
            
            articles = []
            for item in resp.json():
                articles.append({
                    "title": item.get("headline"),
                    "source": item.get("source"),
                    "date": item.get("datetime"),
                    "url": item.get("url"),
                    "summary": item.get("summary", ""),
                })
            
            return articles
        
        except Exception as e:
            logger.warning(f"IEX news error: {e}")
            return []
    
    def validate_api_key(self) -> bool:
        """Validate API key."""
        if not self.api_key:
            return False
        
        try:
            url = f"{self.BASE_URL}/status"
            resp = requests.get(url, params={"token": self.api_key}, timeout=5)
            return resp.status_code == 200
        except:
            return False
