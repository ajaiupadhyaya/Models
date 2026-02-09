"""
Polygon.io data provider.

Supports: Equities, crypto, forex, options
Docs: https://polygon.io/docs
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import requests

from .base import DataProvider, OHLCV, FundamentalsData, AssetType

logger = logging.getLogger(__name__)


class PolygonProvider(DataProvider):
    """Polygon.io data provider."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("polygon", api_key or os.getenv("POLYGON_API_KEY"))
        self.rate_limit = 5  # calls per minute on free tier
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not found; polygon provider disabled")
    
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """Polygon supports equities, crypto, forex."""
        return asset_type in (
            AssetType.EQUITY,
            AssetType.CRYPTO,
            AssetType.FOREX,
        )
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """
        Fetch OHLCV from Polygon.
        
        Maps interval to Polygon multiplier:
        - 1day → timespan=day, multiplier=1
        - 1week → timespan=week, multiplier=1
        - 1hour → timespan=hour, multiplier=1
        - 1min → timespan=minute, multiplier=1
        """
        if not self.api_key:
            raise ValueError("Polygon API key not configured")
        
        # Parse interval
        if interval == "1day":
            timespan = "day"
            multiplier = 1
        elif interval == "1week":
            timespan = "week"
            multiplier = 1
        elif interval == "1hour":
            timespan = "hour"
            multiplier = 1
        elif interval == "5min":
            timespan = "minute"
            multiplier = 5
        else:
            raise ValueError(f"Unsupported interval: {interval}")
        
        # Polygon endpoint: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {"apikey": self.api_key, "sort": "asc", "limit": 50000}
        
        ohlcv_list = []
        next_url = None
        
        try:
            while True:
                # Use next_url if provided (pagination)
                if next_url:
                    resp = requests.get(next_url, timeout=self.timeout)
                else:
                    resp = requests.get(url, params=params, timeout=self.timeout)
                
                if resp.status_code == 401:
                    raise ValueError("Invalid Polygon API key")
                resp.raise_for_status()
                
                data = resp.json()
                
                # Check for results
                if "results" not in data or not data["results"]:
                    logger.info(f"No data for {symbol} on Polygon")
                    break
                
                # Parse bars
                for bar in data["results"]:
                    ohlcv = OHLCV(
                        date=datetime.fromtimestamp(bar["t"] / 1000),  # Polygon ts in ms
                        open=bar["o"],
                        high=bar["h"],
                        low=bar["l"],
                        close=bar["c"],
                        volume=int(bar.get("v", 0)),
                        adjusted_close=bar.get("vw"),  # volume-weighted close
                    )
                    ohlcv_list.append(ohlcv)
                
                # Check for next page
                if "next_url" in data:
                    next_url = data["next_url"]
                else:
                    break
            
            logger.info(f"Polygon: fetched {len(ohlcv_list)} bars for {symbol}")
            return ohlcv_list
        
        except Exception as e:
            logger.error(f"Polygon fetch error for {symbol}: {e}")
            raise
    
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch latest price."""
        if not self.api_key:
            raise ValueError("Polygon API key not configured")
        
        # Use /v3/snapshot/locale/us/markets/stocks/tickers/{ticker}
        url = f"{self.BASE_URL}/v3/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apikey": self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if "results" in data and data["results"]:
                price = data["results"][0]["last_quote"]["ask"]
                logger.info(f"Polygon latest price for {symbol}: ${price:.2f}")
                return float(price)
            
            raise ValueError(f"No price data for {symbol}")
        
        except Exception as e:
            logger.error(f"Polygon latest price error: {e}")
            raise
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalsData]:
        """Fetch fundamentals from Polygon."""
        if not self.api_key:
            return None
        
        # /v3/ticker/{ticker}/fundamentals?limit=1
        url = f"{self.BASE_URL}/v3/ticker/{symbol}/fundamentals"
        params = {"apikey": self.api_key, "limit": 1}
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            if "results" not in data or not data["results"]:
                return None
            
            fund = data["results"][0]
            price = fund.get("stock_price")
            
            return FundamentalsData(
                symbol=symbol,
                price=price or 0.0,
                pe_ratio=fund.get("pe_ratio"),
                pb_ratio=fund.get("pb_ratio"),
                ps_ratio=fund.get("price_to_sales_ratio"),
                dividend_yield=fund.get("dividend_yield"),
                market_cap=fund.get("market_capitalization"),
                earnings_per_share=fund.get("earnings_per_share"),
                book_value_per_share=fund.get("book_value_per_share"),
                net_income=fund.get("net_income"),
                revenue=fund.get("revenue"),
            )
        
        except Exception as e:
            logger.warning(f"Polygon fundamentals error: {e}")
            return None
    
    def validate_api_key(self) -> bool:
        """Validate API key."""
        if not self.api_key:
            return False
        
        try:
            url = f"{self.BASE_URL}/v1/marketstatus/now"
            resp = requests.get(url, params={"apikey": self.api_key}, timeout=5)
            return resp.status_code == 200
        except:
            return False
