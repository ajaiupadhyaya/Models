"""
CoinGecko data provider.

Supports: Cryptocurrencies
Docs: https://www.coingecko.com/en/api
Free tier: No API key required, 10 calls/sec
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import requests

from .base import DataProvider, OHLCV, FundamentalsData, AssetType

logger = logging.getLogger(__name__)


class CoinGeckoProvider(DataProvider):
    """CoinGecko crypto data provider (free tier)."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    COIN_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "SOL": "solana",
        "XRP": "ripple",
        "DOGE": "dogecoin",
        "POLKA": "polkadot",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
    }
    
    def __init__(self):
        super().__init__("coingecko")
        self.rate_limit = 10  # calls per second
        self.timeout = 30
    
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """CoinGecko supports crypto only."""
        return asset_type == AssetType.CRYPTO
    
    def _get_coin_id(self, symbol: str) -> str:
        """Map symbol to CoinGecko ID."""
        # Try exact match first
        if symbol in self.COIN_MAP:
            return self.COIN_MAP[symbol]
        
        # Try lowercase
        symbol_upper = symbol.upper().replace("-USD", "").replace("-USDT", "")
        if symbol_upper in self.COIN_MAP:
            return self.COIN_MAP[symbol_upper]
        
        # Fallback to symbol.lower()
        return symbol.lower()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> List[OHLCV]:
        """
        Fetch daily OHLCV from CoinGecko.
        
        CoinGecko only provides daily data via /coins/{id}/market_chart/range/
        """
        if not interval.endswith("day"):
            raise ValueError(f"CoinGecko only supports daily data; got {interval}")
        
        coin_id = self._get_coin_id(symbol)
        
        # CoinGecko returns OHLC in ranges
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": int(start_date.timestamp()),
            "to": int(end_date.timestamp()),
        }
        
        ohlcv_list = []
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            
            if resp.status_code == 404:
                raise ValueError(f"Coin {symbol} not found")
            
            resp.raise_for_status()
            data = resp.json()
            
            # Extract prices, market_caps, volumes
            prices = data.get("prices", [])  # [timestamp, price]
            
            # CoinGecko returns prices only; reconstruct OHLCV
            # Use close as both open/high/low/close for daily data
            for timestamp, price in prices:
                bar_date = datetime.fromtimestamp(timestamp / 1000)
                ohlcv = OHLCV(
                    date=bar_date,
                    open=float(price),
                    high=float(price),
                    low=float(price),
                    close=float(price),
                    volume=0,  # CoinGecko doesn't provide volume in this endpoint
                )
                ohlcv_list.append(ohlcv)
            
            logger.info(f"CoinGecko: fetched {len(ohlcv_list)} bars for {symbol}")
            return ohlcv_list
        
        except Exception as e:
            logger.error(f"CoinGecko fetch error for {symbol}: {e}")
            raise
    
    def fetch_latest_price(self, symbol: str) -> float:
        """Fetch latest price."""
        coin_id = self._get_coin_id(symbol)
        
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
        }
        
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if coin_id in data:
                price = data[coin_id].get("usd")
                if price:
                    logger.info(f"CoinGecko latest price for {symbol}: ${price:.2f}")
                    return float(price)
            
            raise ValueError(f"No price data for {symbol}")
        
        except Exception as e:
            logger.error(f"CoinGecko latest price error: {e}")
            raise
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalsData]:
        """Fetch crypto market data."""
        try:
            coin_id = self._get_coin_id(symbol)
            
            url = f"{self.BASE_URL}/coins/{coin_id}"
            resp = requests.get(url, params={"localization": "false"}, timeout=self.timeout)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            market_data = data.get("market_data", {})
            
            price = market_data.get("current_price", {}).get("usd", 0)
            market_cap = market_data.get("market_cap", {}).get("usd")
            
            return FundamentalsData(
                symbol=symbol,
                price=float(price) if price else 0.0,
                market_cap=float(market_cap) if market_cap else None,
                # Crypto-specific metrics
                pe_ratio=None,  # Crypto doesn't have P/E
            )
        
        except Exception as e:
            logger.warning(f"CoinGecko fundamentals error: {e}")
            return None
    
    def validate_api_key(self) -> bool:
        """CoinGecko free tier doesn't require API key."""
        try:
            url = f"{self.BASE_URL}/simple/price"
            resp = requests.get(url, params={"ids": "bitcoin", "vs_currencies": "usd"}, timeout=5)
            return resp.status_code == 200
        except:
            return False
