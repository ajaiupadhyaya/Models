"""
Data Providers Module

Unified interface for fetching financial data from multiple sources:
- Equities: Polygon, IEX, yfinance
- Macro: FRED, World Bank, IMF
- Fixed Income: yfinance, IEX
- Crypto: CoinGecko, Polygon
- News: NewsAPI, Alpha Vantage
- Fundamentals: SEC EDGAR, Alpha Vantage
- Forex: OANDA, yfinance
"""

from .base import DataProvider, DataProviderRegistry, OHLCV, FundamentalsData
from .polygon_provider import PolygonProvider
from .iex_provider import IEXProvider
from .coingecko_provider import CoinGeckoProvider
from .newsapi_provider import NewsAPIProvider
from .sec_edgar_provider import SECEdgarProvider

__all__ = [
    "DataProvider",
    "DataProviderRegistry",
    "OHLCV",
    "FundamentalsData",
    "PolygonProvider",
    "IEXProvider",
    "CoinGeckoProvider",
    "NewsAPIProvider",
    "SECEdgarProvider",
]
