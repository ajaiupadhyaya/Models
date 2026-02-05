"""
Data fetching module with API integrations for real-time financial and economic data.
Supports FRED, Alpha Vantage, Yahoo Finance, and more.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import yfinance as yf
from fredapi import Fred
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.economicindicator import EconomicIndicator
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    # Alpha Vantage 3.0+ has different structure
    try:
        from alpha_vantage import AlphaVantage
        ALPHA_VANTAGE_AVAILABLE = True
    except ImportError:
        ALPHA_VANTAGE_AVAILABLE = False
        TimeSeries = None
        EconomicIndicator = None
import requests
from dotenv import load_dotenv
from .data_cache import cached
import warnings
warnings.filterwarnings('ignore')

load_dotenv()


def _get_data_config() -> "tuple[object, object]":
    """API keys from config if available, else env."""
    try:
        from config import get_settings
        s = get_settings()
        return (s.data.fred_api_key, s.data.alpha_vantage_api_key)
    except ImportError:
        return (os.getenv("FRED_API_KEY"), os.getenv("ALPHA_VANTAGE_API_KEY"))


class DataFetcher:
    """
    Unified interface for fetching financial and economic data from multiple sources.
    API keys are read from config (env-driven) when available.
    """
    
    def __init__(self):
        """Initialize data fetcher with API keys from config or environment."""
        fred_key, av_key = _get_data_config()
        self.fred_api_key = fred_key if isinstance(fred_key, str) else None
        self.alpha_vantage_key = av_key if isinstance(av_key, str) else None
        
        # Initialize clients
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
        
        # Alpha Vantage initialization (handle different versions)
        if self.alpha_vantage_key and ALPHA_VANTAGE_AVAILABLE:
            try:
                if TimeSeries is not None:
                    self.alpha_vantage_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
                    self.alpha_vantage_econ = EconomicIndicator(key=self.alpha_vantage_key, output_format='pandas') if EconomicIndicator is not None else None
                else:
                    self.alpha_vantage_ts = None
                    self.alpha_vantage_econ = None
            except Exception:
                self.alpha_vantage_ts = None
                self.alpha_vantage_econ = None
        else:
            self.alpha_vantage_ts = None
            self.alpha_vantage_econ = None
    
    @cached(ttl=300)  # Cache for 5 minutes
    def get_stock_data(self, 
                      ticker: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock price data from Yahoo Finance with retry logic.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                if start_date and end_date:
                    data = stock.history(start=start_date, end=end_date)
                else:
                    data = stock.history(period=period)
                
                if data.empty:
                    raise ValueError(f"No data returned for {ticker}")
                
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
                continue
        
        return pd.DataFrame()
    
    def get_multiple_stocks(self, 
                           tickers: List[str], 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch multiple stocks at once with better error handling.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with multi-index columns
        """
        if not tickers:
            return pd.DataFrame()
        
        try:
            if start_date and end_date:
                data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(tickers, period="1y", progress=False)
            
            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch data for tickers {tickers}: {str(e)}")
    
    @cached(ttl=3600)  # Cache for 1 hour
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get comprehensive stock information including company details.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information with safe fallbacks
            return {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', None))
            }
        except Exception as e:
            raise ValueError(f"Failed to fetch info for {ticker}: {str(e)}")
    
    @cached(ttl=3600)  # Cache for 1 hour (economic data updates less frequently)
    def get_economic_indicator(self, 
                               series_id: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.Series:
        """
        Fetch economic data from FRED.
        
        Common series IDs:
        - UNRATE: Unemployment Rate
        - GDP: Gross Domestic Product
        - CPIAUCSL: Consumer Price Index
        - FEDFUNDS: Federal Funds Rate
        - DGS10: 10-Year Treasury Rate
        - PAYEMS: Total Nonfarm Payrolls
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Series with economic data
        """
        if not self.fred:
            raise ValueError("FRED API key not configured. Set FRED_API_KEY in .env file.")
        
        try:
            data = self.fred.get_series(series_id, start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            return pd.Series()
    
    def get_unemployment_rate(self, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.Series:
        """Get US unemployment rate."""
        return self.get_economic_indicator('UNRATE', start_date, end_date)
    
    def get_gdp(self, 
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> pd.Series:
        """Get US GDP."""
        return self.get_economic_indicator('GDP', start_date, end_date)
    
    def get_cpi(self, 
               start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> pd.Series:
        """Get Consumer Price Index (inflation)."""
        return self.get_economic_indicator('CPIAUCSL', start_date, end_date)
    
    def get_fed_funds_rate(self, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.Series:
        """Get Federal Funds Rate."""
        return self.get_economic_indicator('FEDFUNDS', start_date, end_date)
    
    def get_10y_treasury(self, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.Series:
        """Get 10-Year Treasury Rate."""
        return self.get_economic_indicator('DGS10', start_date, end_date)
    
    def get_nonfarm_payrolls(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.Series:
        """Get Total Nonfarm Payrolls."""
        return self.get_economic_indicator('PAYEMS', start_date, end_date)
    
    def get_macro_dashboard_data(self) -> Dict[str, pd.Series]:
        """
        Fetch comprehensive macroeconomic dashboard data.
        
        Returns:
            Dictionary with key economic indicators
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        return {
            'unemployment': self.get_unemployment_rate(start_date, end_date),
            'gdp': self.get_gdp(start_date, end_date),
            'cpi': self.get_cpi(start_date, end_date),
            'fed_funds': self.get_fed_funds_rate(start_date, end_date),
            'treasury_10y': self.get_10y_treasury(start_date, end_date),
            'payrolls': self.get_nonfarm_payrolls(start_date, end_date)
        }
    
    def get_alpha_vantage_data(self, 
                               symbol: str, 
                               function: str = 'TIME_SERIES_DAILY') -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            function: API function (TIME_SERIES_DAILY, TIME_SERIES_INTRADAY, etc.)
        
        Returns:
            DataFrame with price data
        """
        if not self.alpha_vantage_ts:
            raise ValueError("Alpha Vantage API key not configured.")
        
        try:
            if function == 'TIME_SERIES_DAILY':
                data, meta = self.alpha_vantage_ts.get_daily(symbol=symbol, outputsize='full')
            elif function == 'TIME_SERIES_INTRADAY':
                data, meta = self.alpha_vantage_ts.get_intraday(symbol=symbol, interval='1min')
            else:
                raise ValueError(f"Unsupported function: {function}")
            
            return data
        except Exception as e:
            print(f"Error fetching Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def get_crypto_data(self, 
                       symbol: str, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with crypto price data
        """
        return self.get_stock_data(symbol, start_date, end_date)
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get company information and fundamentals."""
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    
    def search_fred_series(self, search_text: str) -> pd.DataFrame:
        """
        Search for FRED series by keyword.
        
        Args:
            search_text: Search keyword
        
        Returns:
            DataFrame with matching series
        """
        if not self.fred:
            raise ValueError("FRED API key not configured.")
        
        try:
            results = self.fred.search(search_text)
            return results
        except Exception as e:
            print(f"Error searching FRED: {e}")
            return pd.DataFrame()
