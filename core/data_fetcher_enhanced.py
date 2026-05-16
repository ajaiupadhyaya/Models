"""
Enhanced data fetching utilities with rate limiting, caching, and fallback strategies.
This module extends DataFetcher with additional production-ready features.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


class DataFetcherRateLimiter:
    """
    Simple rate limiter to avoid overwhelming data sources.
    Yahoo Finance typically allows ~2000 requests/hour from a single IP.
    """
    
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded rate limit."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            # Wait until oldest request expires
            wait_time = 60 - (now - self.requests[0]) + 1
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests = []
        
        self.requests.append(now)


# Global rate limiter instance
_rate_limiter = DataFetcherRateLimiter()


def with_rate_limiting(func):
    """Decorator to add rate limiting to data fetching functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


class DataValidator:
    """Validates data quality and completeness."""
    
    @staticmethod
    def validate_stock_data(data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Validate stock data quality.
        
        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []
        
        if data.empty:
            issues.append("Data is empty")
            return {"valid": False, "issues": issues, "warnings": warnings}
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for data quality
        if 'Close' in data.columns:
            # Check for null values
            null_count = data['Close'].isnull().sum()
            if null_count > len(data) * 0.1:  # More than 10% nulls
                issues.append(f"High null count: {null_count}/{len(data)} rows")
            elif null_count > 0:
                warnings.append(f"Some null values: {null_count} rows")
            
            # Check for zeros (unusual)
            zero_count = (data['Close'] == 0).sum()
            if zero_count > 0:
                warnings.append(f"Found {zero_count} zero prices (unusual)")
            
            # Check for extreme volatility (potential data errors)
            if len(data) > 1:
                pct_change = data['Close'].pct_change().abs()
                extreme_moves = (pct_change > 0.5).sum()  # >50% moves
                if extreme_moves > 0:
                    warnings.append(f"Found {extreme_moves} extreme price moves (>50%)")
        
        # Check date continuity
        if len(data) > 1:
            date_gaps = pd.Series(data.index).diff()
            max_gap = date_gaps.max()
            if max_gap > timedelta(days=7):
                warnings.append(f"Large date gap detected: {max_gap}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "row_count": len(data),
            "date_range": f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else None,
            "null_percentage": (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100) if len(data) > 0 else 0
        }
    
    @staticmethod
    def validate_multiple_stocks(data: pd.DataFrame, tickers: List[str]) -> Dict[str, Any]:
        """
        Validate multi-stock data.
        
        Returns:
            Dict with validation results per ticker
        """
        results = {}
        
        if data.empty:
            return {"valid": False, "error": "Data is empty"}
        
        # Handle single vs multi-ticker result
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker data
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(1):
                        ticker_data = data.xs(ticker, level=1, axis=1)
                        results[ticker] = DataValidator.validate_stock_data(
                            ticker_data, ticker
                        )
                    else:
                        results[ticker] = {
                            "valid": False,
                            "issues": ["Ticker not in results"]
                        }
            else:
                # Single ticker or flat structure
                results[tickers[0] if tickers else "unknown"] = DataValidator.validate_stock_data(
                    data, tickers[0] if tickers else "unknown"
                )
        except Exception as e:
            return {"valid": False, "error": str(e)}
        
        return {
            "valid": all(r.get("valid", False) for r in results.values()),
            "tickers": results
        }


class DataSourceHealthChecker:
    """Check health of various data sources."""
    
    @staticmethod
    def check_yfinance() -> Dict[str, Any]:
        """Check if yfinance is operational."""
        try:
            from core.data_fetcher import DataFetcher
            df = DataFetcher()
            
            # Try to fetch a small amount of data
            test_data = df.get_stock_data('AAPL', period='5d')
            
            if test_data.empty:
                return {
                    "status": "degraded",
                    "message": "yfinance returned empty data",
                    "operational": False
                }
            
            validation = DataValidator.validate_stock_data(test_data, 'AAPL')
            
            return {
                "status": "operational" if validation["valid"] else "degraded",
                "message": "yfinance is working correctly" if validation["valid"] else "Data quality issues",
                "operational": validation["valid"],
                "details": validation
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }
    
    @staticmethod
    def check_fred() -> Dict[str, Any]:
        """Check if FRED API is operational."""
        try:
            from core.data_fetcher import DataFetcher
            import os
            
            fred_key = os.getenv('FRED_API_KEY')
            if not fred_key or not fred_key.strip():
                return {
                    "status": "not_configured",
                    "message": "FRED_API_KEY not set",
                    "operational": False
                }
            
            df = DataFetcher()
            
            if not df.fred:
                return {
                    "status": "error",
                    "message": "FRED client not initialized",
                    "operational": False
                }
            
            # Try to fetch unemployment data
            data = df.get_unemployment_rate()
            
            if data.empty:
                return {
                    "status": "degraded",
                    "message": "FRED returned empty data",
                    "operational": False
                }
            
            return {
                "status": "operational",
                "message": "FRED API is working correctly",
                "operational": True,
                "data_points": len(data)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_alpha_vantage() -> Dict[str, Any]:
        """Check Alpha Vantage API (daily data endpoint)."""
        import os
        import requests

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key or not api_key.strip():
            return {
                "status": "not_configured",
                "message": "ALPHA_VANTAGE_API_KEY not set",
                "operational": False
            }

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": "AAPL",
                "apikey": api_key.strip(),
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if "Time Series (Daily)" in data:
                return {
                    "status": "operational",
                    "message": "Alpha Vantage daily endpoint OK",
                    "operational": True
                }
            if "Note" in data:
                return {
                    "status": "degraded",
                    "message": "Rate limit reached",
                    "operational": False
                }
            if "Information" in data:
                return {
                    "status": "degraded",
                    "message": data.get("Information", "Premium endpoint or limited access"),
                    "operational": False
                }
            return {
                "status": "degraded",
                "message": "Unexpected response structure",
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_finnhub() -> Dict[str, Any]:
        """Check Finnhub API (news endpoint)."""
        import os
        import requests

        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key or not api_key.strip():
            return {
                "status": "not_configured",
                "message": "FINNHUB_API_KEY not set",
                "operational": False
            }

        try:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": "AAPL",
                "from": from_date,
                "to": to_date,
                "token": api_key.strip(),
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 401:
                return {
                    "status": "error",
                    "message": "Invalid API key",
                    "operational": False
                }
            if resp.status_code == 429:
                return {
                    "status": "degraded",
                    "message": "Rate limit exceeded",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if isinstance(data, list):
                return {
                    "status": "operational",
                    "message": "Finnhub news endpoint OK",
                    "operational": True,
                    "items": len(data)
                }
            return {
                "status": "degraded",
                "message": "Unexpected response format",
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_polygon() -> Dict[str, Any]:
        """Check Polygon.io API (aggregates endpoint)."""
        import os
        import requests

        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key or not api_key.strip():
            return {
                "status": "not_configured",
                "message": "POLYGON_API_KEY not set",
                "operational": False
            }

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}"
            resp = requests.get(url, params={"apikey": api_key.strip()}, timeout=10)
            if resp.status_code in (401, 403):
                return {
                    "status": "error",
                    "message": "Invalid API key",
                    "operational": False
                }
            if resp.status_code == 429:
                return {
                    "status": "degraded",
                    "message": "Rate limit exceeded",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if data.get("results"):
                return {
                    "status": "operational",
                    "message": "Polygon aggregates endpoint OK",
                    "operational": True,
                    "bars": len(data.get("results", []))
                }
            return {
                "status": "degraded",
                "message": data.get("error", "No data returned"),
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_iex() -> Dict[str, Any]:
        """Check IEX Cloud API (chart endpoint)."""
        import os
        import requests

        api_key = os.getenv("IEX_API_KEY")
        if not api_key or not api_key.strip():
            return {
                "status": "not_configured",
                "message": "IEX_API_KEY not set",
                "operational": False
            }

        try:
            url = "https://cloud.iexapis.com/stable/stock/AAPL/chart/1m"
            resp = requests.get(url, params={"token": api_key.strip()}, timeout=10)
            if resp.status_code in (401, 403):
                return {
                    "status": "error",
                    "message": "Invalid API key",
                    "operational": False
                }
            if resp.status_code == 429:
                return {
                    "status": "degraded",
                    "message": "Rate limit exceeded",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if isinstance(data, list):
                return {
                    "status": "operational",
                    "message": "IEX chart endpoint OK",
                    "operational": True,
                    "bars": len(data)
                }
            return {
                "status": "degraded",
                "message": "Unexpected response format",
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_newsapi() -> Dict[str, Any]:
        """Check NewsAPI.org (everything endpoint)."""
        import os
        import requests

        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key or not api_key.strip():
            return {
                "status": "not_configured",
                "message": "NEWSAPI_KEY not set",
                "operational": False
            }

        try:
            url = "https://newsapi.org/v2/everything"
            resp = requests.get(
                url,
                params={"q": "stock market", "pageSize": 5, "apiKey": api_key.strip()},
                timeout=10,
            )
            if resp.status_code == 401:
                return {
                    "status": "error",
                    "message": "Invalid API key",
                    "operational": False
                }
            if resp.status_code == 429:
                return {
                    "status": "degraded",
                    "message": "Rate limit exceeded",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if data.get("status") == "ok":
                return {
                    "status": "operational",
                    "message": "NewsAPI endpoint OK",
                    "operational": True,
                    "articles": len(data.get("articles", []))
                }
            return {
                "status": "degraded",
                "message": data.get("message", "Unexpected response"),
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_coingecko() -> Dict[str, Any]:
        """Check CoinGecko (simple price endpoint)."""
        import requests

        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            resp = requests.get(url, params={"ids": "bitcoin", "vs_currencies": "usd"}, timeout=10)
            if resp.status_code == 429:
                return {
                    "status": "degraded",
                    "message": "Rate limit exceeded",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            data = resp.json()
            if data.get("bitcoin", {}).get("usd") is not None:
                return {
                    "status": "operational",
                    "message": "CoinGecko endpoint OK",
                    "operational": True
                }
            return {
                "status": "degraded",
                "message": "Unexpected response format",
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_alpaca() -> Dict[str, Any]:
        """Check Alpaca account endpoint."""
        import os
        import requests

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            return {
                "status": "not_configured",
                "message": "ALPACA_API_KEY/ALPACA_API_SECRET not set",
                "operational": False
            }

        try:
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            url = f"{base_url}/v2/account"
            headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 401:
                return {
                    "status": "error",
                    "message": "Invalid Alpaca credentials",
                    "operational": False
                }
            if resp.status_code != 200:
                return {
                    "status": "error",
                    "message": f"HTTP {resp.status_code}",
                    "operational": False
                }
            return {
                "status": "operational",
                "message": "Alpaca account endpoint OK",
                "operational": True
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }

    @staticmethod
    def check_sec_edgar() -> Dict[str, Any]:
        """Check SEC EDGAR endpoint."""
        import os
        import requests

        try:
            url = "https://www.sec.gov/cgi-bin/browse-edgar"
            headers = {
                "User-Agent": os.getenv("SEC_USER_AGENT", "financial-terminal/1.0 (contact: support@example.com)"),
                "Accept-Encoding": "gzip, deflate",
            }
            resp = requests.get(url, params={"action": "getcompany", "CIK": "AAPL", "count": 1, "output": "json"}, timeout=10, headers=headers)
            if resp.status_code == 200:
                return {
                    "status": "operational",
                    "message": "SEC EDGAR endpoint OK",
                    "operational": True
                }
            return {
                "status": "error",
                "message": f"HTTP {resp.status_code}",
                "operational": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "operational": False
            }
    
    @staticmethod
    def check_all_sources() -> Dict[str, Any]:
        """Comprehensive health check of all data sources."""
        return {
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "yfinance": DataSourceHealthChecker.check_yfinance(),
                "fred": DataSourceHealthChecker.check_fred(),
                "alpha_vantage": DataSourceHealthChecker.check_alpha_vantage(),
                "finnhub": DataSourceHealthChecker.check_finnhub(),
                "polygon": DataSourceHealthChecker.check_polygon(),
                "iex": DataSourceHealthChecker.check_iex(),
                "newsapi": DataSourceHealthChecker.check_newsapi(),
                "coingecko": DataSourceHealthChecker.check_coingecko(),
                "alpaca": DataSourceHealthChecker.check_alpaca(),
                "sec_edgar": DataSourceHealthChecker.check_sec_edgar(),
            }
        }


def get_data_source_recommendations() -> Dict[str, str]:
    """
    Get recommendations for configuring data sources.
    
    Returns:
        Dict with recommendations for each data source
    """
    import os
    
    recommendations = {}
    
    # Check FRED
    if not os.getenv('FRED_API_KEY'):
        recommendations['fred'] = (
            "FRED API key not configured. Economic data will be unavailable. "
            "Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set FRED_API_KEY in your .env file."
        )

    if not os.getenv('ALPHA_VANTAGE_API_KEY'):
        recommendations['alpha_vantage'] = (
            "Alpha Vantage API key not configured. Some stock data and fundamentals will be unavailable. "
            "Get a free API key at https://www.alphavantage.co/support/#api-key and set ALPHA_VANTAGE_API_KEY."
        )

    if not os.getenv('POLYGON_API_KEY'):
        recommendations['polygon'] = (
            "Polygon API key not configured. Premium equities data will be unavailable. "
            "Get a key at https://polygon.io and set POLYGON_API_KEY."
        )

    if not os.getenv('IEX_API_KEY'):
        recommendations['iex'] = (
            "IEX Cloud API key not configured. IEX fallback data will be unavailable. "
            "Get a key at https://iexcloud.io and set IEX_API_KEY."
        )

    if not os.getenv('FINNHUB_API_KEY'):
        recommendations['finnhub'] = (
            "Finnhub API key not configured. Real-time news headlines will be unavailable. "
            "Get a key at https://finnhub.io and set FINNHUB_API_KEY."
        )

    if not os.getenv('NEWSAPI_KEY'):
        recommendations['newsapi'] = (
            "NewsAPI key not configured. NewsAPI fallback will be unavailable. "
            "Get a key at https://newsapi.org and set NEWSAPI_KEY."
        )

    if not os.getenv('SEC_USER_AGENT'):
        recommendations['sec_edgar'] = (
            "SEC EDGAR requires a User-Agent header. Set SEC_USER_AGENT with contact info "
            "(e.g., 'financial-terminal/1.0 (contact: you@example.com)')."
        )
    
    # yfinance recommendations
    recommendations['yfinance'] = (
        "yfinance is the primary data source for stocks and crypto. "
        "Ensure you have stable internet connection. "
        "Rate limits: ~2000 requests/hour. "
        "For production, consider implementing additional caching."
    )
    
    # Alpaca (for trading, not historical data)
    if not os.getenv('ALPACA_API_KEY'):
        recommendations['alpaca'] = (
            "Alpaca credentials not configured. Paper trading will be unavailable. "
            "Sign up at https://alpaca.markets for free paper trading. "
            "Set ALPACA_API_KEY, ALPACA_API_SECRET, and ALPACA_BASE_URL in .env file. "
            "Note: Alpaca is for trading execution, not historical data."
        )
    
    return recommendations
