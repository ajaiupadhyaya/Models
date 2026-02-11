"""
Data API

Endpoints for unified data access (macro, sample data source, economic calendar).
"""

import os
import requests
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL_CALENDAR = 3600  # 1 hour

# Timeout configurations (seconds)
TIMEOUT_YFINANCE_QUOTES = 10
TIMEOUT_YFINANCE_HISTORICAL = 20
TIMEOUT_FRED_API = 10
TIMEOUT_EXTERNAL_API = 15

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


@router.get("/health-check")
async def data_sources_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for data sources.
    Returns operational status of yfinance, FRED, and other data providers.
    """
    try:
        from core.data_fetcher_enhanced import DataSourceHealthChecker
        health = DataSourceHealthChecker.check_all_sources()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "sources": {}
        }


def _series_to_list(series) -> List[Dict[str, Any]]:
    """Convert pandas Series to list of {date, value} for JSON."""
    if series is None or series.empty:
        return []
    return [
        {"date": str(idx)[:10], "value": float(val)}
        for idx, val in series.items()
        if val is not None and str(val) != "nan"
    ]


@router.get("/macro")
async def get_macro() -> Dict[str, Any]:
    """
    Get macroeconomic indicators (FRED).
    Requires FRED_API_KEY. Returns series with latest values for dashboard.
    Cached 10 minutes (see api/cache.py).
    """
    from api.cache import get_cached, set_cached, cache_key, CACHE_TTL_MACRO
    key = cache_key("data", "macro")
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        from core.data_fetcher import DataFetcher
        try:
            from config.settings import get_settings
            if not get_settings().data.fred_configured:
                return {"error": "FRED API key not configured. Set FRED_API_KEY in Render Environment.", "series": []}
        except ImportError:
            pass
        fetcher = DataFetcher()
        if not fetcher.fred:
            return {"error": "FRED API key not configured. Set FRED_API_KEY in Render Environment.", "series": []}

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        series_list = []
        labels = [
            ("unemployment", "UNRATE", "Unemployment Rate"),
            ("cpi", "CPIAUCSL", "CPI"),
            ("fed_funds", "FEDFUNDS", "Fed Funds Rate"),
            ("treasury_10y", "DGS10", "10Y Treasury"),
        ]
        for key, sid, desc in labels:
            try:
                s = fetcher.get_economic_indicator(sid, start_date, end_date)
                data = _series_to_list(s)
                if data:
                    series_list.append(
                        {"series_id": sid, "description": desc, "data": data[-60:]}
                    )
            except Exception as e:
                logger.warning(f"Macro series {sid} failed: {e}")

        result = {"series": series_list}
        set_cached(key, result, CACHE_TTL_MACRO)
        return result
    except Exception as e:
        logger.warning(f"Macro endpoint failed: {e}")
        return {"series": [], "error": str(e)}


@router.get("/yield-curve")
async def get_yield_curve() -> Dict[str, Any]:
    """
    Get US Treasury yield curve (2Y, 5Y, 10Y, 30Y) for Economic tab visualization.
    Requires FRED_API_KEY. Cached 1 hour.
    """
    from api.cache import get_cached, set_cached, cache_key
    key = cache_key("data", "yield-curve")
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        from core.data_fetcher import DataFetcher
        try:
            from config.settings import get_settings
            if not get_settings().data.fred_configured:
                return {"error": "FRED API key not configured. Set FRED_API_KEY in Render Environment.", "maturities": [], "yields": []}
        except ImportError:
            pass
        fetcher = DataFetcher()
        if not fetcher.fred:
            return {"error": "FRED API key not configured. Set FRED_API_KEY in Render Environment.", "maturities": [], "yields": []}
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        series_ids = [("DGS2", 2), ("DGS5", 5), ("DGS10", 10), ("DGS30", 30)]
        yields_list: List[float] = []
        maturities: List[int] = []
        last_date: Optional[str] = None
        for sid, mat in series_ids:
            try:
                s = fetcher.get_economic_indicator(sid, start_date, end_date)
                if s is not None and not s.empty:
                    last_val = s.dropna().iloc[-1] if len(s.dropna()) else None
                    if last_val is not None:
                        maturities.append(mat)
                        yields_list.append(float(last_val))
                    if last_date is None and s.index is not None and len(s) > 0:
                        last_date = str(s.index[-1])[:10]
            except Exception as e:
                logger.warning("Yield curve series %s failed: %s", sid, e)
        result = {"maturities": maturities, "yields": yields_list, "date": last_date}
        if maturities:
            set_cached(key, result, 3600)
        return result
    except Exception as e:
        logger.warning("Yield curve failed: %s", e)
        return {"error": str(e), "maturities": [2, 5, 10, 30], "yields": []}
@router.get("/economic-calendar")
async def get_economic_calendar(
    days_ahead: int = 30,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get upcoming economic release dates from FRED.
    Requires FRED_API_KEY. Returns date, release_name, release_id. Cached 1 hour.
    """
    from api.cache import get_cached, set_cached, cache_key
    key = cache_key("data", "economic-calendar", str(days_ahead), str(limit))
    cached = get_cached(key)
    if cached is not None:
        return cached

    try:
        from config import get_settings
        fred_key = get_settings().data.fred_api_key if get_settings().data.fred_configured else None
    except ImportError:
        fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key or not fred_key.strip():
        return {"events": [], "error": "FRED_API_KEY not configured. Set it in .env for economic calendar."}

    try:
        start = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = "https://api.stlouisfed.org/fred/releases/dates"
        params = {
            "api_key": fred_key,
            "file_type": "json",
            "realtime_start": start,
            "realtime_end": end,
            "sort_order": "asc",
            "limit": min(limit, 200),
        }
        resp = requests.get(url, params=params, timeout=TIMEOUT_FRED_API)
        resp.raise_for_status()
        data = resp.json()
        release_dates = data.get("release_dates", [])
        events = [
            {"date": r.get("date"), "release_name": r.get("release_name", ""), "release_id": r.get("release_id")}
            for r in release_dates
        ]
        result = {"events": events}
        set_cached(key, result, CACHE_TTL_CALENDAR)
        return result
    except requests.RequestException as e:
        logger.warning("Economic calendar fetch failed: %s", e)
        return {"events": [], "error": str(e)}
    except Exception as e:
        logger.warning("Economic calendar failed: %s", e)
        return {"events": [], "error": str(e)}


@router.get("/quotes")
async def get_quotes(symbols: str = Query("AAPL,MSFT,GOOGL,SPY,QQQ", description="Comma-separated symbols")) -> Dict[str, Any]:
    """
    Get last price and change for a list of symbols (from yfinance).
    Used by the ticker strip for real quotes without depending on AI summary.
    """
    from api.cache import get_cached, set_cached, cache_key
    key = cache_key("data", "quotes", symbols)
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
        if not sym_list:
            return {"quotes": []}
        
        # Fetch with timeout to prevent hanging
        def _fetch():
            return yf.download(sym_list, period="5d", progress=False, auto_adjust=True, group_by="ticker", threads=False)
        
        try:
            data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(executor, _fetch),
                timeout=TIMEOUT_YFINANCE_QUOTES
            )
        except asyncio.TimeoutError:
            logger.warning(f"Quotes fetch timed out for {symbols}")
            return {
                "quotes": [{"symbol": s, "price": None, "change_pct": None} for s in sym_list],
                "error": "Data fetch timed out. Please try again."
            }
        
        if data.empty:
            return {"quotes": [{"symbol": s, "price": None, "change_pct": None} for s in sym_list], "error": "No data from Yahoo Finance. Check Render logs."}
        quotes = []
        if len(sym_list) == 1:
            # Single ticker with group_by='ticker' has multi-level columns: (ticker, OHLCV)
            if isinstance(data.columns, pd.MultiIndex):
                # Extract Close column for the ticker
                ticker = sym_list[0]
                if (ticker, 'Close') in data.columns:
                    close = data[(ticker, 'Close')]
                else:
                    # Fallback: try to find Close in level 1
                    close_cols = [col for col in data.columns if col[1] == 'Close']
                    close = data[close_cols[0]] if close_cols else None
            else:
                close = data["Close"] if "Close" in data.columns else (data.iloc[:, 3] if len(data.columns) > 3 else None)
            
            if close is not None and hasattr(close, "dropna"):
                close = close.dropna()
            
            if close is not None and hasattr(close, "iloc") and len(close) > 0:
                last = float(close.iloc[-1])
                prev = float(close.iloc[-2]) if len(close) > 1 else last
                change = ((last - prev) / prev * 100) if prev and prev != 0 else None
            else:
                last = prev = change = None
            quotes.append({"symbol": sym_list[0], "price": last, "change_pct": change})
        else:
            for s in sym_list:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        # Multi-ticker: columns are (ticker, OHLCV)
                        if (s, 'Close') in data.columns:
                            col = data[(s, 'Close')]
                        else:
                            col = None
                    else:
                        col = data["Close"] if "Close" in data.columns else None
                    
                    if col is not None and hasattr(col, "dropna"):
                        col = col.dropna()
                    
                    if col is not None and hasattr(col, "iloc") and len(col) > 0:
                        last = float(col.iloc[-1])
                        prev = float(col.iloc[-2]) if len(col) > 1 else last
                        change = ((last - prev) / prev * 100) if prev and prev != 0 else None
                    else:
                        last = prev = change = None
                    quotes.append({"symbol": s, "price": last, "change_pct": change})
                except Exception:
                    quotes.append({"symbol": s, "price": None, "change_pct": None})
        result = {"quotes": quotes}
        set_cached(key, result, 60)
        return result
    except Exception as e:
        logger.warning("Quotes fetch failed: %s", e)
        return {"quotes": [], "error": str(e)}


@router.get("/correlation")
async def get_correlation(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN,TSLA", description="Comma-separated symbols for correlation matrix"),
    period: str = Query("1y", description="Price history period"),
) -> Dict[str, Any]:
    """
    Return correlation matrix of daily returns for the given symbols.
    Used by the terminal for correlation heatmap (e.g. Economic or Portfolio).
    """
    from api.cache import get_cached, set_cached, cache_key
    key = cache_key("data", "correlation", symbols, period)
    cached = get_cached(key)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:15]
        if len(sym_list) < 2:
            return {"symbols": [], "matrix": [], "error": "Provide at least 2 symbols"}
        
        # Fetch with timeout
        def _fetch():
            return yf.download(sym_list, period=period, progress=False, auto_adjust=True, group_by="ticker", threads=False)
        
        try:
            data = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(executor, _fetch),
                timeout=TIMEOUT_YFINANCE_HISTORICAL
            )
        except asyncio.TimeoutError:
            logger.warning(f"Correlation fetch timed out for {symbols}")
            return {"symbols": sym_list, "matrix": [], "error": "Data fetch timed out. Try a shorter period."}
        if data.empty:
            return {"symbols": sym_list, "matrix": [], "error": "No price data"}
        if isinstance(data.columns, pd.MultiIndex):
            # group_by='ticker' -> (Ticker, OHLCV); level 1 is Open/High/Low/Close/Volume
            try:
                closes = data.xs("Close", axis=1, level=1)
            except (KeyError, TypeError):
                level0 = data.columns.get_level_values(0).unique()
                closes = pd.DataFrame({s: data[s]["Close"] for s in sym_list if s in level0})
            if closes.empty or not hasattr(closes, "columns"):
                closes = pd.DataFrame()
            else:
                closes = closes.reindex(columns=[s for s in sym_list if s in closes.columns]).dropna(axis=1, how="all").dropna(axis=0, how="all")
        else:
            closes = data[["Close"]].copy() if "Close" in data.columns else data.iloc[:, :1].copy()
            closes.columns = sym_list[: closes.shape[1]]
        closes = closes.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if closes.shape[1] < 2:
            return {"symbols": sym_list, "matrix": [], "error": "Insufficient series"}
        returns = closes.pct_change().dropna()
        if len(returns) < 5:
            return {"symbols": list(closes.columns), "matrix": [], "error": "Insufficient data points"}
        corr = returns.corr()
        matrix = corr.values.tolist()
        symbols_out = list(corr.index)
        result = {"symbols": symbols_out, "matrix": matrix}
        set_cached(key, result, 300)
        return result
    except Exception as e:
        logger.warning("Correlation failed: %s", e)
        return {"symbols": [], "matrix": [], "error": str(e)}
