"""
Data API

Endpoints for unified data access (macro, sample data source, economic calendar).
"""

import os
import requests
import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL_CALENDAR = 3600  # 1 hour


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
            from config import get_settings
            if not get_settings().data.fred_configured:
                return {"error": "FRED API key not configured. Set FRED_API_KEY in .env", "series": []}
        except ImportError:
            pass
        fetcher = DataFetcher()
        if not fetcher.fred:
            return {
                "error": "FRED API key not configured. Set FRED_API_KEY in .env",
                "series": [],
            }

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
        resp = requests.get(url, params=params, timeout=15)
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
        data = yf.download(sym_list, period="5d", progress=False, auto_adjust=True, group_by="ticker", threads=False)
        if data.empty:
            return {"quotes": [{"symbol": s, "price": None, "change_pct": None} for s in sym_list]}
        quotes = []
        if len(sym_list) == 1:
            close = data["Close"] if "Close" in data.columns else (data.iloc[:, 0] if len(data.columns) > 0 else None)
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
                        col = data["Close"][s] if s in data["Close"].columns else None
                    else:
                        col = data["Close"] if "Close" in data.columns else None
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
