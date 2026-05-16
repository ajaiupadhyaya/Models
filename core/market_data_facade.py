"""
Canonical market data facade for service-layer consumers.

This module centralizes DB-first OHLCV access with a single provider fallback path.
Service modules should import this facade instead of instantiating DataFetcher directly.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv_df(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV with DB-first strategy and provider fallback.

    Returns a normalized DataFrame with at least `Close` when data is available.
    """
    from core.db import get_ohlcv_range

    rows = get_ohlcv_range(symbol.upper(), start_date, end_date)
    if rows:
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        return df.rename(
            columns={
                "close": "Close",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
            }
        )

    try:
        from core.data_fetcher import DataFetcher

        fetcher = DataFetcher()
        df = fetcher.get_stock_data(
            symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            period="max",
        )
        if df is not None and not df.empty and "Close" in df.columns:
            return df
    except Exception as e:
        logger.warning("OHLCV fallback failed for %s: %s", symbol, e)
    return None


def fetch_returns_matrix(
    tickers: List[str],
    start_date: str,
    end_date: str,
    min_series: int = 2,
) -> pd.DataFrame:
    """Return aligned daily returns matrix for tickers using the canonical fetch path."""
    prices: Dict[str, pd.Series] = {}
    for sym in [t.upper() for t in tickers]:
        df = fetch_ohlcv_df(sym, start_date, end_date)
        if df is None or df.empty:
            continue
        if "Close" not in df.columns:
            continue
        prices[sym] = df["Close"].dropna().rename(sym)

    if len(prices) < min_series:
        return pd.DataFrame()

    aligned = pd.DataFrame(prices).dropna()
    if aligned.empty or aligned.shape[1] < min_series:
        return pd.DataFrame()
    return aligned.pct_change().dropna()
