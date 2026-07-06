"""Deterministic market-data fallbacks for production demos.

These values are used only when live providers are unavailable or too slow. The
API responses that use them include source/warning metadata so the frontend can
label the data honestly.
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List


FALLBACK_BASE_PRICES: Dict[str, float] = {
    "AAPL": 214.25,
    "MSFT": 497.10,
    "GOOGL": 192.80,
    "GOOG": 194.10,
    "AMZN": 221.40,
    "TSLA": 268.75,
    "NVDA": 141.30,
    "META": 713.50,
    "SPY": 626.20,
    "QQQ": 556.75,
    "IWM": 221.60,
    "GLD": 307.90,
    "TLT": 87.35,
    "BND": 73.10,
    "SCHF": 41.85,
}

_PERIOD_DAYS = {
    "1d": 2,
    "5d": 5,
    "1mo": 22,
    "3mo": 66,
    "6mo": 126,
    "1y": 252,
    "2y": 504,
}


def live_market_data_disabled() -> bool:
    """Return whether live market-provider calls should be skipped."""
    return os.getenv("DISABLE_LIVE_MARKET_DATA", "").lower() in {"1", "true", "yes"}


def has_fallback_symbol(symbol: str) -> bool:
    """Return whether we have explicit fallback coverage for a symbol."""
    return symbol.strip().upper() in FALLBACK_BASE_PRICES


def fallback_quote(symbol: str) -> Dict[str, Any]:
    """Return one fallback quote row."""
    sym = symbol.strip().upper()
    base = FALLBACK_BASE_PRICES.get(sym)
    if base is None:
        return {"symbol": sym, "price": None, "change_pct": None}

    seed = _seed(sym)
    day_offset = date.today().toordinal() % 37
    change = math.sin((seed % 19 + day_offset) / 5) * 1.25
    price = base * (1 + math.sin((seed % 31 + day_offset) / 11) * 0.015)
    return {"symbol": sym, "price": round(price, 2), "change_pct": round(change, 2)}


def fallback_quotes(symbols: Iterable[str]) -> Dict[str, Any]:
    """Return fallback quotes for a symbol list."""
    rows = [fallback_quote(s) for s in symbols]
    return {
        "quotes": rows,
        "source": "fallback",
        "warning": "Live quote provider unavailable; showing deterministic demo prices.",
    }


def fallback_candles(symbol: str, period: str = "3mo") -> Dict[str, Any]:
    """Return deterministic OHLCV candles for supported demo symbols."""
    sym = symbol.strip().upper()
    base = FALLBACK_BASE_PRICES.get(sym)
    if base is None:
        return {
            "candles": [],
            "symbol": sym,
            "period": period,
            "error": "No live data found and no fallback data is configured for this symbol.",
        }

    count = _PERIOD_DAYS.get(period.lower(), 66)
    days = _business_days(count)
    seed = _seed(sym)
    candles: List[Dict[str, Any]] = []
    previous_close = base * 0.96

    for i, day in enumerate(days):
        trend = (i / max(count - 1, 1) - 0.5) * 0.08
        seasonal = math.sin((i + seed % 17) / 4.3) * 0.025
        close = max(1.0, base * (1 + trend + seasonal))
        open_ = previous_close
        high = max(open_, close) * (1 + 0.004 + ((seed + i) % 5) * 0.001)
        low = min(open_, close) * (1 - 0.004 - ((seed + i) % 3) * 0.001)
        volume = int(1_500_000 + ((seed + i * 7919) % 4_000_000))
        candles.append({
            "date": day.isoformat(),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        })
        previous_close = close

    return {
        "candles": candles,
        "symbol": sym,
        "period": period,
        "source": "fallback",
        "warning": "Live historical provider unavailable; showing deterministic demo candles.",
    }


def _seed(symbol: str) -> int:
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _business_days(count: int) -> List[date]:
    days: List[date] = []
    current = date.today()
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)
    return list(reversed(days))
