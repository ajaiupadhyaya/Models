"""
In-memory TTL cache for expensive or rate-limited API responses.

Cached endpoints and TTLs:
- GET /api/v1/data/macro — 10 min (CACHE_TTL_MACRO)
- GET /api/v1/ai/market-summary — 5 min (CACHE_TTL_MARKET_SUMMARY)
- GET /api/v1/company/analyze/{ticker} — 15 min (CACHE_TTL_COMPANY_ANALYZE)

For multi-instance deployments, replace with Redis-backed cache (same key/ttl contract).
"""

import time
from typing import Any, Dict, Optional

# Default TTLs (seconds)
CACHE_TTL_MACRO = 600       # 10 min — FRED rate limits
CACHE_TTL_MARKET_SUMMARY = 300   # 5 min — Alpha Vantage / AI
CACHE_TTL_COMPANY_ANALYZE = 900  # 15 min — heavy fundamental + DCF


class TTLCache:
    """Simple in-memory key -> (value, expiry_ts). Not shared across processes."""

    def __init__(self) -> None:
        self._store: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key not in self._store:
            return None
        val, expiry = self._store[key]
        if now >= expiry:
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: Any, ttl_sec: int) -> None:
        self._store[key] = (value, time.time() + ttl_sec)


_response_cache = TTLCache()


def get_cached(key: str) -> Optional[Any]:
    """Return cached value if present and not expired."""
    return _response_cache.get(key)


def set_cached(key: str, value: Any, ttl_sec: int) -> None:
    """Store value with TTL."""
    _response_cache.set(key, value, ttl_sec)


def cache_key(prefix: str, *parts: str) -> str:
    """Build cache key from prefix and normalized parts."""
    return prefix + ":".join(str(p).strip().upper() for p in parts if p is not None)
