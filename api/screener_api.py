"""
Screener API

Multi-factor screener using company search and optional filters.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from api.cache import get_cached, set_cached, cache_key

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL_SCREENER = 300  # 5 min


@router.get("/run")
async def run_screener(
    sector: Optional[str] = Query(None, description="Filter by sector (e.g. Technology, Healthcare)"),
    min_market_cap: Optional[float] = Query(None, description="Min market cap (USD)"),
    max_market_cap: Optional[float] = Query(None, description="Max market cap (USD)"),
    limit: int = Query(30, ge=1, le=100, description="Max results"),
) -> Dict[str, Any]:
    """
    Run multi-factor screener. Uses cached company database (sector, market_cap).
    Requires company database to be built (first call may be slow).
    """
    key = cache_key("screener", sector or "", str(min_market_cap or ""), str(max_market_cap or ""), str(limit))
    cached = get_cached(key)
    if cached is not None:
        return cached

    try:
        from core.company_search import CompanySearch
        searcher = CompanySearch()
        if sector:
            companies = searcher.filter_by_sector(sector)
        else:
            companies = searcher.get_top_companies(limit * 2)

        results: List[Dict[str, Any]] = []
        for c in companies:
            symbol = c.get("ticker") or c.get("symbol") or ""
            if not symbol:
                continue
            market_cap = c.get("market_cap") or 0
            if min_market_cap is not None and market_cap < min_market_cap:
                continue
            if max_market_cap is not None and market_cap > max_market_cap:
                continue
            results.append({
                "symbol": symbol,
                "name": c.get("name", ""),
                "sector": c.get("sector", ""),
                "market_cap": market_cap,
                "industry": c.get("industry", ""),
            })
            if len(results) >= limit:
                break

        result = {"results": results, "count": len(results), "sector": sector}
        set_cached(key, result, CACHE_TTL_SCREENER)
        return result
    except Exception as e:
        logger.warning("Screener failed: %s", e)
        return {"results": [], "count": 0, "error": str(e), "sector": sector}
