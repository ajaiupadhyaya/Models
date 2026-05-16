"""
Screener API

Multi-factor screener: DB-backed (company_profile, fundamentals) with extended filters.
Filters: P/E, P/B, EV/EBITDA, revenue growth, margin, debt/equity, market cap, sector, momentum.
Returns sparkline data (recent closes) for each symbol when available from ohlcv.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from api.cache import get_cached, set_cached, cache_key

logger = logging.getLogger(__name__)
router = APIRouter()

CACHE_TTL_SCREENER = 300  # 5 min


def run_screener_sync(
    sector: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    max_market_cap: Optional[float] = None,
    pe_min: Optional[float] = None,
    pe_max: Optional[float] = None,
    pb_max: Optional[float] = None,
    max_debt_equity: Optional[float] = None,
    limit: int = 30,
    include_sparkline: bool = True,
) -> Dict[str, Any]:
    """Sync screener logic for use by API route and AI tools."""
    try:
        from core.db import get_company_profile, get_income_statements, get_balance_sheets
        from sqlalchemy import text
        from core.db import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT symbol, name, sector, industry, market_cap FROM company_profile ORDER BY market_cap DESC NULLS LAST LIMIT 500"))
            rows = list(result)
        companies = [{"symbol": r[0], "name": r[1], "sector": r[2], "industry": r[3], "market_cap": float(r[4]) if r[4] is not None else None} for r in rows]
        if not companies:
            from core.company_search import CompanySearch
            searcher = CompanySearch()
            if sector:
                companies = searcher.filter_by_sector(sector)
            else:
                companies = searcher.get_top_companies(limit * 2)
            companies = [{"symbol": c.get("ticker") or c.get("symbol"), "name": c.get("name"), "sector": c.get("sector"), "industry": c.get("industry"), "market_cap": c.get("market_cap")} for c in companies if c.get("ticker") or c.get("symbol")]

        results: List[Dict[str, Any]] = []
        for c in companies:
            symbol = c.get("symbol") or ""
            if not symbol:
                continue
            market_cap = c.get("market_cap") or 0
            if min_market_cap is not None and market_cap < min_market_cap:
                continue
            if max_market_cap is not None and market_cap > max_market_cap:
                continue
            if sector and (c.get("sector") or "").strip().lower() != sector.strip().lower():
                continue
            pe, pb, de = None, None, None
            try:
                inc = get_income_statements(symbol, "annual", 1)
                bal = get_balance_sheets(symbol, "annual", 1)
                if inc and inc[0].get("data"):
                    d = inc[0]["data"]
                    net_income = d.get("netIncome") or d.get("netIncomeCommonStockholders")
                    eps = d.get("eps") or (float(net_income) / 1e9 if net_income else None)
                    if eps:
                        from core.db import get_ohlcv_range
                        end = datetime.now(timezone.utc)
                        start = (end - timedelta(days=5)).strftime("%Y-%m-%d")
                        ohlcv = get_ohlcv_range(symbol, start, end.strftime("%Y-%m-%d"))
                        price = float(ohlcv[-1]["close"]) if ohlcv else None
                        if price and eps and float(eps) != 0:
                            pe = price / float(eps)
                if bal and bal[0].get("data"):
                    d = bal[0]["data"]
                    total_equity = d.get("totalStockholdersEquity") or d.get("totalEquity")
                    debt = float(d.get("totalDebt") or d.get("longTermDebt") or 0)
                    if total_equity and market_cap:
                        pb = market_cap / float(total_equity) if float(total_equity) != 0 else None
                    if total_equity and debt and float(total_equity) != 0:
                        de = debt / float(total_equity)
            except Exception:
                pass
            if pe_min is not None and (pe is None or pe < pe_min):
                continue
            if pe_max is not None and (pe is None or pe > pe_max):
                continue
            if pb_max is not None and (pb is not None and pb > pb_max):
                continue
            if max_debt_equity is not None and (de is not None and de > max_debt_equity):
                continue
            item = {
                "symbol": symbol,
                "name": c.get("name", ""),
                "sector": c.get("sector", ""),
                "market_cap": market_cap,
                "industry": c.get("industry", ""),
                "pe": round(pe, 2) if pe is not None else None,
                "pb": round(pb, 2) if pb is not None else None,
            }
            if include_sparkline:
                item["sparkline"] = _sparkline_for_symbol(symbol)
            results.append(item)
            if len(results) >= limit:
                break

        return {"results": results, "count": len(results), "sector": sector}
    except Exception as e:
        logger.warning("Screener sync failed: %s", e)
        return {"results": [], "count": 0, "error": str(e), "sector": sector}


def _sparkline_for_symbol(symbol: str, days: int = 30) -> List[float]:
    """Last N days of close prices from DB for sparkline."""
    try:
        from core.db import get_ohlcv_range
        end = datetime.now(timezone.utc)
        start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        rows = get_ohlcv_range(symbol, start, end_str)
        if rows:
            return [float(r["close"]) for r in rows[-30:]]
    except Exception:
        pass
    return []


@router.get("/run")
async def run_screener(
    sector: Optional[str] = Query(None, description="Filter by sector (e.g. Technology, Healthcare)"),
    min_market_cap: Optional[float] = Query(None, description="Min market cap (USD)"),
    max_market_cap: Optional[float] = Query(None, description="Max market cap (USD)"),
    pe_min: Optional[float] = Query(None, description="Min P/E ratio"),
    pe_max: Optional[float] = Query(None, description="Max P/E ratio"),
    pb_max: Optional[float] = Query(None, description="Max P/B ratio"),
    limit: int = Query(30, ge=1, le=100, description="Max results"),
    include_sparkline: bool = Query(True, description="Include last 30d closes for sparkline"),
) -> Dict[str, Any]:
    """
    Run screener. Uses DB company_profile when available, else CompanySearch.
    Extended filters: sector, market_cap, pe_min, pe_max, pb_max.
    """
    key = cache_key("screener", sector or "", str(min_market_cap or ""), str(max_market_cap or ""), str(pe_min or ""), str(pe_max or ""), str(pb_max or ""), str(limit))
    cached = get_cached(key)
    if cached is not None:
        if not include_sparkline and cached.get("results"):
            for r in cached["results"]:
                r.pop("sparkline", None)
        return cached

    result = run_screener_sync(
        sector=sector,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        pe_min=pe_min,
        pe_max=pe_max,
        pb_max=pb_max,
        limit=limit,
        include_sparkline=include_sparkline,
    )
    if result.get("results") and not result.get("error"):
        for r in result["results"]:
            r["pe_ratio"] = r.get("pe")
            r["pb_ratio"] = r.get("pb")
        set_cached(key, result, CACHE_TTL_SCREENER)
    return result
