"""
AI tool implementations for Claude function-calling.
Each tool is a Python function called when Claude invokes the corresponding tool.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_dcf(symbol: str, wacc: float, terminal_growth_rate: float) -> Dict[str, Any]:
    """Run DCF valuation. Returns intrinsic value, upside/downside, sensitivity table."""
    try:
        from api.equity_api import compute_dcf
        result = compute_dcf(symbol.upper(), wacc, terminal_growth_rate)
        if "error" in result:
            return {"error": result["error"]}
        return {
            "ticker": result.get("ticker"),
            "intrinsic_value_per_share": result.get("intrinsic_value_per_share"),
            "current_price": result.get("current_price"),
            "upside_downside_pct": result.get("upside_downside_pct"),
            "enterprise_value": result.get("enterprise_value"),
            "equity_value": result.get("equity_value"),
            "wacc": result.get("wacc"),
            "terminal_growth": result.get("terminal_growth"),
            "sensitivity": result.get("sensitivity"),
        }
    except Exception as e:
        logger.exception("run_dcf failed: %s", e)
        return {"error": str(e)}


def screen_stocks(
    sector: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    max_debt_equity: Optional[float] = None,
) -> Dict[str, Any]:
    """Run screener. Returns top 10 matching tickers with key stats."""
    try:
        from api.screener_api import run_screener_sync
        resp = run_screener_sync(
            sector=sector,
            min_market_cap=min_market_cap,
            pe_max=max_pe,
            max_debt_equity=max_debt_equity,
            limit=10,
            include_sparkline=False,
        )
        if resp.get("error"):
            return {"error": resp["error"]}
        results = resp.get("results", [])[:10]
        return {
            "count": len(results),
            "tickers": [
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("name"),
                    "sector": r.get("sector"),
                    "market_cap": r.get("market_cap"),
                    "pe": r.get("pe") or r.get("pe_ratio"),
                    "pb": r.get("pb") or r.get("pb_ratio"),
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.exception("screen_stocks failed: %s", e)
        return {"error": str(e)}


def get_company_overview(symbol: str) -> Dict[str, Any]:
    """Return price, market cap, P/E, EV/EBITDA, revenue, EBITDA, sector, description."""
    try:
        from core.db import get_company_profile, get_income_statements, get_balance_sheets
        from core.db import get_ohlcv_range
        from datetime import datetime, timedelta, timezone

        sym = symbol.upper()
        profile = get_company_profile(sym)
        if not profile:
            try:
                from core.data_providers import FMPProvider
                fmp = FMPProvider()
                if fmp.api_key:
                    p = fmp.fetch_profile(sym)
                    if p:
                        return {
                            "symbol": sym,
                            "name": p.get("companyName"),
                            "sector": p.get("sector"),
                            "industry": p.get("industry"),
                            "market_cap": p.get("mktCap"),
                            "price": p.get("price"),
                            "pe_ratio": p.get("pe"),
                            "description": (p.get("description") or "")[:500],
                        }
            except Exception:
                pass
            return {"error": f"No data for {sym}"}

        end = datetime.now(timezone.utc)
        start = (end - timedelta(days=5)).strftime("%Y-%m-%d")
        rows = get_ohlcv_range(sym, start, end.strftime("%Y-%m-%d"))
        price = float(rows[-1]["close"]) if rows else profile.get("market_cap")
        inc = get_income_statements(sym, "annual", 1)
        bal = get_balance_sheets(sym, "annual", 1)
        revenue = None
        ebitda = None
        pe = None
        ev_ebitda = None
        debt_equity = None
        if inc and inc[0].get("data"):
            d = inc[0]["data"]
            revenue = d.get("revenue") or d.get("totalRevenue")
            ebitda = d.get("ebitda") or d.get("operatingIncome")
        if bal and bal[0].get("data"):
            d = bal[0]["data"]
            equity = d.get("totalStockholdersEquity") or d.get("totalEquity")
            debt = d.get("totalDebt") or d.get("longTermDebt") or 0
            if equity and debt:
                debt_equity = float(debt) / float(equity) if float(equity) != 0 else None
        mcap = profile.get("market_cap")
        if mcap and price and inc and inc[0].get("data"):
            ni = inc[0]["data"].get("netIncome")
            if ni and float(ni) != 0:
                pe = mcap / float(ni) if mcap else None

        return {
            "symbol": sym,
            "name": profile.get("name"),
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
            "market_cap": mcap,
            "price": price,
            "pe_ratio": pe,
            "ev_ebitda": ev_ebitda,
            "revenue": revenue,
            "ebitda": ebitda,
            "debt_equity": debt_equity,
            "description": (profile.get("description") or "")[:500],
        }
    except Exception as e:
        logger.exception("get_company_overview failed: %s", e)
        return {"error": str(e)}


def run_backtest(
    strategy: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run backtest. Returns Sharpe, CAGR, max drawdown, equity curve summary."""
    try:
        from core.backtest_service import run_backtest as _run_bt

        symbol = tickers[0] if tickers else "AAPL"
        result = _run_bt(
            symbol=symbol.upper(),
            strategy=strategy if strategy in ("sma_cross", "rsi_mean_reversion", "factor_momentum") else "sma_cross",
            start_date=start_date,
            end_date=end_date,
        )
        if "error" in result:
            return {"error": result["error"]}
        m = result.get("metrics", {})
        eq = result.get("equity_curve", [])
        return {
            "symbol": result.get("symbol"),
            "strategy": result.get("strategy"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "sortino_ratio": m.get("sortino_ratio"),
            "cagr_pct": m.get("cagr_pct"),
            "max_drawdown_pct": m.get("max_drawdown_pct"),
            "win_rate_pct": m.get("win_rate_pct"),
            "num_trades": m.get("num_trades"),
            "alpha_vs_spy": m.get("alpha_vs_spy"),
            "beta_vs_spy": m.get("beta_vs_spy"),
            "equity_curve_points": len(eq),
            "final_equity": eq[-1]["equity"] if eq else None,
        }
    except Exception as e:
        logger.exception("run_backtest failed: %s", e)
        return {"error": str(e)}


def get_macro_snapshot() -> Dict[str, Any]:
    """Return latest CPI, PCE, unemployment, GDP growth, Fed funds rate, 10Y yield from DB/FRED."""
    try:
        from core.db import get_macro_latest

        series_map = {
            "CPIAUCSL": "cpi",
            "PCEPI": "pce",
            "UNRATE": "unemployment_rate",
            "GDP": "gdp",
            "FEDFUNDS": "fed_funds_rate",
            "DGS10": "treasury_10y",
        }
        ids = list(series_map.keys())
        vals = get_macro_latest(ids)
        if not any(v is not None for v in vals.values()):
            try:
                from core.data_fetcher import DataFetcher
                from datetime import datetime, timedelta
                fetcher = DataFetcher()
                if fetcher.fred:
                    end = datetime.now().strftime("%Y-%m-%d")
                    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                    for sid in ids:
                        s = fetcher.get_economic_indicator(sid, start, end)
                        if s is not None and not s.empty:
                            vals[sid] = float(s.dropna().iloc[-1])
            except Exception:
                pass
        out = {}
        for sid, key in series_map.items():
            v = vals.get(sid)
            if v is not None:
                out[key] = round(float(v), 4)
        return out
    except Exception as e:
        logger.exception("get_macro_snapshot failed: %s", e)
        return {"error": str(e)}


TOOL_REGISTRY: Dict[str, callable] = {
    "run_dcf": run_dcf,
    "screen_stocks": screen_stocks,
    "get_company_overview": get_company_overview,
    "run_backtest": run_backtest,
    "get_macro_snapshot": get_macro_snapshot,
}


def execute_tool(name: str, **kwargs) -> Any:
    """Execute a tool by name with the given kwargs. Returns result dict."""
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(**kwargs)
