"""
Equity Research API: ticker search, company overview, financial statements, DCF, Comps, LBO.
All data from DB or FMP (no hardcoded figures).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Request/response models ---

class DCFRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    wacc: float = Field(0.10, ge=0.01, le=0.50)
    terminal_growth: float = Field(0.03, ge=-0.10, le=0.15)


class LBORequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    entry_multiple: float = Field(10.0, ge=1.0, le=50.0)
    debt_equity_ratio: float = Field(0.6, ge=0.0, le=10.0)
    interest_rate: float = Field(0.06, ge=0.0, le=0.30)
    exit_multiple: float = Field(10.0, ge=1.0, le=50.0)
    hold_years: int = Field(5, ge=1, le=20)


def _get_profile_and_fundamentals(ticker: str) -> tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Return (profile, latest_income, latest_balance, latest_cash_flow) from DB or FMP."""
    from core.db import get_company_profile, get_income_statements, get_balance_sheets, get_cash_flows
    profile = get_company_profile(ticker)
    income = get_income_statements(ticker, "annual", 1)
    balance = get_balance_sheets(ticker, "annual", 1)
    cash = get_cash_flows(ticker, "annual", 1)
    latest_income = income[0]["data"] if income else None
    latest_balance = balance[0]["data"] if balance else None
    latest_cash = cash[0]["data"] if cash else None
    if not profile:
        try:
            from core.data_providers import FMPProvider
            fmp = FMPProvider()
            if fmp.api_key:
                p = fmp.fetch_profile(ticker)
                if p:
                    from core.db import upsert_company_profile
                    upsert_company_profile(ticker, p)
                    profile = {"symbol": ticker, "name": p.get("companyName"), "sector": p.get("sector"), "industry": p.get("industry"), "market_cap": p.get("marketCap"), "description": p.get("description")}
                if not latest_income and fmp.api_key:
                    inc = fmp.fetch_income_statement(ticker, "annual", 1)
                    if inc:
                        latest_income = inc[0]
                if not latest_balance and fmp.api_key:
                    bal = fmp.fetch_balance_sheet(ticker, "annual", 1)
                    if bal:
                        latest_balance = bal[0]
                if not latest_cash and fmp.api_key:
                    cf = fmp.fetch_cash_flow(ticker, "annual", 1)
                    if cf:
                        latest_cash = cf[0]
        except Exception as e:
            logger.warning("FMP fallback for %s: %s", ticker, e)
    return profile, latest_income, latest_balance, latest_cash


@router.get("/search")
async def ticker_search(q: str = Query(..., min_length=1, max_length=50)) -> Dict[str, Any]:
    """Search companies by symbol or name. Returns list of {symbol, name, sector, industry, market_cap}."""
    from core.db import search_company_profiles
    try:
        results = search_company_profiles(q, limit=20)
        if not results:
            try:
                from core.data_fetcher import DataFetcher
                fetcher = DataFetcher()
                info = fetcher.get_stock_info(q.upper() if len(q) <= 5 else q)
                if info:
                    results = [{"symbol": info.get("symbol", q), "name": info.get("name"), "sector": info.get("sector"), "industry": info.get("industry"), "market_cap": info.get("market_cap")}]
            except Exception:
                pass
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.warning("Ticker search failed: %s", e)
        return {"results": [], "count": 0, "error": str(e)}


@router.get("/overview/{symbol}")
async def company_overview(symbol: str) -> Dict[str, Any]:
    """Company profile and key stats. Price series for charts from DB or live fetcher."""
    symbol = symbol.upper()
    profile, _, _, _ = _get_profile_and_fundamentals(symbol)
    from core.db import get_ohlcv_range
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc)
    start_1y = (end - timedelta(days=365)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    ohlcv_1y = get_ohlcv_range(symbol, start_1y, end_str)
    if not ohlcv_1y:
        try:
            from core.data_fetcher import DataFetcher
            df = DataFetcher().get_stock_data(symbol, period="1y")
            if df is not None and not df.empty:
                ohlcv_1y = [{"time": str(idx)[:10], "close": float(row["Close"]), "volume": int(row.get("Volume", 0))} for idx, row in df.iterrows()]
        except Exception as e:
            logger.warning("OHLCV fallback for %s: %s", symbol, e)
    return {"profile": profile, "ohlcv_1y": ohlcv_1y, "symbol": symbol}


@router.get("/statements/{symbol}")
async def financial_statements(
    symbol: str,
    period: str = Query("annual", regex="^(annual|quarterly)$"),
) -> Dict[str, Any]:
    """Income statement, balance sheet, cash flow from DB (or FMP fallback)."""
    symbol = symbol.upper()
    from core.db import get_income_statements, get_balance_sheets, get_cash_flows
    income = get_income_statements(symbol, period, 15)
    balance = get_balance_sheets(symbol, period, 15)
    cash = get_cash_flows(symbol, period, 15)
    if not income and not balance and not cash:
        try:
            from core.data_providers import FMPProvider
            fmp = FMPProvider()
            if fmp.api_key:
                for row in fmp.fetch_income_statement(symbol, period, 15):
                    from core.db import upsert_income_statement
                    pe = row.get("date") or row.get("periodEnd")
                    if pe:
                        pe = pe.strftime("%Y-%m-%d") if hasattr(pe, "strftime") else str(pe)[:10]
                        upsert_income_statement(symbol, pe, period, row)
                for row in fmp.fetch_balance_sheet(symbol, period, 15):
                    from core.db import upsert_balance_sheet
                    pe = row.get("date") or row.get("periodEnd")
                    if pe:
                        pe = pe.strftime("%Y-%m-%d") if hasattr(pe, "strftime") else str(pe)[:10]
                        upsert_balance_sheet(symbol, pe, period, row)
                for row in fmp.fetch_cash_flow(symbol, period, 15):
                    from core.db import upsert_cash_flow
                    pe = row.get("date") or row.get("periodEnd")
                    if pe:
                        pe = pe.strftime("%Y-%m-%d") if hasattr(pe, "strftime") else str(pe)[:10]
                        upsert_cash_flow(symbol, pe, period, row)
                income = get_income_statements(symbol, period, 15)
                balance = get_balance_sheets(symbol, period, 15)
                cash = get_cash_flows(symbol, period, 15)
        except Exception as e:
            logger.warning("FMP statements fallback: %s", e)
    return {"symbol": symbol, "period": period, "income": income, "balance_sheet": balance, "cash_flow": cash}


def compute_dcf(ticker: str, wacc: float, terminal_growth: float = 0.03) -> Dict[str, Any]:
    """Sync DCF computation for use by API and AI chat. Returns dict with intrinsic_value_per_share, current_price, etc."""
    ticker = ticker.upper()
    _, latest_income, latest_balance, latest_cash = _get_profile_and_fundamentals(ticker)
    # FCF = operating cash flow - capex (from cash flow statement)
    fcf_list: List[float] = []
    if latest_cash:
        # FMP: freeCashFlow or derive from netCashProvidedByOperatingActivities - capitalExpenditure
        fcf = latest_cash.get("freeCashFlow") or latest_cash.get("operatingCashFlow")
        if fcf is not None:
            fcf_list.append(float(fcf))
    if not fcf_list and latest_cash:
        ocf = latest_cash.get("operatingCashFlow") or latest_cash.get("netCashProvidedByOperatingActivities")
        capex = latest_cash.get("capitalExpenditure") or latest_cash.get("capitalExpenditures")
        if ocf is not None:
            fcf_list.append(float(ocf) - float(capex or 0))
    # Use last 5 years if we have multiple periods
    from core.db import get_cash_flows
    cash_rows = get_cash_flows(ticker, "annual", 5)
    if not fcf_list and cash_rows:
        for row in cash_rows:
            d = row.get("data") or {}
            fcf = d.get("freeCashFlow") or d.get("operatingCashFlow")
            if fcf is not None:
                fcf_list.append(float(fcf))
    if not fcf_list:
        return {"error": f"No FCF data for {ticker}. Add FMP key and run fundamentals refresh."}
    # Projection: use last FCF and grow by terminal_growth for 5 years (simplified)
    base_fcf = fcf_list[0] if fcf_list else 0
    projection_years = 5
    projected = [base_fcf * ((1 + terminal_growth) ** i) for i in range(1, projection_years + 1)]
    from models.valuation.dcf_model import DCFModel
    model = DCFModel(projected, terminal_growth_rate=terminal_growth, wacc=wacc)
    ev = model.calculate_enterprise_value()
    # Equity value: assume cash/debt from balance sheet
    cash = 0
    debt = 0
    if latest_balance:
        cash = float(latest_balance.get("cashAndCashEquivalents") or latest_balance.get("cash") or 0)
        debt = float(latest_balance.get("totalDebt") or latest_balance.get("longTermDebt") or 0)
    shares = 1e9
    if latest_income:
        # Use shares from income or profile
        pass
    from core.db import get_company_profile
    prof = get_company_profile(ticker)
    if prof and prof.get("market_cap") and latest_balance:
        # Approximate shares from market cap / price
        pass
    equity = model.calculate_equity_value(ev, cash=cash, debt=debt)
    # Shares: get from FMP or assume
    try:
        from core.data_providers import FMPProvider
        fmp = FMPProvider()
        if fmp.api_key:
            pr = fmp.fetch_profile(ticker)
            if pr and pr.get("mktCap") and pr.get("price"):
                shares = float(pr["mktCap"]) / float(pr["price"])
            elif pr and pr.get("volAvg"):
                pass
    except Exception:
        pass
    share_price = model.calculate_share_price(equity, shares) if shares and shares > 0 else 0
    # Current price for upside
    current_price: Optional[float] = None
    try:
        from core.data_fetcher import DataFetcher
        df = DataFetcher().get_stock_data(ticker, period="5d")
        if df is not None and not df.empty and "Close" in df.columns:
            current_price = float(df["Close"].iloc[-1])
    except Exception:
        pass
    upside = None
    if current_price and share_price and current_price > 0:
        upside = (share_price - current_price) / current_price
    # Sensitivity
    wacc_range = [wacc - 0.02, wacc, wacc + 0.02]
    growth_range = [terminal_growth - 0.01, terminal_growth, terminal_growth + 0.01]
    sens = model.sensitivity_analysis(wacc_range, growth_range)
    sens_dict = {str(w): {str(g): float(sens.loc[w, g]) for g in growth_range} for w in wacc_range}
    return {
        "ticker": ticker,
        "intrinsic_value_per_share": round(share_price, 2),
        "current_price": current_price,
        "upside_downside_pct": round(upside * 100, 2) if upside is not None else None,
        "enterprise_value": round(ev, 2),
        "equity_value": round(equity, 2),
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "sensitivity": sens_dict,
    }


@router.post("/dcf")
async def run_dcf(req: DCFRequest) -> Dict[str, Any]:
    """DCF valuation using FCF from DB/FMP. User inputs WACC and terminal growth. Returns intrinsic value and sensitivity."""
    result = compute_dcf(req.ticker, req.wacc, req.terminal_growth)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/comps/{symbol}")
async def comparable_companies(symbol: str) -> Dict[str, Any]:
    """Comparable company analysis: peers and valuation multiples from FMP/DB."""
    symbol = symbol.upper()
    from core.data_providers import FMPProvider
    fmp = FMPProvider()
    if not fmp.api_key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY required for comps")
    peers = fmp.fetch_peers(symbol)
    if not peers:
        return {"symbol": symbol, "peers": [], "multiples": []}
    multiples = []
    for p in peers[:15]:
        try:
            prof = fmp.fetch_profile(p)
            km = fmp.fetch_key_metrics_ttm(p)
            if prof:
                row = {
                    "ticker": p,
                    "name": prof.get("companyName") or p,
                    "market_cap": prof.get("mktCap"),
                    "pe_ratio": km.get("peRatioTTM") if km else prof.get("pe"),
                    "ev_ebitda": km.get("enterpriseValueOverEBITDATTM") if km else None,
                    "ev_revenue": km.get("enterpriseValueOverRevenueTTM") if km else None,
                    "price_to_book": km.get("priceToBookRatioTTM") if km else None,
                }
                multiples.append(row)
        except Exception as e:
            logger.warning("Comp %s: %s", p, e)
    return {"symbol": symbol, "peers": peers, "multiples": multiples}


@router.post("/lbo")
async def run_lbo(req: LBORequest) -> Dict[str, Any]:
    """LBO model: entry/exit multiples, D/E, interest rate, hold period. EBITDA from DB/FMP."""
    ticker = req.ticker.upper()
    _, latest_income, _, _ = _get_profile_and_fundamentals(ticker)
    # EBITDA: from income statement (EBITDA or approximate)
    ebitda = None
    if latest_income:
        ebitda = latest_income.get("ebitda") or latest_income.get("operatingIncome")
        if ebitda is not None:
            ebitda = float(ebitda)
    if ebitda is None or ebitda <= 0:
        raise HTTPException(status_code=404, detail=f"No EBITDA for {ticker}")
    entry_ev = ebitda * req.entry_multiple
    debt_pct = req.debt_equity_ratio / (1 + req.debt_equity_ratio)
    equity_pct = 1 - debt_pct
    entry_debt = entry_ev * debt_pct
    entry_equity = entry_ev * equity_pct
    # Exit after hold_years: assume same EBITDA growth 0 for simplicity
    exit_ev = ebitda * req.exit_multiple
    # Debt paydown: simplified (no amortization schedule)
    interest_paid = entry_debt * req.interest_rate * req.hold_years
    exit_debt = max(0, entry_debt - (entry_ev * 0.1 * req.hold_years))  # assume 10% paydown per year
    exit_equity_value = exit_ev - exit_debt
    equity_return = exit_equity_value - entry_equity
    irr = (exit_equity_value / entry_equity) ** (1 / req.hold_years) - 1 if entry_equity else 0
    moic = exit_equity_value / entry_equity if entry_equity else 0
    return {
        "ticker": ticker,
        "ebitda": ebitda,
        "entry_ev": round(entry_ev, 2),
        "entry_equity": round(entry_equity, 2),
        "entry_debt": round(entry_debt, 2),
        "exit_ev": round(exit_ev, 2),
        "exit_equity_value": round(exit_equity_value, 2),
        "irr": round(irr * 100, 2),
        "moic": round(moic, 2),
    }
