"""
Quant Research Lab API: factor builder, pairs trading (cointegration), options pricer with live data.
Backtest endpoint uses real OHLCV from DB.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.backtest_contracts import StrategyBacktestRequest, strategy_params_from_request

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Backtest request/response ---
BacktestRequest = StrategyBacktestRequest


def _get_prices_for_symbols(symbols: List[str], days: int = 252) -> Any:
    """Return DataFrame of close prices from DB or DataFetcher."""
    from core.db import get_ohlcv_range
    import pandas as pd
    end = datetime.now(timezone.utc)
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    data: Dict[str, List[float]] = {}
    dates: List[str] = []
    for sym in symbols:
        rows = get_ohlcv_range(sym, start, end_str)
        if rows:
            if not dates:
                dates = [r["time"][:10] for r in rows]
            data[sym] = [r["close"] for r in rows]
    if not data:
        try:
            from core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            for sym in symbols[:10]:
                df_one = fetcher.get_stock_data(sym, period=f"{max(1, days // 252)}y")
                if df_one is not None and not df_one.empty and "Close" in df_one.columns:
                    if not dates:
                        dates = [str(ts)[:10] for ts in df_one.index]
                    data[sym] = df_one["Close"].tolist()
            if not data:
                return None
        except Exception as e:
            logger.warning("Price fetch fallback failed: %s", e)
            return None
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    return df


@router.post("/factor-rank")
async def factor_rank(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,V,PG,JNJ", description="Comma-separated symbols"),
    factors: str = Query("momentum,value", description="Comma-separated: momentum, value, quality, low_vol, size"),
) -> Dict[str, Any]:
    """Rank symbols by selected factors using real price/fundamental data from DB or fetcher.
    Momentum: 12-1m return from OHLCV. Value: P/E, P/B from fundamentals. Quality: ROE, D/E.
    Low-vol: 60d realized vol. Size: log(market cap).
    """
    import numpy as np
    from core.db import get_income_statements, get_balance_sheets, get_company_profile

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:50]
    factor_list = [f.strip().lower() for f in factors.split(",") if f.strip()]
    if not sym_list or not factor_list:
        raise HTTPException(status_code=400, detail="Provide symbols and factors")
    prices = _get_prices_for_symbols(sym_list)
    if prices is None or prices.empty:
        raise HTTPException(status_code=404, detail="No price data for symbols")
    closes = prices
    if hasattr(closes, "columns") and len(closes.columns) == 1 and sym_list:
        single = closes.columns[0] if list(closes.columns) else sym_list[0]
        closes = closes.rename(columns={closes.columns[0]: single})
    returns = closes.pct_change().dropna()

    mom = None
    if "momentum" in factor_list:
        mom = (closes.iloc[-1] / closes.iloc[max(0, len(closes) - 252)] - 1) if len(closes) >= 252 else (closes.iloc[-1] / closes.iloc[0] - 1)
        mom = mom.fillna(0)

    low_vol = None
    if "low_vol" in factor_list and len(returns) >= 60:
        vol_60d = returns.tail(60).std() * np.sqrt(252)
        low_vol = -vol_60d  # lower vol = higher score

    value_scores = {}
    if "value" in factor_list:
        for sym in sym_list:
            pe, pb = None, None
            inc = get_income_statements(sym, "annual", 1)
            bal = get_balance_sheets(sym, "annual", 1)
            prof = get_company_profile(sym)
            mcap = float(prof["market_cap"]) if prof and prof.get("market_cap") else None
            if not mcap:
                try:
                    from core.data_fetcher import DataFetcher
                    info = DataFetcher().get_company_info(sym)
                    mcap = float(info.get("marketCap") or 0) if info else None
                except Exception:
                    pass
            ni = None
            if inc and inc[0].get("data"):
                d = inc[0]["data"]
                ni = d.get("netIncome") or d.get("NetIncome") or d.get("net_income")
            equity = None
            debt = None
            if bal and bal[0].get("data"):
                d = bal[0]["data"]
                equity = d.get("totalStockholderEquity") or d.get("TotalShareholdersEquity") or d.get("totalEquity")
                debt = d.get("totalDebt") or d.get("TotalDebt") or d.get("longTermDebt", 0) + d.get("shortTermDebt", 0)
            if mcap and ni and ni > 0:
                pe = mcap / ni
            if mcap and equity and equity > 0:
                pb = mcap / equity
            if pe is not None or pb is not None:
                val = 0.0
                if pe is not None:
                    val += 1.0 / min(pe, 100)  # lower PE = higher score
                if pb is not None:
                    val += 1.0 / min(pb, 20)
                value_scores[sym] = val

    quality_scores = {}
    if "quality" in factor_list:
        for sym in sym_list:
            inc = get_income_statements(sym, "annual", 1)
            bal = get_balance_sheets(sym, "annual", 1)
            ni = None
            if inc and inc[0].get("data"):
                d = inc[0]["data"]
                ni = d.get("netIncome") or d.get("NetIncome")
            equity = None
            debt = None
            if bal and bal[0].get("data"):
                d = bal[0]["data"]
                equity = d.get("totalStockholderEquity") or d.get("TotalShareholdersEquity")
                debt = d.get("totalDebt") or d.get("TotalDebt") or 0
            roe = (ni / equity) if equity and equity > 0 and ni else None
            de = (debt / equity) if equity and equity > 0 and debt else None
            q = 0.0
            if roe is not None:
                q += min(roe, 1.0)
            if de is not None:
                q -= min(de / 2, 1.0)  # lower D/E = higher score
            if roe is not None or de is not None:
                quality_scores[sym] = q

    size_scores = {}
    if "size" in factor_list:
        for sym in sym_list:
            prof = get_company_profile(sym)
            mcap = float(prof["market_cap"]) if prof and prof.get("market_cap") else None
            if mcap and mcap > 0:
                size_scores[sym] = np.log1p(mcap)

    rows = []
    for sym in (closes.columns if hasattr(closes, "columns") else sym_list):
        s = sym if isinstance(sym, str) else str(sym)
        scores_per_factor = {}
        composite = 0.0
        n = 0
        if mom is not None and s in mom.index:
            scores_per_factor["momentum"] = round(float(mom[s]), 4)
            composite += float(mom[s])
            n += 1
        if low_vol is not None and s in low_vol.index:
            scores_per_factor["low_vol"] = round(float(low_vol[s]), 4)
            composite += float(low_vol[s])
            n += 1
        if s in value_scores:
            scores_per_factor["value"] = round(value_scores[s], 4)
            composite += value_scores[s]
            n += 1
        if s in quality_scores:
            scores_per_factor["quality"] = round(quality_scores[s], 4)
            composite += quality_scores[s]
            n += 1
        if s in size_scores:
            scores_per_factor["size"] = round(size_scores[s], 4)
            composite += size_scores[s]
            n += 1
        if n > 0:
            rows.append({"symbol": s, "scores": scores_per_factor, "composite": round(composite / n, 4)})
    ranked = sorted(rows, key=lambda x: -x["composite"])
    return {"factors": factor_list, "ranked": ranked, "count": len(ranked)}


@router.get("/pairs")
async def pairs_trading_analysis(
    symbol1: str = Query(..., description="First symbol"),
    symbol2: str = Query(..., description="Second symbol"),
    period_days: int = Query(252, ge=60, le=2520),
) -> Dict[str, Any]:
    """Engle-Granger cointegration test, spread, z-score, and simple backtest on two symbols."""
    import numpy as np
    import pandas as pd
    s1, s2 = symbol1.upper(), symbol2.upper()
    prices = _get_prices_for_symbols([s1, s2], days=period_days)
    if prices is None or prices.empty:
        raise HTTPException(status_code=404, detail="No price data for pair")
    if hasattr(prices, "columns"):
        cols = list(prices.columns)
        if len(cols) < 2:
            if len(cols) == 1:
                raise HTTPException(status_code=400, detail="Need data for both symbols")
            raise HTTPException(status_code=404, detail="Insufficient data")
        c1, c2 = cols[0], cols[1]
    else:
        c1, c2 = s1, s2
    df = prices.dropna()
    if len(df) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 observations")
    try:
        from arch.unitroot import engle_granger
        res = engle_granger(df[c1], df[c2], trend="c")
        coint = not res.pvalue > 0.05  # reject non-cointegration at 5%
        pvalue = float(res.pvalue)
        test_stat = float(res.test_statistic)
    except ImportError:
        coint = False
        pvalue = 1.0
        test_stat = 0.0
    # Spread: regress y on x, spread = y - beta*x
    from sklearn.linear_model import LinearRegression
    X = df[c2].values.reshape(-1, 1)
    y = df[c1].values
    lr = LinearRegression().fit(X, y)
    beta = float(lr.coef_[0])
    spread = y - (X.flatten() * beta)
    spread_series = pd.Series(spread, index=df.index)
    zscore = (spread_series - spread_series.mean()) / spread_series.std() if spread_series.std() > 0 else spread_series * 0
    zscore_last = float(zscore.iloc[-1]) if len(zscore) else 0
    # Simple backtest: long spread when z < -2, short when z > 2
    signals = np.where(zscore < -2, 1, np.where(zscore > 2, -1, 0))
    ret1 = df[c1].pct_change()
    ret2 = df[c2].pct_change()
    strat_ret = pd.Series(0.0, index=df.index)
    strat_ret.iloc[1:] = (signals[:-1] * (ret1.iloc[1:].values - beta * ret2.iloc[1:].values))
    cum = (1 + strat_ret).cumprod()
    total_return = float(cum.iloc[-1] - 1) if len(cum) else 0
    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0
    # Spread and z-score time series for frontend chart
    spread_series_out = [{"date": str(idx)[:10], "spread": round(float(v), 4)} for idx, v in spread_series.items()]
    zscore_series_out = [{"date": str(idx)[:10], "zscore": round(float(v), 4)} for idx, v in zscore.items()]
    signals_out = [{"date": str(df.index[i])[:10], "signal": "enter_long" if s == 1 else "enter_short" if s == -1 else "hold"} for i, s in enumerate(signals) if s != 0]
    return {
        "symbol1": s1,
        "symbol2": s2,
        "cointegrated": coint,
        "pvalue": round(pvalue, 4),
        "test_statistic": round(test_stat, 4),
        "hedge_ratio_beta": round(beta, 4),
        "spread_mean": round(float(spread_series.mean()), 4),
        "spread_std": round(float(spread_series.std()), 4),
        "zscore_current": round(zscore_last, 4),
        "backtest_total_return_pct": round(total_return * 100, 2),
        "backtest_sharpe": round(float(sharpe), 4),
        "spread_series": spread_series_out[-252:],
        "zscore_series": zscore_series_out[-252:],
        "signals": signals_out[-50:],
    }


@router.get("/options-pricer")
async def options_pricer(
    ticker: str = Query(..., description="Underlying symbol"),
    strike: float = Query(..., gt=0),
    expiry_days: int = Query(30, ge=1, le=3650),
    option_type: str = Query("call", regex="^(call|put)$"),
    volatility: Optional[float] = Query(None, description="Override vol; if not set, use 0.30"),
) -> Dict[str, Any]:
    """Options pricer with live spot from DB/fetcher and risk-free rate from FRED. Returns price and Greeks."""
    spot = None
    from core.db import get_ohlcv_range
    end = datetime.now(timezone.utc)
    start = (end - timedelta(days=5)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    rows = get_ohlcv_range(ticker.upper(), start, end_str)
    if rows:
        spot = float(rows[-1]["close"])
    if spot is None:
        try:
            from core.data_fetcher import DataFetcher
            df = DataFetcher().get_stock_data(ticker, period="5d")
            if df is not None and not df.empty and "Close" in df.columns:
                spot = float(df["Close"].iloc[-1])
        except Exception as e:
            logger.warning("Spot fetch failed: %s", e)
    if spot is None or spot <= 0:
        raise HTTPException(status_code=404, detail=f"Could not get spot price for {ticker}")
    rfr = 0.05
    try:
        from core.data_fetcher import DataFetcher
        fred_series = DataFetcher().get_economic_indicator("DGS3MO", start, end_str)
        if fred_series is not None and not fred_series.empty:
            rfr = float(fred_series.iloc[-1]) / 100.0
    except Exception:
        pass
    vol = volatility if volatility is not None and volatility > 0 else 0.30
    time_years = expiry_days / 365.0
    try:
        from models.derivatives.option_pricing import BlackScholes, OptionAnalyzer
        if option_type.lower() == "call":
            price = BlackScholes.call_price(spot, strike, time_years, rfr, vol, 0.0)
        else:
            price = BlackScholes.put_price(spot, strike, time_years, rfr, vol, 0.0)
        analysis = OptionAnalyzer.analyze_option(option_type.lower(), spot, strike, time_years, rfr, vol, 0.0)
        return {
            "ticker": ticker.upper(),
            "spot_price": round(spot, 2),
            "strike": strike,
            "expiry_days": expiry_days,
            "option_type": option_type.lower(),
            "volatility": vol,
            "risk_free_rate": round(rfr, 4),
            "option_price": round(price, 4),
            "greeks": {
                "delta": round(analysis["delta"], 4),
                "gamma": round(analysis["gamma"], 6),
                "theta": round(analysis.get("theta", 0), 4),
                "vega": round(analysis.get("vega", 0), 4),
                "rho": round(analysis.get("rho", 0), 4),
            },
        }
    except Exception as e:
        logger.exception("Options pricer failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def run_quant_backtest(req: BacktestRequest) -> Dict[str, Any]:
    """
    Run backtest using real OHLCV from DB. Strategies: sma_cross, rsi_mean_reversion, factor_momentum.
    Returns equity curve, trade log, Sharpe, Sortino, CAGR, max drawdown, win rate, alpha/beta vs SPY.
    """
    params = strategy_params_from_request(req)
    try:
        from core.backtest_api_adapter import run_backtest_contract

        return run_backtest_contract(
            symbol=req.symbol.upper(),
            strategy=req.strategy,
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital,
            commission=req.commission,
            model_name=req.strategy,
            strategy_params=params,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backtest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options-chain/{ticker}")
async def get_options_chain(ticker: str) -> Dict[str, Any]:
    """Fetch full options chain from yfinance (calls + puts) with market price vs model price and IV."""
    try:
        import yfinance as yf
        from models.derivatives.option_pricing import BlackScholes

        sym = yf.Ticker(ticker.upper())
        expirations = sym.options
        if not expirations:
            return {"ticker": ticker.upper(), "calls": [], "puts": [], "error": "No option expirations available"}

        info = sym.info
        spot = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0)
        if spot <= 0:
            q = sym.history(period="5d")
            if q is not None and not q.empty and "Close" in q.columns:
                spot = float(q["Close"].iloc[-1])
        if spot <= 0:
            raise HTTPException(status_code=404, detail=f"Could not get spot price for {ticker}")

        rfr = 0.05
        try:
            from core.data_fetcher import DataFetcher
            end = datetime.now(timezone.utc)
            start = (end - timedelta(days=5)).strftime("%Y-%m-%d")
            fred = DataFetcher().get_economic_indicator("DGS3MO", start, end.strftime("%Y-%m-%d"))
            if fred is not None and not fred.empty:
                rfr = float(fred.iloc[-1]) / 100.0
        except Exception:
            pass

        calls_out: List[Dict] = []
        puts_out: List[Dict] = []

        for exp_str in expirations[:5]:
            chain = sym.option_chain(exp_str)
            if chain.calls is not None and not chain.calls.empty:
                for _, row in chain.calls.iterrows():
                    strike = float(row.get("strike", 0))
                    if strike <= 0:
                        continue
                    vol = float(row.get("impliedVolatility") or 0.30)
                    market = float(row.get("lastPrice") or 0)
                    t = (pd.to_datetime(exp_str) - datetime.now(timezone.utc)).days / 365.25
                    model = BlackScholes.call_price(spot, strike, max(1 / 365, t), rfr, vol, 0.0) if t > 0 else 0
                    calls_out.append({
                        "expiry": exp_str,
                        "strike": strike,
                        "market_price": round(market, 4),
                        "model_price": round(model, 4),
                        "iv": round(vol, 4),
                        "bid": float(row.get("bid", 0) or 0),
                        "ask": float(row.get("ask", 0) or 0),
                    })
            if chain.puts is not None and not chain.puts.empty:
                for _, row in chain.puts.iterrows():
                    strike = float(row.get("strike", 0))
                    if strike <= 0:
                        continue
                    vol = float(row.get("impliedVolatility") or 0.30)
                    market = float(row.get("lastPrice") or 0)
                    t = (pd.to_datetime(exp_str) - datetime.now(timezone.utc)).days / 365.25
                    model = BlackScholes.put_price(spot, strike, max(1 / 365, t), rfr, vol, 0.0) if t > 0 else 0
                    puts_out.append({
                        "expiry": exp_str,
                        "strike": strike,
                        "market_price": round(market, 4),
                        "model_price": round(model, 4),
                        "iv": round(vol, 4),
                        "bid": float(row.get("bid", 0) or 0),
                        "ask": float(row.get("ask", 0) or 0),
                    })

        return {
            "ticker": ticker.upper(),
            "spot_price": round(spot, 2),
            "risk_free_rate": round(rfr, 4),
            "expirations": expirations[:5],
            "calls": calls_out[:100],
            "puts": puts_out[:100],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Options chain failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
