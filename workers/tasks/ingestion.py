"""
Celery tasks for data ingestion: OHLCV, macro, news, fundamentals.
All data is written to PostgreSQL via core.db.
Logging: start, success (row count), failure (traceback).
Retry: exponential backoff, max 3 retries.
"""
import logging
import traceback
from datetime import datetime, timedelta, timezone

from workers.celery_app import app

logger = logging.getLogger(__name__)

# Default universe for daily OHLCV (configurable via env or DB later)
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V"]


def _ohlcv_rows_from_dataframe(symbol: str, df):
    """Convert a pandas DataFrame with OHLCV columns to list of dicts for upsert_ohlcv."""
    if df is None or df.empty:
        return []
    rows = []
    for ts, row in df.iterrows():
        t = ts
        if hasattr(t, "to_pydatetime"):
            t = t.to_pydatetime()
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        rows.append({
            "time": t,
            "Open": float(row.get("Open", row.get("Open", 0))),
            "High": float(row.get("High", row.get("High", 0))),
            "Low": float(row.get("Low", row.get("Low", 0))),
            "Close": float(row.get("Close", row.get("Close", 0))),
            "Volume": int(row.get("Volume", row.get("Volume", 0)) or 0),
            "Adj Close": row.get("Adj Close") if "Adj Close" in row else row.get("Close"),
        })
    return rows


@app.task(name="workers.tasks.ingestion.refresh_ohlcv_daily", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def refresh_ohlcv_daily(self):
    """Fetch daily OHLCV for default universe and upsert into ohlcv table."""
    import pandas as pd
    from core.db import upsert_ohlcv, update_data_status

    logger.info("[refresh_ohlcv_daily] starting for %s", ",".join(DEFAULT_SYMBOLS))
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        logger.error("[refresh_ohlcv_daily] failed: %s", traceback.format_exc())
        update_data_status("yfinance", "ohlcv", last_error=str(e))
        raise

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 2)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    total = 0
    for symbol in DEFAULT_SYMBOLS:
        try:
            fetcher = DataFetcher()
            df = fetcher.get_stock_data(symbol, start_date=start_str, end_date=end_str, period="2y")
            if df is not None and not df.empty:
                rows = _ohlcv_rows_from_dataframe(symbol, df)
                n = upsert_ohlcv(symbol, rows, source="yfinance")
                total += n
        except Exception as e:
            logger.warning("OHLCV fetch failed for %s: %s", symbol, e)
    update_data_status("yfinance", "ohlcv", last_updated=datetime.now(timezone.utc))
    logger.info("[refresh_ohlcv_daily] complete — %d rows written", total)
    return {"upserted": total, "symbols": len(DEFAULT_SYMBOLS)}


@app.task(name="workers.tasks.ingestion.refresh_macro_weekly", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def refresh_macro_weekly(self):
    """Fetch FRED macro series and upsert into macro_series table."""
    from core.db import upsert_macro_series, update_data_status

    logger.info("[refresh_macro_weekly] starting for macro data")
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        logger.error("[refresh_macro_weekly] failed: %s", traceback.format_exc())
        update_data_status("fred", "macro", last_error=str(e))
        raise

    fetcher = DataFetcher()
    if not fetcher.fred:
        update_data_status("fred", "macro", last_error="FRED API key not configured")
        logger.info("[refresh_macro_weekly] skipped — FRED not configured")
        return {"error": "FRED not configured"}

    series_ids = ["DGS10", "DGS2", "DGS5", "DGS30", "FEDFUNDS", "UNRATE", "CPIAUCSL", "GDP"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 5)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    total = 0
    for sid in series_ids:
        try:
            s = fetcher.get_economic_indicator(sid, start_date=start_str, end_date=end_str)
            if s is not None and not s.empty:
                rows = [{"time": ts, "value": float(v)} for ts, v in s.items()]
                total += upsert_macro_series(sid, rows)
        except Exception as e:
            logger.warning("FRED series %s failed: %s", sid, e)
    update_data_status("fred", "macro", last_updated=datetime.now(timezone.utc))
    logger.info("[refresh_macro_weekly] complete — %d rows written", total)
    return {"upserted": total}


@app.task(name="workers.tasks.ingestion.refresh_news_hourly", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def refresh_news_hourly(self):
    """Fetch news for default symbols and insert into news table."""
    from core.db import insert_news, update_data_status

    logger.info("[refresh_news_hourly] starting for news data")
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        logger.error("[refresh_news_hourly] failed: %s", traceback.format_exc())
        update_data_status("newsapi", "news", last_error=str(e))
        raise

    fetcher = DataFetcher()
    total = 0
    for symbol in DEFAULT_SYMBOLS[:5]:
        try:
            items = fetcher.get_news(symbol, limit=20)
            if items:
                for it in items:
                    it["symbol"] = symbol
                total += insert_news(items)
        except Exception as e:
            logger.warning("News fetch failed for %s: %s", symbol, e)
    update_data_status("newsapi", "news", last_updated=datetime.now(timezone.utc))
    logger.info("[refresh_news_hourly] complete — %d rows written", total)
    return {"inserted": total}


@app.task(name="workers.tasks.ingestion.refresh_fundamentals_quarterly", bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def refresh_fundamentals_quarterly(self):
    """Fetch fundamentals from FMP and upsert into company_profile, income_statement, balance_sheet, cash_flow."""
    from core.db import (
        update_data_status,
        upsert_company_profile,
        upsert_income_statement,
        upsert_balance_sheet,
        upsert_cash_flow,
    )
    from core.data_providers import FMPProvider

    logger.info("[refresh_fundamentals_quarterly] starting for fundamentals")
    try:
        fmp = FMPProvider()
        if not fmp.api_key:
            update_data_status("fmp", "fundamentals", last_error="FMP_API_KEY not configured")
            return {"status": "skipped", "reason": "no_api_key"}
        count_profile = 0
        count_income = 0
        count_balance = 0
        count_cash = 0
        for symbol in DEFAULT_SYMBOLS:
            try:
                profile = fmp.fetch_profile(symbol)
                if profile:
                    upsert_company_profile(symbol, profile)
                    count_profile += 1
                for period_type in ("annual", "quarterly"):
                    for row in fmp.fetch_income_statement(symbol, period=period_type, limit=8):
                        period_end = row.get("date") or row.get("periodEnd")
                        if period_end:
                            if hasattr(period_end, "strftime"):
                                period_end = period_end.strftime("%Y-%m-%d")
                            upsert_income_statement(symbol, period_end, period_type, row)
                            count_income += 1
                    for row in fmp.fetch_balance_sheet(symbol, period=period_type, limit=8):
                        period_end = row.get("date") or row.get("periodEnd")
                        if period_end:
                            if hasattr(period_end, "strftime"):
                                period_end = period_end.strftime("%Y-%m-%d")
                            upsert_balance_sheet(symbol, period_end, period_type, row)
                            count_balance += 1
                    for row in fmp.fetch_cash_flow(symbol, period=period_type, limit=8):
                        period_end = row.get("date") or row.get("periodEnd")
                        if period_end:
                            if hasattr(period_end, "strftime"):
                                period_end = period_end.strftime("%Y-%m-%d")
                            upsert_cash_flow(symbol, period_end, period_type, row)
                            count_cash += 1
            except Exception as e:
                logger.warning("FMP fundamentals failed for %s: %s", symbol, e)
        update_data_status("fmp", "fundamentals", last_updated=datetime.now(timezone.utc))
        row_count = count_profile + count_income + count_balance + count_cash
        logger.info("[refresh_fundamentals_quarterly] complete — %d rows written", row_count)
        return {
            "status": "ok",
            "profiles": count_profile,
            "income": count_income,
            "balance": count_balance,
            "cash_flow": count_cash,
        }
    except Exception as e:
        logger.error("[refresh_fundamentals_quarterly] failed: %s", traceback.format_exc())
        update_data_status("fmp", "fundamentals", last_error=str(e))
        raise
