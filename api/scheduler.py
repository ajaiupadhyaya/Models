"""
In-process job scheduler (APScheduler) replacing Celery+Redis for free-tier hosting.

Disabled by default; enable with SCHEDULER_ENABLED=true.
Jobs only run when DATABASE_URL is set — otherwise they no-op (nothing to write).

Schedule mirrors workers/celery_app.py (kept for v2 reference):
- refresh_ohlcv_daily   — every 24h
- refresh_macro_weekly  — every 7d
- refresh_news_hourly   — every 1h
- refresh_fundamentals_quarterly — every 90d

Started from api.main.lifespan; shut down cleanly on app exit.
"""
from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_scheduler = None

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V"]


def _ohlcv_rows(symbol: str, df) -> list:
    if df is None or df.empty:
        return []
    rows = []
    for ts, row in df.iterrows():
        t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if hasattr(t, "tzinfo") and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        rows.append({
            "time": t,
            "Open": float(row.get("Open", 0)),
            "High": float(row.get("High", 0)),
            "Low": float(row.get("Low", 0)),
            "Close": float(row.get("Close", 0)),
            "Volume": int(row.get("Volume", 0) or 0),
            "Adj Close": row.get("Adj Close") if "Adj Close" in row else row.get("Close"),
        })
    return rows


def refresh_ohlcv_daily() -> dict:
    from core.db import upsert_ohlcv, update_data_status, db_configured
    if not db_configured():
        return {"status": "skipped", "reason": "no_db"}
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        update_data_status("yfinance", "ohlcv", last_error=str(e))
        logger.error("ohlcv refresh import failed: %s", e)
        return {"status": "error", "reason": str(e)}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 2)
    total = 0
    fetcher = DataFetcher()
    for symbol in DEFAULT_SYMBOLS:
        try:
            df = fetcher.get_stock_data(
                symbol,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                period="2y",
            )
            if df is not None and not df.empty:
                total += upsert_ohlcv(symbol, _ohlcv_rows(symbol, df), source="yfinance")
        except Exception as e:
            logger.warning("ohlcv %s failed: %s", symbol, e)
    update_data_status("yfinance", "ohlcv", last_updated=datetime.now(timezone.utc))
    return {"upserted": total, "symbols": len(DEFAULT_SYMBOLS)}


def refresh_macro_weekly() -> dict:
    from core.db import upsert_macro_series, update_data_status, db_configured
    if not db_configured():
        return {"status": "skipped", "reason": "no_db"}
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        update_data_status("fred", "macro", last_error=str(e))
        return {"status": "error", "reason": str(e)}
    fetcher = DataFetcher()
    if not getattr(fetcher, "fred", None):
        update_data_status("fred", "macro", last_error="FRED API key not configured")
        return {"status": "skipped", "reason": "no_fred_key"}

    series_ids = ["DGS10", "DGS2", "DGS5", "DGS30", "FEDFUNDS", "UNRATE", "CPIAUCSL", "GDP"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 5)
    total = 0
    for sid in series_ids:
        try:
            s = fetcher.get_economic_indicator(
                sid,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            if s is not None and not s.empty:
                rows = [{"time": ts, "value": float(v)} for ts, v in s.items()]
                total += upsert_macro_series(sid, rows)
        except Exception as e:
            logger.warning("macro %s failed: %s", sid, e)
    update_data_status("fred", "macro", last_updated=datetime.now(timezone.utc))
    return {"upserted": total}


def refresh_news_hourly() -> dict:
    from core.db import insert_news, update_data_status, db_configured
    if not db_configured():
        return {"status": "skipped", "reason": "no_db"}
    try:
        from core.data_fetcher import DataFetcher
    except Exception as e:
        update_data_status("newsapi", "news", last_error=str(e))
        return {"status": "error", "reason": str(e)}
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
            logger.warning("news %s failed: %s", symbol, e)
    update_data_status("newsapi", "news", last_updated=datetime.now(timezone.utc))
    return {"inserted": total}


def refresh_fundamentals_quarterly() -> dict:
    from core.db import (
        update_data_status, upsert_company_profile, upsert_income_statement,
        upsert_balance_sheet, upsert_cash_flow, db_configured,
    )
    if not db_configured():
        return {"status": "skipped", "reason": "no_db"}
    try:
        from core.data_providers import FMPProvider
        fmp = FMPProvider()
        if not fmp.api_key:
            update_data_status("fmp", "fundamentals", last_error="FMP_API_KEY not configured")
            return {"status": "skipped", "reason": "no_key"}
        counts = {"profile": 0, "income": 0, "balance": 0, "cash": 0}
        for symbol in DEFAULT_SYMBOLS:
            try:
                profile = fmp.fetch_profile(symbol)
                if profile:
                    upsert_company_profile(symbol, profile)
                    counts["profile"] += 1
                for period_type in ("annual", "quarterly"):
                    for upsert_fn, fetch_fn, key in (
                        (upsert_income_statement, fmp.fetch_income_statement, "income"),
                        (upsert_balance_sheet, fmp.fetch_balance_sheet, "balance"),
                        (upsert_cash_flow, fmp.fetch_cash_flow, "cash"),
                    ):
                        for row in fetch_fn(symbol, period=period_type, limit=8):
                            pe = row.get("date") or row.get("periodEnd")
                            if pe:
                                if hasattr(pe, "strftime"):
                                    pe = pe.strftime("%Y-%m-%d")
                                upsert_fn(symbol, pe, period_type, row)
                                counts[key] += 1
            except Exception as e:
                logger.warning("fmp %s failed: %s", symbol, e)
        update_data_status("fmp", "fundamentals", last_updated=datetime.now(timezone.utc))
        return {"status": "ok", **counts}
    except Exception as e:
        logger.error("fundamentals refresh failed: %s", traceback.format_exc())
        update_data_status("fmp", "fundamentals", last_error=str(e))
        return {"status": "error", "reason": str(e)}


def start_scheduler() -> Optional[object]:
    """Start APScheduler when SCHEDULER_ENABLED=true. Returns scheduler or None."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    if os.getenv("SCHEDULER_ENABLED", "false").strip().lower() not in ("1", "true", "yes"):
        logger.info("scheduler disabled (set SCHEDULER_ENABLED=true to enable)")
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        logger.warning("apscheduler not installed; scheduler disabled")
        return None

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(refresh_ohlcv_daily, "interval", hours=24, id="ohlcv_daily", max_instances=1, coalesce=True)
    sched.add_job(refresh_macro_weekly, "interval", days=7, id="macro_weekly", max_instances=1, coalesce=True)
    sched.add_job(refresh_news_hourly, "interval", hours=1, id="news_hourly", max_instances=1, coalesce=True)
    sched.add_job(refresh_fundamentals_quarterly, "interval", days=90, id="fundamentals_q", max_instances=1, coalesce=True)
    sched.start()
    _scheduler = sched
    logger.info("scheduler started with 4 jobs")
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning("scheduler shutdown error: %s", e)
        _scheduler = None
