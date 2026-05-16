#!/usr/bin/env python3
"""
DB seed script: pre-load 20 large-cap tickers with 5 years OHLCV and latest fundamentals.
Run on first deployment: python -m db.seed_db
Called from docker entrypoint when FIRST_RUN=1 or when ohlcv is empty.
"""
import os
import sys
import logging
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK-B", "JPM", "GS", "BAC", "V",
    "MA", "XOM", "CVX", "JNJ", "UNH", "LLY", "NVDA", "TSLA", "SPY", "QQQ",
]


def _ohlcv_rows_from_df(symbol: str, df):
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
            "Open": float(row.get("Open", 0)),
            "High": float(row.get("High", 0)),
            "Low": float(row.get("Low", 0)),
            "Close": float(row.get("Close", 0)),
            "Volume": int(row.get("Volume", 0) or 0),
            "Adj Close": row.get("Adj Close") if "Adj Close" in row else row.get("Close"),
        })
    return rows


def seed_ohlcv() -> int:
    """Fetch 5 years OHLCV for SEED_TICKERS and upsert to DB."""
    from core.db import upsert_ohlcv, get_ohlcv_range

    try:
        from core.data_fetcher import DataFetcher
    except ImportError as e:
        logger.error("DataFetcher not available: %s", e)
        return 0

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 5)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    total = 0
    fetcher = DataFetcher()
    for symbol in SEED_TICKERS:
        try:
            df = fetcher.get_stock_data(symbol, start_date=start_str, end_date=end_str, period="5y")
            if df is not None and not df.empty:
                rows = _ohlcv_rows_from_df(symbol, df)
                n = upsert_ohlcv(symbol, rows, source="yfinance")
                total += n
                logger.info("OHLCV %s: %d rows", symbol, n)
        except Exception as e:
            logger.warning("OHLCV %s failed: %s", symbol, e)
    return total


def seed_fundamentals() -> int:
    """Fetch company profile and fundamentals for SEED_TICKERS (FMP if configured, else yfinance profile only)."""
    from core.db import upsert_company_profile, upsert_income_statement, upsert_balance_sheet, upsert_cash_flow

    count = 0

    try:
        from core.data_providers import FMPProvider
        fmp = FMPProvider()
        if fmp.api_key:
            for symbol in SEED_TICKERS:
                try:
                    profile = fmp.fetch_profile(symbol)
                    if profile:
                        upsert_company_profile(symbol, profile)
                        count += 1
                    for row in fmp.fetch_income_statement(symbol, period="annual", limit=5):
                        pe = row.get("date") or row.get("periodEnd")
                        if pe:
                            if hasattr(pe, "strftime"):
                                pe = pe.strftime("%Y-%m-%d")
                            upsert_income_statement(symbol, str(pe)[:10], "annual", row)
                            count += 1
                    for row in fmp.fetch_balance_sheet(symbol, period="annual", limit=5):
                        pe = row.get("date") or row.get("periodEnd")
                        if pe:
                            if hasattr(pe, "strftime"):
                                pe = pe.strftime("%Y-%m-%d")
                            upsert_balance_sheet(symbol, str(pe)[:10], "annual", row)
                            count += 1
                    for row in fmp.fetch_cash_flow(symbol, period="annual", limit=5):
                        pe = row.get("date") or row.get("periodEnd")
                        if pe:
                            if hasattr(pe, "strftime"):
                                pe = pe.strftime("%Y-%m-%d")
                            upsert_cash_flow(symbol, str(pe)[:10], "annual", row)
                            count += 1
                except Exception as e:
                    logger.warning("FMP %s: %s", symbol, e)
            return count
    except ImportError:
        pass

    try:
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        for symbol in SEED_TICKERS:
            try:
                info = fetcher.get_company_info(symbol)
                if info:
                    upsert_company_profile(symbol, info)
                    count += 1
            except Exception as e:
                logger.warning("Profile %s: %s", symbol, e)
    except ImportError:
        pass

    return count


def _is_already_seeded() -> bool:
    """Return True if DB already has OHLCV data (skip seed on subsequent runs)."""
    try:
        from core.db import get_ohlcv_range
        rows = get_ohlcv_range("SPY", "2020-01-01", "2020-01-10")
        return len(rows) > 0
    except Exception:
        return False


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set, using default")
        os.environ.setdefault("DATABASE_URL", "postgresql://trader:trading123@localhost:5432/trading_metrics")

    if os.environ.get("SKIP_SEED_IF_POPULATED") == "1" and _is_already_seeded():
        logger.info("DB already seeded, skipping")
        return 0

    logger.info("Seeding DB with %d tickers (5y OHLCV + fundamentals)", len(SEED_TICKERS))
    ohlcv_n = seed_ohlcv()
    logger.info("OHLCV: %d rows upserted", ohlcv_n)
    fund_n = seed_fundamentals()
    logger.info("Fundamentals: %d rows upserted", fund_n)
    logger.info("Seed complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
