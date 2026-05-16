"""
Database layer for Financial Research Platform.
PostgreSQL + TimescaleDB; all reads/writes go through this module.
Uses DATABASE_URL from environment.
"""
import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Generator

logger = logging.getLogger(__name__)

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://trader:trading123@localhost:5432/trading_metrics",
)


def get_database_url() -> str:
    return _DATABASE_URL


_engine = None


def get_engine():
    """Lazy-create SQLAlchemy engine (synchronous)."""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        _engine = create_engine(
            _DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


@contextmanager
def get_connection():
    """Context manager for a DB connection (for workers and sync code)."""
    engine = get_engine()
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_data_status(source: str, entity: str, last_updated: Optional[datetime] = None, last_error: Optional[str] = None) -> None:
    """Upsert one row into data_status."""
    from sqlalchemy import text
    with get_connection() as conn:
        conn.execute(
            text("""
                INSERT INTO data_status (source, entity, last_updated, last_error)
                VALUES (:source, :entity, :last_updated, :last_error)
                ON CONFLICT (source, entity) DO UPDATE SET
                    last_updated = COALESCE(EXCLUDED.last_updated, data_status.last_updated),
                    last_error = EXCLUDED.last_error
            """),
            {
                "source": source,
                "entity": entity,
                "last_updated": last_updated or datetime.now(timezone.utc),
                "last_error": last_error,
            },
        )


def upsert_ohlcv(symbol: str, rows: List[Dict[str, Any]], source: str = "yfinance") -> int:
    """Insert OHLCV bars; on conflict (symbol, time) update. Returns count inserted/updated."""
    if not rows:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        for r in rows:
            t = r.get("time") or r.get("date")
            if hasattr(t, "isoformat"):
                t = t.isoformat()
            conn.execute(
                text("""
                    INSERT INTO ohlcv (symbol, time, open, high, low, close, volume, adjusted_close, source)
                    VALUES (:symbol, :time, :open, :high, :low, :close, :volume, :adjusted_close, :source)
                    ON CONFLICT (symbol, time) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume,
                        adjusted_close = EXCLUDED.adjusted_close, source = EXCLUDED.source
                """),
                {
                    "symbol": symbol,
                    "time": t,
                    "open": float(r.get("Open", r.get("open", 0))),
                    "high": float(r.get("High", r.get("high", 0))),
                    "low": float(r.get("Low", r.get("low", 0))),
                    "close": float(r.get("Close", r.get("close", 0))),
                    "volume": int(r.get("Volume", r.get("volume", 0)) or 0),
                    "adjusted_close": float(r.get("Adj Close", r.get("adjusted_close")) or r.get("Close", r.get("close"))) if r.get("Adj Close") is not None or r.get("adjusted_close") is not None else None,
                    "source": source,
                },
            )
            count += 1
    return count


def upsert_macro_series(series_id: str, rows: List[Dict[str, Any]]) -> int:
    """Insert macro_series rows. (series_id, time) unique."""
    if not rows:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        for r in rows:
            t = r.get("time") or r.get("date")
            if hasattr(t, "isoformat"):
                t = t.isoformat()
            val = r.get("value")
            if val is None and "value" not in r:
                val = r.get(list(r.keys())[0]) if r else None
            conn.execute(
                text("""
                    INSERT INTO macro_series (series_id, time, value)
                    VALUES (:series_id, :time, :value)
                    ON CONFLICT (series_id, time) DO UPDATE SET value = EXCLUDED.value
                """),
                {"series_id": series_id, "time": t, "value": float(val) if val is not None else None},
            )
            count += 1
    return count


def insert_news(items: List[Dict[str, Any]]) -> int:
    """Insert news rows (no unique constraint; append-only for simplicity)."""
    if not items:
        return 0
    from sqlalchemy import text
    engine = get_engine()
    with engine.begin() as conn:
        for n in items:
            conn.execute(
                text("""
                    INSERT INTO news (symbol, source, title, url, published_at, summary)
                    VALUES (:symbol, :source, :title, :url, :published_at, :summary)
                """),
                {
                    "symbol": n.get("symbol"),
                    "source": n.get("source", ""),
                    "title": n.get("title", "")[:500],
                    "url": n.get("url"),
                    "published_at": n.get("published_at") or n.get("publishedAt"),
                    "summary": n.get("summary", "")[:10000] if n.get("summary") else None,
                },
            )
    return len(items)


def get_data_status() -> List[Dict[str, Any]]:
    """Return all data_status rows for dashboard."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT source, entity, last_updated, last_error FROM data_status ORDER BY source, entity"))
        return [{"source": r[0], "entity": r[1], "last_updated": r[2].isoformat() if r[2] else None, "last_error": r[3]} for r in result]


def upsert_company_profile(symbol: str, data: Dict[str, Any]) -> None:
    """Upsert company_profile row (symbol primary key)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO company_profile (symbol, name, sector, industry, market_cap, description, updated_at)
                VALUES (:symbol, :name, :sector, :industry, :market_cap, :description, :updated_at)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name, sector = EXCLUDED.sector, industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap, description = EXCLUDED.description, updated_at = EXCLUDED.updated_at
            """),
            {
                "symbol": symbol.upper(),
                "name": (data.get("companyName") or data.get("name") or data.get("symbol"))[:255] if data.get("companyName") or data.get("name") or data.get("symbol") else None,
                "sector": (data.get("sector") or "")[:100],
                "industry": (data.get("industry") or "")[:150],
                "market_cap": float(data["marketCap"]) if data.get("marketCap") is not None else None,
                "description": (data.get("description") or "")[:10000] if data.get("description") else None,
                "updated_at": datetime.now(timezone.utc),
            },
        )


def upsert_income_statement(symbol: str, period_end: str, period_type: str, data: Dict[str, Any]) -> None:
    """Upsert one income statement row. period_end YYYY-MM-DD, period_type annual|quarterly."""
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO income_statement (symbol, period_end, period_type, data, updated_at)
                VALUES (:symbol, :period_end, :period_type, :data::jsonb, :updated_at)
                ON CONFLICT (symbol, period_end, period_type) DO UPDATE SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
            """),
            {
                "symbol": symbol.upper(),
                "period_end": period_end,
                "period_type": period_type,
                "data": json.dumps(data) if isinstance(data, dict) else data,
                "updated_at": datetime.now(timezone.utc),
            },
        )


def upsert_balance_sheet(symbol: str, period_end: str, period_type: str, data: Dict[str, Any]) -> None:
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO balance_sheet (symbol, period_end, period_type, data, updated_at)
                VALUES (:symbol, :period_end, :period_type, :data::jsonb, :updated_at)
                ON CONFLICT (symbol, period_end, period_type) DO UPDATE SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
            """),
            {
                "symbol": symbol.upper(),
                "period_end": period_end,
                "period_type": period_type,
                "data": json.dumps(data) if isinstance(data, dict) else data,
                "updated_at": datetime.now(timezone.utc),
            },
        )


def upsert_cash_flow(symbol: str, period_end: str, period_type: str, data: Dict[str, Any]) -> None:
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO cash_flow (symbol, period_end, period_type, data, updated_at)
                VALUES (:symbol, :period_end, :period_type, :data::jsonb, :updated_at)
                ON CONFLICT (symbol, period_end, period_type) DO UPDATE SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
            """),
            {
                "symbol": symbol.upper(),
                "period_end": period_end,
                "period_type": period_type,
                "data": json.dumps(data) if isinstance(data, dict) else data,
                "updated_at": datetime.now(timezone.utc),
            },
        )


# --- Read helpers for API ---

def get_company_profile(symbol: str) -> Optional[Dict[str, Any]]:
    """Return one company_profile row or None."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        r = conn.execute(
            text("SELECT symbol, name, sector, industry, market_cap, description FROM company_profile WHERE symbol = :sym"),
            {"sym": symbol.upper()},
        ).fetchone()
    if not r:
        return None
    return {"symbol": r[0], "name": r[1], "sector": r[2], "industry": r[3], "market_cap": float(r[4]) if r[4] is not None else None, "description": r[5]}


def search_company_profiles(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search company_profile by symbol or name (ILIKE)."""
    from sqlalchemy import text
    engine = get_engine()
    q = f"%{query.strip()}%"
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT symbol, name, sector, industry, market_cap FROM company_profile WHERE symbol ILIKE :q OR name ILIKE :q ORDER BY symbol LIMIT :lim"),
            {"q": q, "lim": limit},
        )
        return [{"symbol": r[0], "name": r[1], "sector": r[2], "industry": r[3], "market_cap": float(r[4]) if r[4] is not None else None} for r in result]


def get_ohlcv_range(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Return OHLCV rows for symbol between start and end (YYYY-MM-DD)."""
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT time, open, high, low, close, volume, adjusted_close FROM ohlcv WHERE symbol = :sym AND time >= :start AND time <= :end ORDER BY time"),
            {"sym": symbol.upper(), "start": start_date, "end": end_date},
        )
        return [
            {
                "time": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": int(r[5]) if r[5] is not None else None,
                "adjusted_close": float(r[6]) if r[6] is not None else None,
            }
            for r in result
        ]


def get_income_statements(symbol: str, period_type: str = "annual", limit: int = 10) -> List[Dict[str, Any]]:
    """Return income statement rows (data as dict)."""
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT period_end, data FROM income_statement WHERE symbol = :sym AND period_type = :pt ORDER BY period_end DESC LIMIT :lim"),
            {"sym": symbol.upper(), "pt": period_type, "lim": limit},
        )
        return [{"period_end": str(r[0]), "data": r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})} for r in result]


def get_balance_sheets(symbol: str, period_type: str = "annual", limit: int = 10) -> List[Dict[str, Any]]:
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT period_end, data FROM balance_sheet WHERE symbol = :sym AND period_type = :pt ORDER BY period_end DESC LIMIT :lim"),
            {"sym": symbol.upper(), "pt": period_type, "lim": limit},
        )
        return [{"period_end": str(r[0]), "data": r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})} for r in result]


def get_cash_flows(symbol: str, period_type: str = "annual", limit: int = 10) -> List[Dict[str, Any]]:
    import json
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT period_end, data FROM cash_flow WHERE symbol = :sym AND period_type = :pt ORDER BY period_end DESC LIMIT :lim"),
            {"sym": symbol.upper(), "pt": period_type, "lim": limit},
        )
        return [{"period_end": str(r[0]), "data": r[1] if isinstance(r[1], dict) else (json.loads(r[1]) if r[1] else {})} for r in result]


def get_macro_latest(series_ids: List[str]) -> Dict[str, Optional[float]]:
    """Return latest value per series_id from macro_series (one row per series, max time)."""
    from sqlalchemy import text
    engine = get_engine()
    out = {}
    with engine.connect() as conn:
        for sid in series_ids:
            r = conn.execute(
                text("SELECT value FROM macro_series WHERE series_id = :sid ORDER BY time DESC LIMIT 1"),
                {"sid": sid},
            ).fetchone()
            out[sid] = float(r[0]) if r and r[0] is not None else None
    return out
