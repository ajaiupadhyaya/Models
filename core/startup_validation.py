"""
Startup validation checks for deployment environments.

Run before migrations/server boot to fail fast on missing critical settings.
"""

from __future__ import annotations

import argparse
import logging
import os
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _validate_url(name: str, raw: str) -> None:
    parsed = urlparse(raw)
    if not parsed.scheme:
        raise ValueError(f"{name} is missing URL scheme")
    if not parsed.netloc and parsed.scheme not in {"sqlite"}:
        raise ValueError(f"{name} is missing host information")


def _check_database_connection(database_url: str) -> None:
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        raise RuntimeError(f"SQLAlchemy not available for DB check: {exc}") from exc

    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    engine.dispose()


def run_validation(require_db_url: bool, require_redis_url: bool, check_db: bool) -> None:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    redis_url = os.environ.get("REDIS_URL", "").strip()

    if require_db_url and not database_url:
        raise ValueError("DATABASE_URL is required but not set")
    if require_redis_url and not redis_url:
        raise ValueError("REDIS_URL is required but not set")

    if database_url:
        _validate_url("DATABASE_URL", database_url)
    if redis_url:
        _validate_url("REDIS_URL", redis_url)

    if check_db and database_url:
        logger.info("Checking database connectivity...")
        _check_database_connection(database_url)
        logger.info("Database connectivity check passed")

    logger.info("Startup validation passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate runtime environment before startup")
    parser.add_argument("--require-db-url", action="store_true", help="Fail if DATABASE_URL is missing")
    parser.add_argument("--require-redis-url", action="store_true", help="Fail if REDIS_URL is missing")
    parser.add_argument("--check-db", action="store_true", help="Verify DB connectivity with SELECT 1")
    args = parser.parse_args()

    try:
        run_validation(
            require_db_url=args.require_db_url,
            require_redis_url=args.require_redis_url,
            check_db=args.check_db,
        )
        return 0
    except Exception as exc:
        logger.error("Startup validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
