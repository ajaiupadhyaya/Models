#!/usr/bin/env python3
"""
Migration smoke check:
1) Run Alembic upgrade head
2) Verify required tables exist

Usage:
  python -m db.migration_smoke_check
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REQUIRED_TABLES = {
    "ohlcv",
    "macro_series",
    "company_profile",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "news",
    "data_status",
}


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is required for migration smoke check", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    alembic_ini = repo_root / "db" / "alembic.ini"

    try:
        _run(["alembic", "-c", str(alembic_ini), "upgrade", "head"])

        from sqlalchemy import create_engine, inspect

        engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise RuntimeError(f"Missing required tables after migration: {missing}")
    except Exception as exc:
        print(f"Migration smoke check failed: {exc}", file=sys.stderr)
        return 1

    print("Migration smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
