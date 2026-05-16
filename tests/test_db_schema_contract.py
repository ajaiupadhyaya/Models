"""
Schema contract checks between core DB layer and Alembic migrations.
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _extract_tables_from_core_db(sql_text: str) -> set[str]:
    pattern = r"(?:INSERT INTO|FROM|UPDATE|DELETE FROM)\s+([a-z_]+)"
    return {m.group(1) for m in re.finditer(pattern, sql_text)}


def _extract_tables_from_migrations(migration_text: str) -> set[str]:
    pattern = r'op\.create_table\(\s*"([a-z_]+)"'
    return {m.group(1) for m in re.finditer(pattern, migration_text)}


def test_core_db_tables_are_present_in_migrations():
    core_tables = _extract_tables_from_core_db(_read("core/db.py"))
    migration_tables = _extract_tables_from_migrations(
        _read("db/versions/001_initial_timescale_schema.py")
    )
    missing = sorted(core_tables - migration_tables)
    assert not missing, f"Tables referenced in core/db.py but missing in migrations: {missing}"
