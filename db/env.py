"""
Alembic environment. Reads DATABASE_URL from environment.
Run from repo root: alembic -c db/alembic.ini upgrade head
"""
import os
from pathlib import Path

# Add project root to path so config can be loaded if needed
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_project_root))

from logging.config import fileConfig
from sqlalchemy import create_engine
from sqlalchemy import pool
from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DATABASE_URL from env (e.g. postgresql://trader:trading123@localhost:5432/trading_metrics)
database_url = os.environ.get(
    "DATABASE_URL",
    "postgresql://trader:trading123@localhost:5432/trading_metrics",
)
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = None  # We use raw SQL in migrations for TimescaleDB


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
