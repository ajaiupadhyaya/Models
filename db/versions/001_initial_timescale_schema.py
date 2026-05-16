"""Initial schema: TimescaleDB extension, ohlcv, macro_series, fundamentals, news, data_status.

Revision ID: 001_initial
Revises:
Create Date: Initial migration for Financial Research Platform

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable TimescaleDB extension (requires superuser or extension pre-installed)
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # --- OHLCV (hypertable) ---
    op.create_table(
        "ohlcv",
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(20, 6), nullable=False),
        sa.Column("high", sa.Numeric(20, 6), nullable=False),
        sa.Column("low", sa.Numeric(20, 6), nullable=False),
        sa.Column("close", sa.Numeric(20, 6), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("adjusted_close", sa.Numeric(20, 6), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
    )
    op.create_index("ix_ohlcv_symbol_time", "ohlcv", ["symbol", "time"], unique=True)
    op.execute(
        "SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE, "
        "migrate_data => TRUE, chunk_time_interval => INTERVAL '1 month');"
    )

    # --- Macro series (hypertable) ---
    op.create_table(
        "macro_series",
        sa.Column("series_id", sa.String(50), nullable=False),
        sa.Column("time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("value", sa.Numeric(20, 6), nullable=True),
    )
    op.create_index("ix_macro_series_id_time", "macro_series", ["series_id", "time"], unique=True)
    op.execute(
        "SELECT create_hypertable('macro_series', 'time', if_not_exists => TRUE, "
        "migrate_data => TRUE, chunk_time_interval => INTERVAL '1 month');"
    )

    # --- Company profile ---
    op.create_table(
        "company_profile",
        sa.Column("symbol", sa.String(20), primary_key=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("sector", sa.String(100), nullable=True),
        sa.Column("industry", sa.String(150), nullable=True),
        sa.Column("market_cap", sa.Numeric(20, 2), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
    )

    # --- Income statement (annual/quarterly) ---
    op.create_table(
        "income_statement",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("period_type", sa.String(10), nullable=False),  # annual | quarterly
        sa.Column("data", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_income_statement_symbol_period", "income_statement", ["symbol", "period_end", "period_type"], unique=True)

    # --- Balance sheet ---
    op.create_table(
        "balance_sheet",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("period_type", sa.String(10), nullable=False),
        sa.Column("data", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_balance_sheet_symbol_period", "balance_sheet", ["symbol", "period_end", "period_type"], unique=True)

    # --- Cash flow ---
    op.create_table(
        "cash_flow",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("period_type", sa.String(10), nullable=False),
        sa.Column("data", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_cash_flow_symbol_period", "cash_flow", ["symbol", "period_end", "period_type"], unique=True)

    # --- News ---
    op.create_table(
        "news",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.create_index("ix_news_symbol_published", "news", ["symbol", "published_at"])

    # --- Options chain cache (optional) ---
    op.create_table(
        "options_chain",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("expiry", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(12, 4), nullable=False),
        sa.Column("option_type", sa.String(4), nullable=False),  # call | put
        sa.Column("bid", sa.Numeric(12, 4), nullable=True),
        sa.Column("ask", sa.Numeric(12, 4), nullable=True),
        sa.Column("implied_volatility", sa.Numeric(10, 6), nullable=True),
        sa.Column("delta", sa.Numeric(10, 6), nullable=True),
        sa.Column("gamma", sa.Numeric(10, 6), nullable=True),
        sa.Column("theta", sa.Numeric(12, 6), nullable=True),
        sa.Column("vega", sa.Numeric(10, 6), nullable=True),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_options_chain_symbol_expiry", "options_chain", ["symbol", "expiry", "strike", "option_type"])

    # --- Data status (for dashboard) ---
    op.create_table(
        "data_status",
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("entity", sa.String(100), nullable=False),  # e.g. "ohlcv", "macro", "news"
        sa.Column("last_updated", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("source", "entity"),
    )


def downgrade() -> None:
    op.drop_table("data_status")
    op.drop_table("options_chain")
    op.drop_table("news")
    op.drop_table("cash_flow")
    op.drop_table("balance_sheet")
    op.drop_table("income_statement")
    op.drop_table("company_profile")
    op.drop_table("macro_series")
    op.drop_table("ohlcv")
    op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
