-- Models — schema bootstrap for managed Postgres (Supabase / Neon / Render PG).
-- Run once in the SQL editor of your chosen provider.
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS data_status (
    source        TEXT NOT NULL,
    entity        TEXT NOT NULL,
    last_updated  TIMESTAMPTZ,
    last_error    TEXT,
    PRIMARY KEY (source, entity)
);

CREATE TABLE IF NOT EXISTS ohlcv (
    symbol          TEXT        NOT NULL,
    time            TIMESTAMPTZ NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          BIGINT,
    adjusted_close  DOUBLE PRECISION,
    source          TEXT,
    PRIMARY KEY (symbol, time)
);
CREATE INDEX IF NOT EXISTS ohlcv_symbol_time_idx ON ohlcv (symbol, time DESC);

CREATE TABLE IF NOT EXISTS macro_series (
    series_id  TEXT        NOT NULL,
    time       TIMESTAMPTZ NOT NULL,
    value      DOUBLE PRECISION,
    PRIMARY KEY (series_id, time)
);

CREATE TABLE IF NOT EXISTS news (
    id            BIGSERIAL PRIMARY KEY,
    symbol        TEXT,
    source        TEXT,
    title         TEXT,
    url           TEXT,
    published_at  TIMESTAMPTZ,
    summary       TEXT
);
CREATE INDEX IF NOT EXISTS news_symbol_published_idx ON news (symbol, published_at DESC);

CREATE TABLE IF NOT EXISTS company_profile (
    symbol       TEXT PRIMARY KEY,
    name         TEXT,
    sector       TEXT,
    industry     TEXT,
    market_cap   DOUBLE PRECISION,
    description  TEXT,
    updated_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS income_statement (
    symbol       TEXT NOT NULL,
    period_end   DATE NOT NULL,
    period_type  TEXT NOT NULL,
    data         JSONB,
    updated_at   TIMESTAMPTZ,
    PRIMARY KEY (symbol, period_end, period_type)
);

CREATE TABLE IF NOT EXISTS balance_sheet (
    symbol       TEXT NOT NULL,
    period_end   DATE NOT NULL,
    period_type  TEXT NOT NULL,
    data         JSONB,
    updated_at   TIMESTAMPTZ,
    PRIMARY KEY (symbol, period_end, period_type)
);

CREATE TABLE IF NOT EXISTS cash_flow (
    symbol       TEXT NOT NULL,
    period_end   DATE NOT NULL,
    period_type  TEXT NOT NULL,
    data         JSONB,
    updated_at   TIMESTAMPTZ,
    PRIMARY KEY (symbol, period_end, period_type)
);
