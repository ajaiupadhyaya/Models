# Current State Architecture Map (May 2026)

This document replaces older inventory assumptions with a code-verified map of what is currently wired, what is partially wired, and what is broken by contract mismatch.

## 1) Runtime Entry Points

- Backend app entry is `backend/main.py`, which re-exports `app` from `api/main.py`.
- Docker and compose both run `uvicorn backend.main:app`.
- `api/main.py` is the actual composition root (middleware, lifespan, router mounting, static frontend serving).

## 2) System Boundaries (As Implemented)

- `api/`: HTTP transport and endpoint modules.
- `core/`: business logic and data access utilities (`*_service.py`, fetchers, provider abstractions, DB helper).
- `core/data_providers/`: provider interfaces and concrete adapters (Polygon, IEX, CoinGecko, FMP, NewsAPI, SEC EDGAR).
- `db/`: Alembic migration scaffolding and a single initial migration.
- `workers/`: Celery app and ingestion tasks.
- `frontend/src/terminal/`: command-driven UI with modular panels mapped to backend domains.

## 3) API Composition Status

### Mounted in `api/main.py`

- `/api/auth` -> `auth_api`
- `/api/v1/models` -> `models_api`
- `/api/v1/predictions` -> `predictions_api`
- `/api/v1/backtest` -> `backtesting_api`
- `/api/v1/ws` -> `websocket_api`
- `/api/v1/monitoring` -> `monitoring`
- `/api/v1/paper-trading` -> `paper_trading_api`
- `/api/v1/reports` -> `investor_reports_api`
- `/api/v1/company` -> `company_analysis_api`
- `/api/v1/ai` -> `ai_analysis_api`
- `/api/v1/data` -> `data_api`
- `/api/v1/data` -> `news_api` (news is mounted under the data prefix)
- `/api/v1/risk` -> `risk_api`
- `/api/v1/automation` -> `automation_api`
- `/api/v1/orchestrator` -> `orchestrator_api`
- `/api/v1/screener` -> `screener_api`
- `/api/v1/comprehensive` -> `comprehensive_api`
- `/api/v1/institutional` -> `institutional_api`

### Present but not mounted

- `api/quant_api.py` (frontend calls `/api/v1/quant/*`)
- `api/equity_api.py` (frontend calls `/api/v1/equity/*`)
- `api/news_sentiment_api.py` (frontend calls `/api/v1/news/{symbol}`)

This creates immediate contract mismatches and likely 404s for core UI panels.

## 4) Frontend-to-Backend Contract Gaps

The terminal frontend uses endpoints that are currently not registered in the backend composition:

- `FundamentalPanel` and `TickerSearchBar` call `/api/v1/equity/*`.
- `QuantPanel` and `BacktestPanel` call `/api/v1/quant/*`.
- `NewsSentimentPanel` calls `/api/v1/news/{symbol}`.

Additionally, command typing was inconsistent:

- `TerminalContext.tsx` supports modules `newsSentiment`, `backtest`, `optimizer`, and `stressTest`.
- `parseCommand.ts` now imports `ActiveModule` directly from `TerminalContext.tsx` to keep one source-of-truth.

Terminal cleanup status:

- Shared endpoint constants now live in `frontend/src/terminal/endpoints.ts`.
- Overlapping Quant backtest execution UI is consolidated by routing users to the dedicated Backtest module.

## 5) Data Access and Service Layer Drift

There are multiple overlapping pathways for the same responsibilities:

- `core/data_fetcher.py` (`DataFetcher`) is widely used.
- `core/data_fetcher_enhanced.py` introduces health/rate-limiting behavior.
- `core/unified_fetcher.py` provides a provider registry/facade but is not the universal path.

Service files (`core/backtest_service.py`, `core/optimizer_service.py`, `core/stress_test_service.py`) follow a DB-first fallback-to-fetcher approach, while legacy routes still use `core/backtesting.py` through `api/backtesting_api.py`.

Result: duplicated behavior, inconsistent error semantics, and difficult observability.

### Consolidation status

- Canonical service-layer fetch path is now `core/market_data_facade.py`.
- `core/backtest_service.py`, `core/optimizer_service.py`, and `core/stress_test_service.py` now use the facade instead of direct `DataFetcher` fallback code.
- Remaining work: migrate API-layer callsites that still instantiate `DataFetcher` directly.

## 6) Async and Persistence State

- Celery is configured in `workers/celery_app.py` with ingestion tasks in `workers/tasks/ingestion.py`.
- API triggers async refresh jobs from `api/data_api.py`.
- DB helpers are in `core/db.py`.
- Migration folder exists (`db/`), but only one revision is present (`001_initial_timescale_schema.py`).
- Startup now validates env + DB connectivity and fails fast on migration errors in Docker/compose paths.
- Migration smoke script is available at `db/migration_smoke_check.py` to validate required tables after upgrade.

## 7) Configuration Inconsistencies

- Canonical key is now `NEWSAPI_KEY`.
- Legacy alias fallback has been removed as part of shim cleanup.

## 8) Main Risks by Severity

1. Unmounted API routers used by frontend (critical user-facing breakage).
2. Dual backtest stacks (`/api/v1/backtest` vs `/api/v1/quant/backtest`) with different implementations.
3. Multiple fetcher/provider pathways producing inconsistent behavior.
4. DB migration coverage is still thin (single revision) despite fail-fast startup.
5. Environment variable drift risk is reduced but legacy alias cleanup remains.
6. Type/API drift in command parser vs terminal module model.

## 9) Target Architecture Boundaries (Recovery Target)

- `api/` handles only HTTP concerns and DTO validation.
- `core/services/` owns use-case orchestration.
- `core/data_providers/` + one canonical fetch facade owns external data acquisition.
- `core/db.py` (or repository layer) owns database reads/writes.
- `workers/` reuses the same service layer as APIs; no duplicate business logic.
- `frontend/` binds only to documented `/api/v1/*` contracts generated from mounted routers.

## 10) Unknowns to Verify via Runtime Smoke Tests

- Whether a separate runtime path mounts `quant_api`, `equity_api`, or `news_sentiment_api`.
- Whether current DB schema in running environments matches `core/db.py` assumptions.
- Which optional dependencies are missing at runtime due to best-effort dependency install patterns.
- Whether all mounted routes enforce the intended auth model consistently.
