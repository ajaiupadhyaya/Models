# Recovery Execution Plan (Phase 0 + Phase 1)

This is the execution-grade plan to stabilize, understand, and then repair the project without introducing additional architectural drift.

## Outcome Goals

- Restore frontend/API contract reliability for core terminal workflows.
- Establish one canonical path for data access and one for each major domain use case.
- Ensure deployment fails fast on broken schema state.
- Create durable architecture documentation and guardrails so drift does not reappear.

## Workstreams

## WS1 - Contract Recovery (Highest Priority)

### WS1 Objectives

- Eliminate 404s and panel failures caused by unmounted routers and route drift.
- Align terminal commands and module types.

### WS1 Tasks

1. Mount `quant_api`, `equity_api`, and `news_sentiment_api` in `api/main.py` with stable prefixes.
2. Decide canonical route ownership between:
   - `/api/v1/backtest/*` (legacy path), and
   - `/api/v1/quant/backtest` (new path).
3. Add compatibility aliases where needed instead of breaking frontend immediately.
4. Update `parseCommand.ts` `ActiveModule` union to match `TerminalContext.tsx`.
5. Add API smoke tests for:
   - `/api/v1/quant/backtest`
   - `/api/v1/equity/search`
   - `/api/v1/news/{symbol}`

### WS1 Exit Criteria

- No 404s for currently shipped terminal panels in default workflow.

## WS2 - Configuration and Startup Hardening

### WS2 Objectives

- Make startup deterministic and configuration explicit.

### WS2 Tasks

1. Normalize News API env naming:
   - Canonical: `NEWSAPI_KEY`
   - Legacy alias removed during Phase 6 cleanup
2. Update `.env.example` to include required runtime vars (`DATABASE_URL`, `REDIS_URL`, worker settings).
3. Replace `alembic ... || true` with fail-fast logic in startup scripts.
4. Add startup validation checks for required env vars and DB reachability.

### WS2 Exit Criteria

- Misconfigured deployments fail during startup with actionable errors.

## WS3 - Data Layer Consolidation

### WS3 Objectives

- Reduce duplicated fetch logic and unify behavior.

### WS3 Tasks

1. Select canonical data facade for new code (recommended: provider-backed facade).
2. Mark secondary fetch paths as deprecated and prevent new callsites.
3. Create migration map from old pathways (`DataFetcher`, enhanced variants) to canonical facade.
4. Standardize retry, timeout, and error taxonomy across providers.
5. Instrument provider-level metrics and health checks.

### WS3 Exit Criteria

- One primary fetch pathway for each data category (OHLCV, fundamentals, news, macro).

## WS4 - Domain Service Consolidation

### WS4 Objectives

- Remove duplicate implementations for backtest/risk/optimization flows.

### WS4 Tasks

1. Define a single service contract for backtesting.
2. Route both API surfaces through the same service implementation.
3. Consolidate optimization and stress endpoints to service layer contracts.
4. Add deterministic unit tests for each service contract.

### WS4 Exit Criteria

- One service implementation per use case with API-level adapters only.

## WS5 - Database and Migration Integrity

### WS5 Objectives

- Align code expectations, schema migrations, and deployment behavior.

### WS5 Tasks

1. Compare `core/db.py` writes/reads against Alembic revisions.
2. Add missing migrations to cover current runtime schema needs.
3. Add migration test in CI (`upgrade -> smoke -> downgrade` where feasible).
4. Document bootstrap and seed expectations in `db/README.md`.

### WS5 Exit Criteria

- Schema state is reproducible from migration history; no hidden manual drift.

## WS6 - Documentation and Governance

### WS6 Objectives

- Keep architecture understandable and enforceable.

### WS6 Tasks

1. Keep `docs/architecture/current-state.md` updated with every boundary-affecting change.
2. Add API contract source of truth (OpenAPI snapshot + generated endpoint map).
3. Add PR checklist requirements:
   - contract change noted
   - tests updated
   - migration impact checked
   - docs updated
4. Add ownership tags for each subsystem.

### WS6 Exit Criteria

- New work cannot bypass contract and architecture checks.

## Suggested Sequence (Execution Order)

1. WS1 Contract Recovery
2. WS2 Startup Hardening
3. WS5 DB/Migration Integrity
4. WS4 Domain Service Consolidation
5. WS3 Data Layer Consolidation
6. WS6 Governance hardening in parallel after WS1

## First 5 Tickets to Implement Immediately

1. **Mount orphan routers** in `api/main.py` and add route tests.
2. **Normalize News env var** usage and fallback handling.
3. **Command parser type alignment** between `parseCommand.ts` and `TerminalContext.tsx`.
4. **Fail-fast migration startup** in Docker and compose command paths.
5. **Backtest contract decision doc + adapter** to unify `/backtest` and `/quant/backtest`.

## Risk Controls During Execution

- No broad refactors without passing smoke tests for terminal critical path.
- Keep compatibility shims for one release cycle before removing old endpoints.
- Feature-freeze all non-recovery work until WS1+WS2 are complete.

## Definition of Done for Recovery Program

- Terminal critical panels load and execute without endpoint mismatches.
- Startup and migration behavior is deterministic and observable.
- Core use cases map to single service implementations.
- Data access path is consolidated and documented.
- CI enforces contract, tests, and migration checks.
