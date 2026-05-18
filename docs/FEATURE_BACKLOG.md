# Models — v1.0 Scope + v2 Roadmap

**Goal:** Ship a tagged, demoable `v1.0.0` that's "finished" — a polished slice users can actually use — then keep the wider vision alive as a documented v2 roadmap.

**Why this exists:** The previous backlog (Feb 2026) scoped ~16 weeks of work across 27 critical stories. For a solo project, that's a treadmill. This document trades breadth for a real stopping point: 4 of 8 terminal modules, 3 of 6 D3 charts, equity-only backtesting, no MLflow, no LLM agent — with everything cut explicitly preserved below as v2.

**Status as of 2026-05-18:** Phase A complete (deploy unblocked, JWT + JSON logs + SECURITY landed, .env.example completed). Phase B is ~95% delivered per audit (4 modules + 11 bonus panels, 3 charts working, command bar with 18 commands, workspace persistence). Phase C in progress (README rewritten, DEPLOYMENT_GUIDE, TROUBLESHOOTING, SECURITY committed). Next: ARCHITECTURE.md audit, screenshot grid, tag `v1.0.0`.

**Related docs:** [`../SECURITY.md`](../SECURITY.md) · [`../DEPLOYMENT_GUIDE.md`](../DEPLOYMENT_GUIDE.md) · [`../TROUBLESHOOTING.md`](../TROUBLESHOOTING.md)

---

## v1.0 — Stopping Point (~3–5 weeks)

### Phase A — Deploy, secure, observable (1 week) ✅ COMPLETE

**A1. Deploy verified end-to-end** ✅
- Fixed Dockerfile (`api.main:app` entrypoint, removed broken alembic + `--require-db-url` flags)
- Three free-tier deploy paths documented in `DEPLOYMENT_GUIDE.md`: Fly+Vercel+Supabase (recommended), Render single-service, Render+Supabase

**A1 (original Render-only criteria)**
- Render service deploys cleanly from `main` (no curl_cffi/import regressions)
- `/health` returns 200, `/docs` loads, `/api/auth/login` issues a token, one data endpoint returns real data using the token
- Document the exact env-var set in `render.yaml` matches `.env.example`
- **Acceptance:** Cold deploy from clean Render account → working terminal in <15 min following `DEPLOYMENT_GUIDE.md`

**A2. JWT protection on expensive routes** ✅
- `Depends(get_current_user)` wired as router-level dep on backtesting, ai, paper_trading, automation, predictions
- Read-only routes (data, quant, news, risk) remain open — documented in `SECURITY.md`

**A2 (original criteria)**
- `get_current_user` already exists in `api/auth_api.py:73` — wire it as a router-level dependency on:
  - `backtesting_api` (cpu-heavy)
  - `ai_analysis_api` (paid OpenAI)
  - `paper_trading_api` (state-mutating)
  - `automation_api` (state-mutating)
  - `predictions_api` (cpu-heavy)
- Read-only data/quant/news/risk routes stay open (or behind same dep — decide once, document)
- **Acceptance:** Unauthenticated POST to `/api/backtest/run` returns 401; with valid token returns 200

**A3. Structured JSON logging + request IDs** ✅
- `api/logging_config.py` ships a JSON formatter driven by `LOG_LEVEL`
- `RequestIDMiddleware` in `api/main.py` mints a UUID per request, attaches to `request_id_ctx` contextvar, returns `X-Request-ID` header
- CORS also tightened in the same pass: env-driven `CORS_ORIGINS`, credentialed mode auto-disabled when set to `*`

**A3 (original criteria)**
- Create `api/logging_config.py` — JSON formatter, level via env (`LOG_LEVEL`)
- Middleware in `api/main.py` that mints a UUID per request, attaches to logging context, returns it as `X-Request-ID` header
- Replace `logging.basicConfig` calls; keep `logger = logging.getLogger(__name__)` pattern unchanged
- **Acceptance:** `curl /health` produces a single JSON log line with `request_id`, `path`, `status`, `latency_ms`

**A4. `.env.example` covers all required vars** ✅
- Grouped into auth / logging / data providers / paper trading / infra / scheduler / frontend build-time
- Includes generator command for `AUTH_SECRET`

**A4 (original criteria)**
- Add `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET`, `LOG_LEVEL`
- Group: API keys, auth, runtime infra, optional features
- Cross-check against `render.yaml` and `config/settings.py`
- **Acceptance:** Fresh clone + `cp .env.example .env` + fill values → `make dev` works

**A5. `SECURITY.md`** ✅
- Threat model, auth, protected vs open routes, secret rotation, transport, rate limiting, v2 deferred items, disclosure email

**A5 (original criteria)**
- How auth works (JWT, single-user MVP, env-based creds)
- Secret rotation (`AUTH_SECRET` change invalidates all tokens)
- What's not in v1 (RBAC, OAuth, secret scanning) — link to v2 roadmap
- Responsible disclosure email
- **Acceptance:** Reviewed and committed; linked from README

---

### Phase A bonus (free-tier hosting unblockers) ✅
- Data providers (Polygon, IEX) degrade gracefully when key missing (no more ValueError)
- DB layer no-ops when `DATABASE_URL` unset; read helpers return `[]` / `None`
- `db/init.sql` ships the schema for Supabase / Neon / Render PG
- Celery+Redis replaced by APScheduler in-process (`api/scheduler.py`, gated by `SCHEDULER_ENABLED`); legacy `workers/` kept for v2
- `fly.toml`, `frontend/vercel.json`, updated `render.yaml` (plan: free)

---

### Phase B — Frontend MVP (2 weeks) ⚠️ ~95% delivered (per 2026-05-18 audit)

Audit found: 4 scope modules + 11 bonus modules already working, all 3 D3 charts (candlestick, correlation matrix, factor heatmap) built, command bar with 18 commands, localStorage workspace persistence. Remaining v1 work is polish (loading/error states, acceptance-criteria smoke tests, AI panel integration in Backtest module).

**Goal:** 4 polished terminal modules, not 8 half-finished ones.

**B1. Module: Primary Instrument (polish existing)**
- Candlestick + volume chart works smoothly for any symbol via command bar `GP AAPL`
- Loading states, error states, timeframe selector (1D/1W/1M/3M/1Y/5Y)
- **Acceptance:** Symbol switching <500ms; no flicker; charts handle 5Y daily without lag

**B2. Module: Quant**
- Two charts: **factor exposure bar chart** + **regime timeline**
- Wire to existing `api/quant_api.py` endpoints
- Show factor exposures (FF3/FF5) for a symbol/portfolio + regime classification
- **Acceptance:** `QUANT AAPL` shows factor breakdown + current regime; data refreshes on symbol change

**B3. Module: Portfolio**
- One chart: **allocation treemap** (D3) + risk metrics table (Sharpe, max DD, vol, VaR)
- Wire to existing `api/risk_api.py` and portfolio endpoints
- **Acceptance:** `PORT AAPL MSFT GOOGL` shows treemap + risk table; weights editable

**B4. Module: Backtest/AI**
- Strategy selector (SMA crossover, momentum, mean-reversion — pick 3 from existing `core/backtesting.py`)
- Equity curve + Sharpe + max DD displayed
- AI panel: ask a question, get a natural-language answer (use existing `api/ai_analysis_api.py`)
- **Acceptance:** `BACKTEST AAPL 50sma.200sma` runs in <10s and shows results; `AI why is AAPL down today?` returns answer

**B5. D3 chart library (3 charts only)**
- Candlestick (exists; polish for shared use)
- **Correlation matrix** (new) — shared by Portfolio + Quant
- **Factor heatmap** (new) — Quant module
- All charts: typed props, loading/error/empty states, zoom/pan where appropriate, no inline data fetching (props only)
- **Acceptance:** Each chart has a storybook-style demo page or test rendering with mock data

**B6. Command bar — 6 commands**
- `GP <SYMBOL>` (primary instrument), `QUANT <SYMBOL>`, `PORT <SYMBOL...>`, `BACKTEST <SYMBOL> <STRATEGY>`, `AI <free text>`, `HELP`
- Symbol completion from recent history (localStorage)
- Unknown command → show HELP
- **Acceptance:** All 6 commands parse correctly; bad input shows useful error, never crashes

**B7. Workspace persistence (basic)**
- Current symbol + active module + recent symbols saved to localStorage on every change
- Restore on page reload
- No per-module settings tracking yet (v2)
- **Acceptance:** Reload browser → land back exactly where you were

---

### Phase C — Demo polish + docs (3–5 days)

**C1. README rewrite**
- Hero section: 1-sentence value prop, demo GIF, screenshot grid (4 modules)
- "Try it" — Render deploy button + local quickstart (3 commands)
- Architecture diagram (link to `ARCHITECTURE.md`)
- v1.0 scope + link to v2 roadmap (this doc)
- Badges: CI, license, last commit

**C2. `DEPLOYMENT_GUIDE.md`**
- Render path (recommended): blueprint, env vars, common failures
- Local Docker path: `docker compose up`, expected output
- Troubleshooting checklist (port conflicts, env-var typos, build cache)

**C3. `TROUBLESHOOTING.md`**
- Top 5 issues from your own commit history (curl_cffi blocks, yfinance imports, cold-start timeouts, missing env vars, CORS)
- For each: symptom → diagnosis → fix

**C4. `ARCHITECTURE.md` audit**
- Update to reflect current code (not Feb baseline)
- Layer diagram: frontend ↔ FastAPI ↔ models/core ↔ data providers
- Module-by-module description

**C5. Tag `v1.0.0`**
- Release notes summarizing what's in v1
- Link to v2 roadmap section below
- GitHub release with screenshots/GIF

---

## v2 Roadmap — Upgrade Paths (post-v1)

Explicitly cut from v1.0 to make the stopping point reachable. Each is a coherent next chunk of work.

### Data & Storage
- **Provider rotation v2** — primary→fallback logic, per-provider rate limits, audit trail in JSONL
- **World Bank + IMF + Reddit + Google Trends** providers
- **10-year backfill** of top-500 symbols into Parquet cold storage (script exists, needs run + validation)
- **Point-in-time dataset snapshots** with hash verification (partial — needs audit)

### Quant & ML
- **Cross-asset backtesting** — options (exercise/assignment), futures (margin/settlement), crypto (24/7)
- **MLflow + model registry** — tracking, promotion dev→staging→prod, auto-rollback on metric degradation
- **Feature store** with versioning
- **RL training pipeline** — PPO on Gymnasium env, 100k+ step training runs
- **Vol surface modeling** (smiles, skew)

### AI / LLM
- **LLM research agent** (Epic 3.3 flagship) — LangChain with tools: fetch_price, fundamentals, backtest, risk, news. This is its own project really.

### Frontend
- **Remaining 4 modules**: Fundamental (waterfall), Technical (indicators-on-candlestick), Economic (macro time-series), News (sentiment timeline), Screening (multi-factor scatter)
- **Remaining 3 charts**: vol surface, heatmap (generic), regime shift
- **Command bar expansion** — 15+ commands, secondary symbols (`PORT AAPL MSFT GOOGL`), parameter parsing (`BACKTEST 50sma.200ema`)
- **Per-module workspace settings** — indicators, timeframes, color scales persist per layout

### Security & Ops
- **OAuth2** (replace single-user JWT)
- **Per-route rate limits + quotas** (current limiter is global per-IP)
- **Prometheus `/metrics`** + Grafana dashboards + alert rules (high latency, error rate, CPU)
- **Security scanning in CI** — Bandit, pip-audit, safety, truffleHog
- **Redis-backed rate limiter** (current is in-memory, single-process only)

### Testing & Docs
- **Playwright E2E** — 5 critical user workflows
- **Coverage gate** — ≥75% overall, ≥85% on critical paths
- **OpenAPI examples** — curl/Python/TypeScript snippets per endpoint
- **User guides** — `WORKFLOWS.md` with 5+ example tasks

---

## Out of scope (forever, or until requirements change)

- Multi-tenant / multi-user (single-user MVP is intentional)
- Live trading (paper trading only; live requires regulatory work)
- Mobile-native apps (web responsive is enough)
- Real-time tick-level data (daily/intraday bars only)

---

## How to use this doc

1. **Working on v1?** Phase A → B → C top to bottom, no skipping.
2. **Tempted to scope-creep?** Check the v2 list first — if it's there, defer.
3. **Found something not in either list?** Add to v2, not v1.
4. **Hit a v1 acceptance criterion?** Check it off here, commit the doc update with the change.

Source-of-truth status lives in this file. Delete `PHASE_1_2_SUMMARY.md`, `CONTEXT_COMPLIANCE_MATRIX.md` after v1 ships if they're not maintained.
