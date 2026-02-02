# Final Development Plan — Professional Quant Terminal

This document outlines the **next and final stages** of development for the financial analysis terminal. The bar is **Bloomberg-level**: a professional, high-level quant analysis personal terminal suitable for daily use by a serious quant / trader / researcher. All stages include **tests** and adherence to the **highest standards** defined in `context.md` and `ARCHITECTURE.md`.

---

## Quality bar (non-negotiable)

- **Institutional-grade**: No demo or toy assumptions; every component must be correct, config-driven, and testable.
- **Quant rigor**: Backtesting, risk, and analytics must use proper assumptions (slippage, costs, point-in-time data goals); no magic numbers.
- **Strong typing**: TypeScript strict mode; Python type hints on public APIs and critical paths.
- **Testing**: Unit tests for config, APIs, and core quant logic; integration tests for pipelines; optional E2E for critical workflows; coverage targets on critical paths.
- **Documentation**: API (OpenAPI), architecture, workflows, and runbooks sufficient for a new developer or operator.
- **Observability**: Structured logging, health checks, and optional metrics (Prometheus) for production readiness.

---

## Current state (baseline)

- **Done**: Command bar (Bloomberg-like codes), tabbed modules (Primary, Fundamental, Technical, Quant, Economic, News, Portfolio, Screening), D3 candlestick + volume + indicators (SMA, RSI, MACD, Bollinger), backtesting API (institutional engine), risk API (VaR/CVaR/volatility/drawdown/Sharpe), centralized config, error/retry handling in panels, persistent workspaces, keyboard shortcuts, ARCHITECTURE.md, WORKFLOWS.md.
- **Tests**: `tests/test_config.py`, `tests/test_risk_api.py`; root-level `test_integration.py`, `test_core_imports.py` (manual-style).
- **Gaps**: Broader test coverage, integration tests as pytest, E2E for terminal flows, quant validation tests, API auth/rate limiting, CI/CD, security checklist, and final UI/performance polish.

---

## Stage 1: Testing & quality assurance

**Goal:** Establish a robust test pyramid and automated quality checks so every change can be validated against the professional bar.

### 1.1 Backend unit tests (pytest)

| Area | Scope | Location | Acceptance |
|------|--------|----------|------------|
| **Config** | Already present; add edge cases for `_env_bool`, `_env_float`, and env overrides | `tests/test_config.py` | All tests pass; no magic numbers in defaults |
| **Risk API** | Already present; add test for “no Close column” and malformed response | `tests/test_risk_api.py` | 200/404/400 cases and numeric types covered |
| **Backtesting API** | Sample-data endpoint: valid symbol returns candles; invalid returns 4xx | `tests/test_backtesting_api.py` (new) | Mock DataFetcher/yfinance; assert schema of candles |
| **Data API** | Macro endpoint: mock FRED; assert series shape and keys | `tests/test_data_api.py` (new) | 200 with `series` array; 4xx when no key or upstream error |
| **Company analysis API** | One endpoint: mock company analyzer; assert ratios/valuation/risk keys | `tests/test_company_api.py` (new) | 200 response shape; 404 for unknown ticker |
| **Core backtesting** | `BacktestEngine` and `InstitutionalBacktestEngine`: fixed price/signal series; assert PnL, drawdown, trade count | `tests/test_core_backtesting.py` (new) | Deterministic inputs → deterministic metrics; institutional engine produces different (worse) equity under costs |

### 1.2 Backend integration tests (pytest)

| Test | Purpose | Acceptance |
|------|--------|------------|
| **Data pipeline** | DataFetcher (or mock) → sample-data path → backtest run | End-to-end request/response; no uncaught exceptions |
| **Risk pipeline** | DataFetcher (mock) → risk metrics endpoint | Valid VaR/CVaR/volatility/drawdown/Sharpe in response |
| **Health and info** | `GET /health`, `GET /info` | 200; `info` lists routers and capabilities |

Prefer pytest fixtures and `TestClient`; mock external APIs (yfinance, FRED, Alpha Vantage) so tests run without keys.

### 1.3 Frontend unit tests (Vitest or Jest)

| Area | Scope | Acceptance |
|------|--------|------------|
| **useFetchWithRetry** | Retry on 5xx, no retry on 4xx, parse null = error, retry() refetches | Mock `fetch`; assert call counts and state |
| **Command bar** | Parse function codes (GP, FA, ECO, etc.) and symbol; emit correct module/symbol | Input strings → expected module + symbol |
| **TerminalContext** | activeModule, primarySymbol, setActiveModule, setPrimarySymbol | State updates and persistence (workspace) if testable in isolation |

Run with existing Vite/TS setup; no UI dependency if possible for hooks/parsers.

### 1.4 E2E (optional but recommended)

- **Playwright or Cypress**: One critical path, e.g. “Load terminal → type AAPL → FA → see Fundamental panel with ratios”.
- **Goal**: Catch regressions in command bar → module → API → render. Run in CI on a built frontend + mocked or stub API if needed.

### 1.5 Linting and typing

- **Backend**: `ruff` or `flake8` + `mypy` on `config/`, `api/`, `core/` (strict on new code).
- **Frontend**: ESLint + TypeScript strict; no `any` in new code.
- **CI**: Lint and type checks must pass before merge.

### 1.6 Coverage

- **Target**: ≥ 80% line coverage for `config/`, `api/risk_api.py`, `api/backtesting_api.py`, `core/backtesting.py`, `core/institutional_backtesting.py`. Lower bound for rest of `api/` and `core/` (e.g. 60%) with critical paths covered.
- **Report**: `pytest --cov=config --cov=api --cov=core --cov-report=html`; publish or archive in CI.

---

## Stage 2: Quant & data rigor

**Goal:** Ensure the quant engine and data layer meet institutional standards and are fully testable and documented.

### 2.1 Backtest validation

- **Tests**: Known price series + known signals → expected cumulative return, max drawdown, number of trades. Compare standard vs institutional engine (institutional should show impact of costs/slippage).
- **Docs**: Short “Backtest methodology” in `ARCHITECTURE.md` or `docs/`: assumptions, slippage/cost model, and how to reproduce results.

### 2.2 Risk model tests

- **Unit tests** for `VaRModel` and `CVaRModel` (`models/risk/var_cvar.py`): fixed return series → known percentile and tail expectation (e.g. historical VaR 95% = 5th percentile).
- **Integration**: Risk API returns consistent VaR/CVaR for same ticker/period (already partially covered in `tests/test_risk_api.py`).

### 2.3 Data layer

- **Validation**: Document and, where feasible, test: required columns for backtest (OHLCV), required columns for risk (e.g. Close), and handling of missing/invalid data (404 or 400 with clear message).
- **Point-in-time / survivorship**: Keep as documented goals in ARCHITECTURE; no “toy” data assumptions in code (e.g. no hardcoded lookahead).

### 2.4 Config and env

- **No magic numbers**: All backtest defaults, risk parameters, and data source choices come from `config/settings.py` or env. Assert in tests or code review.

---

## Stage 3: API & platform hardening

**Goal:** Production-ready API surface: documented, observable, and protected.

### 3.1 OpenAPI and documentation

- **OpenAPI**: Ensure all terminal-used endpoints are documented (summary, parameters, response schema, errors). Tag by domain (Risk, Backtest, Data, Company, AI, etc.).
- **API_DOCUMENTATION.md**: Update with any new endpoints; link to `/docs` and `/redoc` for interactive docs.

### 3.2 Health and readiness

- **Health**: Already present; extend if needed (e.g. DB/cache connectivity when added).
- **Readiness**: Optional `/ready` that checks critical dependencies (data source connectivity optional; avoid blocking on external API keys).

### 3.3 Rate limiting and auth (production path)

- **Rate limiting**: Per-IP or per-key limits on expensive endpoints (backtest, AI, company analysis) to avoid abuse. Document in API docs.
- **Auth**: Plan for API keys or OAuth for programmatic access; document in FINAL_DEVELOPMENT_PLAN or security doc. Implementation can be phased (e.g. API key middleware).

### 3.4 Logging and observability

- **Structured logging**: JSON or key-value logs for request id, endpoint, status, duration; errors with stack traces in non-production.
- **Metrics**: Optional Prometheus endpoint for request counts, latency percentiles, and error rates by route. Align with existing `monitoring/` if used.

---

## Stage 4: Terminal & UI polish

**Goal:** Bloomberg-level UX: keyboard-first, dense information, and performant.

### 4.1 Keyboard and accessibility

- **Command bar**: Already focusable; ensure focus management (no trap), Escape clears and blurs, Enter submits. Document in WORKFLOWS.md.
- **Module cycling**: Alt+Left/Right; ensure visible focus and no skip of panels.
- **Shortcuts**: One place of truth (e.g. HELP overlay + ARCHITECTURE.md + WORKFLOWS.md); no conflicting shortcuts.

### 4.2 Charts (D3)

- **Performance**: No unnecessary re-renders on symbol/timeframe change; throttle/resize handling. Test with 1–2 years of daily data.
- **Consistency**: All new charts use D3; same axis/tooltip/zoom conventions as Primary Instrument.
- **Export**: Optional “Export chart as PNG/SVG” for Primary Instrument (document in WORKFLOWS).

### 4.3 Data tables and formatting

- **Numbers**: Monospace, aligned decimals; consistent precision (e.g. 2 for %, 2–4 for ratios). Use existing `num-mono` and shared formatting.
- **Conditional formatting**: Gains/losses color-coded (already in places); apply consistently in Portfolio, Screening, and any new tables.

### 4.4 Performance budgets

- **Frontend**: Target first contentful paint and time-to-interactive; avoid large bundle chunks for above-the-fold. Lazy-load heavy panels if needed.
- **API**: Target p95 latency for critical endpoints (e.g. sample-data, risk metrics) under 2s with cold cache; document with “expected latency” in API docs.

---

## Stage 5: Production readiness

**Goal:** Deployable, secure, and maintainable.

### 5.1 Docker and runbooks

- **Docker**: `Dockerfile` and `docker-compose.yml` build and run API + frontend (or static build). Document in DOCKER.md or LAUNCH_GUIDE.md.
- **Runbooks**: How to start API and frontend locally; how to run tests (unit, integration); how to run E2E if present. Env vars listed in `.env.example` and docs.

### 5.2 CI/CD

- **CI pipeline**: On push/PR: install deps, lint, type-check, run backend unit + integration tests, run frontend unit tests, (optional) E2E. Fail on failure.
- **Coverage**: Upload or report coverage; optionally fail if coverage drops below threshold on critical paths.

### 5.3 Security checklist

- **Secrets**: No API keys in repo; use env and `.env.example` as template. Document in README or security doc.
- **Dependencies**: Periodic `pip audit` / `npm audit`; fix high/critical. Document in plan or README.
- **CORS and headers**: CORS configured for known origins; security headers (e.g. X-Content-Type-Options) where applicable.

### 5.4 Documentation matrix

| Doc | Purpose |
|-----|--------|
| **README.md** | Quick start, env, how to run tests and dev servers |
| **ARCHITECTURE.md** | High-level design, tech stack, data/quant/UI, example workflows |
| **WORKFLOWS.md** | Step-by-step user workflows (already present) |
| **API_DOCUMENTATION.md** | Endpoint list and examples; link to OpenAPI |
| **FINAL_DEVELOPMENT_PLAN.md** | This plan: stages, tests, adherence |
| **.env.example** | All required/optional env vars with placeholders |

---

## Adherence checklist (context.md deliverables)

Use this to verify alignment with the “non-negotiable” and deliverables in `context.md`:

- [ ] **Architecture**: Modular, API-first, clear separation (data / analytics / modeling / AI / UI). Documented in ARCHITECTURE.md.
- [ ] **Data**: Unified layer (DataFetcher); support equities + macro; config-driven keys; goals for point-in-time and survivorship documented.
- [ ] **Quant engine**: Backtesting with slippage/costs/impact; risk (VaR, CVaR, vol, drawdown, Sharpe); no magic numbers; config-driven.
- [ ] **AI/ML**: Pipelines and APIs available; terminal consumes them (e.g. market summary, predictions). LLM/sentiment as first-class where implemented.
- [ ] **Visualization**: D3-only for charts; candlestick, volume, indicators; interactive and performant.
- [ ] **UI/UX**: Bloomberg-inspired; command bar; multi-panel; keyboard-first; workspaces; shortcuts documented.
- [ ] **Engineering**: Strong typing (TS + Python); config-driven; logging; tests for core logic; clean structure; no toy abstractions.
- [ ] **Testing**: Unit (config, APIs, core backtest/risk); integration (pipelines); optional E2E; coverage targets on critical paths.
- [ ] **Documentation**: Architecture, workflows, API, runbooks, env example.

---

## Success criteria (final stage)

- **Functional**: All terminal workflows in WORKFLOWS.md work against a running API (with appropriate env keys).
- **Tests**: Backend unit + integration tests pass in CI; frontend unit tests pass; optional E2E passes for at least one critical path.
- **Quality**: Lint and type checks pass; no high/critical security advisories on main dependencies.
- **Documentation**: New developer can start API + frontend, run tests, and perform “single-asset deep dive” and “backtest” from README + WORKFLOWS.
- **Bar**: The system is suitable for daily professional use as a personal quant terminal: correct, fast enough, and maintainable.

---

## Suggested order of execution

1. **Stage 1** (Testing & QA): Expand backend unit tests, add integration tests, add frontend unit tests, set up lint/type/coverage and CI. This stabilizes the bar for all later work.
2. **Stage 2** (Quant & data): Backtest and risk validation tests, data validation docs, config audit.
3. **Stage 3** (API hardening): OpenAPI polish, rate limiting, logging/metrics.
4. **Stage 4** (Terminal polish): Keyboard a11y, chart performance, tables and formatting.
5. **Stage 5** (Production): Docker/runbooks, CI/CD, security checklist, documentation pass.

Each stage should be merged only when its acceptance criteria and the adherence checklist items it touches are satisfied. This plan is the **final stages** roadmap to bring the terminal to professional, high-level quant analysis standard with tests and highest-quality engineering throughout.
