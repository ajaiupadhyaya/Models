# Feature Backlog - Production Completion

**Generated:** February 9, 2026  
**Roadmap:** See PRODUCTION_ROADMAP.md  
**Status:** Phase 0 (Gap Lock)

---

## Backlog Organization

Epics and stories are organized by:
1. **Priority:** CRITICAL → HIGH → MEDIUM
2. **Phase:** 0-6 (as per PRODUCTION_ROADMAP.md)
3. **Effort:** S (small, 1-3 days), M (medium, 3-7 days), L (large, 1-2 weeks), XL (extra large, 2+ weeks)
4. **Status:** not-started, in-progress, blocked, ready-for-review, done

---

## Epic 1: DATA FOUNDATION

### CRITICAL Priority

**Story 1.1: Multi-Provider Connectors (Phase 1)**
- **Effort:** XL (2-3 weeks)
- **Status:** not-started
- **Description:** Implement data providers for equities, options, futures, FX, crypto, fundamentals, news.
- **Tasks:**
  - [ ] Create `core/data_providers/base.py` abstract class
  - [ ] Implement `core/data_providers/polygon_provider.py` (Polygon.io)
  - [ ] Implement `core/data_providers/iex_provider.py` (IEX Cloud)
  - [ ] Implement `core/data_providers/oanda_provider.py` (OANDA FX)
  - [ ] Implement `core/data_providers/coingecko_provider.py` (CoinGecko)
  - [ ] Implement `core/data_providers/sec_edgar.py` (SEC fundamentals)
  - [ ] Implement `core/data_providers/newsapi_provider.py` (News feed)
  - [ ] Unit tests for each provider (edge cases, rate limiting, fallback)
  - [ ] API key management (env vars, secrets)
- **Acceptance:** Each provider fetches OHLCV with correct schema; tests pass 100%
- **Owner:** You
- **Blocks:** 1.2, 1.3

**Story 1.2: Unified Data Fetcher V2 (Phase 1)**
- **Effort:** L (1 week)
- **Status:** not-started
- **Description:** Refactor `core/data_fetcher.py` to route to best provider; add caching, fallback, audit trail.
- **Tasks:**
  - [ ] Update DataFetcher to use provider registry
  - [ ] Implement provider selection logic (primary → fallback)
  - [ ] Add caching layer (hot/cold TTLs)
  - [ ] Add audit logging for all fetches
  - [ ] Rate limiting per provider
  - [ ] Integration tests (multiple symbols, fallback)
- **Acceptance:** Fetch any asset type; cache working; 1000 requests < 5s
- **Owner:** You
- **Depends on:** 1.1
- **Blocks:** 1.3, 1.4

**Story 1.3: Point-in-Time Dataset Layer (Phase 1)**
- **Effort:** L (1 week)
- **Status:** not-started
- **Description:** Implement DatasetSnapshot + DatasetManager for reproducibility.
- **Tasks:**
  - [ ] Create `core/datasets.py` with DatasetSnapshot dataclass
  - [ ] Implement save_snapshot (parquet + JSON metadata)
  - [ ] Implement load_snapshot (reconstruct historical exact data)
  - [ ] Hash-based verification (sha256 of CSV for reproducibility)
  - [ ] Tests: save → load → verify identical OHLCV
- **Acceptance:** Snapshot ID → exact same data; hash verification passes
- **Owner:** You
- **Depends on:** 1.2
- **Blocks:** 1.5

**Story 1.4: Cold Storage + Audit Trail (Phase 1)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Archive historical data in cold storage (SQLite/Parquet); track all fetches.
- **Tasks:**
  - [ ] Design cold storage schema (symbol/year/month partitioning)
  - [ ] Implement ColdStorageManager (write parquet, metadata JSON)
  - [ ] Implement retrieve (reconstruct from multiple files)
  - [ ] Audit logging: all fetches logged to JSONL
  - [ ] Tests: archive → query → verify integrity
- **Acceptance:** 10 years of data for 500 symbols stored & queryable; no gaps > 1 day
- **Owner:** You
- **Depends on:** 1.3
- **Blocks:** 1.5

**Story 1.5: 10-Year Backfill Script (Phase 1)**
- **Effort:** M (3-5 days)
- **Status:** not-started
- **Description:** Batch download script for top-500 symbols + macro; run once/quarter.
- **Tasks:**
  - [ ] Create `scripts/backfill_historical_data.py`
  - [ ] Fetch top-500 by market cap
  - [ ] For each: download 2015-01-01 to 2025-01-01 via UnifiedDataFetcher
  - [ ] Store in cold storage
  - [ ] Log progress + errors
  - [ ] Tests: 10 symbols backfill successfully
- **Acceptance:** 500 symbols archived; backfill completes in < 1 hour
- **Owner:** You
- **Depends on:** 1.2, 1.4

### HIGH Priority

**Story 1.6: World Bank + IMF Data Integration (Phase 1)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Add macro data from World Bank and IMF to complement FRED.
- **Tasks:**
  - [ ] Implement `core/data_providers/world_bank_provider.py`
  - [ ] Implement `core/data_providers/imf_provider.py`
  - [ ] Map common indicators (world GDP, EM growth, inflation)
  - [ ] Tests: data fetches correctly aligned with FRED timeline
- **Acceptance:** World Bank + IMF data available alongside FRED; aligned by date
- **Owner:** You
- **Depends on:** 1.1, 1.2

**Story 1.7: Alternative Data + Sentiment (Phase 1)**
- **Effort:** L (1+ weeks)
- **Status:** not-started
- **Description:** Integrate Reddit sentiment, Google Trends, social media feeds.
- **Tasks:**
  - [ ] Implement `core/data_providers/reddit_sentiment.py`
  - [ ] Implement `core/data_providers/google_trends.py`
  - [ ] Aggregate sentiment scores (positive/negative/neutral)
  - [ ] Tests: sentiment scores 0-1 range; trends align with events
- **Acceptance:** Sentiment data fetched; scores normalized 0-1
- **Owner:** You

---

## Epic 2: QUANT ENGINE

### CRITICAL Priority

**Story 2.1: Factor Modeling Framework (Phase 2)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** Implement factor model base class + Fama-French 3/5 factors.
- **Tasks:**
  - [ ] Create `models/factor_models.py`
  - [ ] Define FactorModel base class + methods
  - [ ] Implement FamaFrenchModel (download FF data)
  - [ ] compute_factor_returns() via regression
  - [ ] analyze_factor_exposure() for portfolios
  - [ ] Tests: factor exposures match benchmarks
- **Acceptance:** Factor regression R² > 0.7 for SPY; custom portfolio exposures computed
- **Owner:** You
- **Blocks:** 2.3, 2.4

**Story 2.2: Regime Detection (Phase 2)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** HMM-based market regimes + regime-specific performance analysis.
- **Tasks:**
  - [ ] Create `models/regime_detection.py`
  - [ ] Implement RegimeDetector (HMM, volatility-based)
  - [ ] Implement RegimeAnalyzer (Sharpe/drawdown per regime)
  - [ ] Tests: 3-regime HMM identifies bull/bear/sideways
- **Acceptance:** Regimes detected; performance by regime calculated; Sharpe varies by regime
- **Owner:** You
- **Blocks:** 2.4

**Story 2.3: Options Pricing + Greeks (Phase 2)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Black-Scholes + Greeks; volatility surface modeling.
- **Tasks:**
  - [ ] Enhance/create `models/options_pricing.py`
  - [ ] Implement Black-Scholes call/put + Greeks (delta, gamma, theta, vega, rho)
  - [ ] Volatility surface modeling (smiles, skew)
  - [ ] Tests: compare against Haug 2007, known benchmarks
- **Acceptance:** Greeks match Black-Scholes reference implementations; tests pass
- **Owner:** You
- **Blocks:** 2.5

**Story 2.4: Risk Models: VaR, CVaR, Stress, Scenario (Phase 2)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** Enhanced risk analytics: stress testing, scenario analysis, correlation regimes.
- **Tasks:**
  - [ ] Enhance `models/risk/var_cvar.py` (already exists; add more scenarios)
  - [ ] Create `models/risk/stress_testing.py`
  - [ ] Implement shock_scenario() for market shocks
  - [ ] Implement correlation_regime_shift() for risk-off
  - [ ] Tests: stress scenarios produce predicted losses
- **Acceptance:** Stress test results match Monte Carlo baseline; scenario losses deterministic
- **Owner:** You
- **Blocks:** API integration

**Story 2.5: Backtesting: Cross-Asset Support (Phase 2)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** Extend backtesting to equities, options, futures, crypto.
- **Tasks:**
  - [ ] Enhance `core/backtesting.py` (already supports equity; extend)
  - [ ] Add option contract handling (exercise, assignment)
  - [ ] Add futures handling (margin, daily settlement)
  - [ ] Add crypto handling (24/7 trading, no settlement)
  - [ ] Tests: Options backtest; futures margin; crypto overnight
- **Acceptance:** Cross-asset backtest runs; PnL accounts for asset-specific mechanics
- **Owner:** You
- **Depends on:** 2.3, 2.4

---

## Epic 3: AI/ML/RL PIPELINE

### CRITICAL Priority

**Story 3.1: MLflow Integration + Model Registry (Phase 3)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** Set up MLflow tracking; model versioning; promotion workflow.
- **Tasks:**
  - [ ] Create `ml/mlflow_setup.py` (initialize tracking, artifact storage)
  - [ ] Create `ml/model_registry.py` (promotion, rollback)
  - [ ] Wire model training to MLflow logging (metrics, params, artifacts)
  - [ ] Implement auto-rollback on metric degradation
  - [ ] Tests: model logged → fetched → served
- **Acceptance:** Models logged with metrics; promoted dev → staging → production; rollback works
- **Owner:** You
- **Blocks:** 3.2, 3.3

**Story 3.2: Feature Store (Phase 3)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Centralized feature engineering with versioning.
- **Tasks:**
  - [ ] Create `ml/feature_store.py`
  - [ ] Implement compute_features() (SMA, RSI, volatility, volume, etc.)
  - [ ] Implement get_features() (from store or compute)
  - [ ] Versioning: track feature definitions
  - [ ] Tests: features reproducible; aligned with train/test dates
- **Acceptance:** 50+ technical features computed; versioned; E2E reproducibility
- **Owner:** You
- **Depends on:** 3.1

**Story 3.3: LLM-Powered Research Agent (Phase 3)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** LangChain agent with 5+ tools; query interface.
- **Tasks:**
  - [ ] Create `ai/llm_agent.py`
  - [ ] Define tools: fetch_price, fetch_fundamentals, backtest, analyze_risk, fetch_news
  - [ ] Initialize agent_executor (zero-shot-react)
  - [ ] Hook to OpenAI GPT-4
  - [ ] Tests: agent runs queries; calls correct tools
- **Acceptance:** Query "should I buy AAPL?" → agent fetches data, backtests, returns analysis
- **Owner:** You
- **Depends on:** 1.2 (data), 2.4 (risk)

**Story 3.4: RL Training: PPO Agent (Phase 3)**
- **Effort:** L (1+ weeks)
- **Status:** not-started
- **Description:** Gymnasium environment + Stable Baselines3 PPO.
- **Tasks:**
  - [ ] Create `models/rl/trading_env.py` (Gymnasium environment)
  - [ ] Create `models/rl/ppo_trainer.py` (train PPO on OHLCV)
  - [ ] Implement reward function (trading P&L)
  - [ ] Tests: agent improves over episodes; policy converges
- **Acceptance:** Agent trains for 100k steps; equity curve improves; policy saved
- **Owner:** You
- **Depends on:** 1.2 (data), 3.1 (MLflow)

---

## Epic 4: TERMINAL UI + D3

### CRITICAL Priority

**Story 4.1: D3 Chart Library (Phase 4)**
- **Effort:** L (2 weeks)
- **Status:** not-started
- **Description:** Implement 6 D3 chart types (candlestick, heatmap, vol surface, correlation, factor, regime).
- **Tasks:**
  - [ ] Create `frontend/src/components/charts/VolatilitySurfaceChart.tsx`
  - [ ] Create `frontend/src/components/charts/HeatmapChart.tsx`
  - [ ] Create `frontend/src/components/charts/CorrelationMatrix.tsx`
  - [ ] Create `frontend/src/components/charts/FactorExposureChart.tsx`
  - [ ] Create `frontend/src/components/charts/RegimeShiftChart.tsx`
  - [ ] Polish existing CandlestickChart
  - [ ] Tests: charts render; data binds; zoom/pan works
- **Acceptance:** All 6 charts render smoothly; no lag with 10 years data
- **Owner:** You
- **Blocks:** 4.2

**Story 4.2: Terminal Panels Integration (Phase 4)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** Wire D3 charts into 8 terminal modules.
- **Tasks:**
  - [ ] Update Primary Instrument module (candlestick + volume) ✅ exists; polish
  - [ ] Update Fundamental module (add waterfall chart via D3)
  - [ ] Update Technical module (indicators on candlestick)
  - [ ] Update Quant module (factor heatmap, regime timeline)
  - [ ] Update Economic module (macro time-series via D3)
  - [ ] Update News module (sentiment timeline)
  - [ ] Update Portfolio module (pie/treemap allocation)
  - [ ] Update Screening module (scatter plot multi-factor)
- **Acceptance:** All 8 modules render D3 charts; no crashes; smooth transitions
- **Owner:** You
- **Depends on:** 4.1

**Story 4.3: Command Bar Enhancements (Phase 4)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Support 15+ commands; parse parameters; route to modules.
- **Tasks:**
  - [ ] Enhance `frontend/src/components/CommandBar.tsx`
  - [ ] Support: GP, FA, FLDS, QUANT, ECO, NEWS, PORT, SCREEN, BACKTEST, AI, WORKSPACE, ?
  - [ ] Parse secondary symbols (e.g., PORT AAPL MSFT GOOGL)
  - [ ] Parse parameters (e.g., BACKTEST 50SMA.200EMA)
  - [ ] Tests: parse various command formats; emit correct module + context
- **Acceptance:** All 15+ commands parse correctly; module switches; parameters passed
- **Owner:** You

**Story 4.4: Workspace Persistence (Phase 4)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Save/load terminal layouts + settings per workspace.
- **Tasks:**
  - [ ] Enhance `frontend/src/context/WorkspaceContext.tsx`
  - [ ] Implement saveWorkspace (state → localStorage)
  - [ ] Implement loadWorkspace (localStorage → state)
  - [ ] Track module-specific settings (indicators, timeframes, etc.)
  - [ ] Tests: save workspace → reload browser → restore state
- **Acceptance:** Workspaces persist; user-created layouts restored
- **Owner:** You
- **Depends on:** 4.3

---

## Epic 5: SECURITY & HARDENING

### CRITICAL Priority

**Story 5.1: OAuth2 + JWT Authentication (Phase 5)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** JWT tokens; protected endpoints; role-based access.
- **Tasks:**
  - [ ] Create `api/auth.py` (TokenManager, login endpoint)
  - [ ] Implement JWT creation + verification
  - [ ] Add `get_current_user` dependency (verify token)
  - [ ] Protect critical endpoints (backtest, AI, admin)
  - [ ] Tests: login → token; protected endpoint without token → 401
- **Acceptance:** Login works; token-protected endpoints reject unauthenticated; token expires
- **Owner:** You
- **Blocks:** 5.2, 5.3

**Story 5.2: Rate Limiting + Quota (Phase 5)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Per-endpoint rate limits; track usage; enforce quotas.
- **Tasks:**
  - [ ] Enhance `api/rate_limit.py` (configure limits per route)
  - [ ] Apply to expensive endpoints (backtest 10/hour, AI 20/hour)
  - [ ] Tests: exceed rate limit → 429 Too Many Requests
- **Acceptance:** Rate limiting enforced; users see limits in error response
- **Owner:** You
- **Depends on:** 5.1

**Story 5.3: Structured Logging + Tracing (Phase 5)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** JSON structured logs; request ID tracing; observability.
- **Tasks:**
  - [ ] Create `api/logging_config.py` (StructuredLogger with JSON output)
  - [ ] Create `api/instrumentation.py` (middleware for tracing)
  - [ ] Log all requests: method, path, user, request_id, status, latency
  - [ ] Tests: logs are valid JSON; request IDs trace end-to-end
- **Acceptance:** All logs JSON; request tracing visible; latency tracked
- **Owner:** You
- **Depends on:** 5.1

**Story 5.4: Prometheus Metrics + Alerts (Phase 5)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Export metrics; Prometheus scraping; Grafana dashboards; alert rules.
- **Tasks:**
  - [ ] Create `api/metrics.py` (Counter, Histogram, Gauge)
  - [ ] Add `/metrics` endpoint
  - [ ] Wire metrics into critical endpoints
  - [ ] Create `prometheus.yml` (scrape config)
  - [ ] Create `grafana/` dashboard (API latency, requests, errors)
  - [ ] Define alert rules (high latency, error rate, CPU)
- **Acceptance:** Metrics exported; Prometheus scrapes; Grafana dashboard visible
- **Owner:** You
- **Depends on:** 5.3

**Story 5.5: Security Scanning + Dependency Audit (Phase 5)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** SAST, dependency audit, secrets scanning in CI/CD.
- **Tasks:**
  - [ ] Add Bandit (SAST) to CI
  - [ ] Add `pip audit` (dependency vulnerabilities)
  - [ ] Add `safety check` (CVE check)
  - [ ] Add `truffleHog` (secrets scanning)
  - [ ] Create `.github/workflows/security.yml`
  - [ ] Tests: no high/critical findings; pass all scans
- **Acceptance:** CI pipeline runs security checks; no CVEs; no secrets in repo
- **Owner:** You

---

## Epic 6: TESTING & DOCUMENTATION

### CRITICAL Priority

**Story 6.1: Test Infrastructure + Coverage (Phase 6)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** pytest setup; fixture library; coverage targets; CI gates.
- **Tasks:**
  - [ ] Organize `tests/` by domain (data, quant, ml, api, e2e)
  - [ ] Create `tests/conftest.py` (common fixtures: mock fetcher, test data)
  - [ ] Write unit tests for each story (backtest, risk, models, endpoints)
  - [ ] Calculate coverage: `pytest --cov=config --cov=api --cov=core`
  - [ ] Set targets: ≥85% critical, ≥75% overall
  - [ ] Create `.github/workflows/tests.yml` (run on push/PR)
- **Acceptance:** 75%+ coverage; CI passes tests before merge
- **Owner:** You

**Story 6.2: E2E Workflows (Phase 6)**
- **Effort:** L (1-2 weeks)
- **Status:** not-started
- **Description:** End-to-end tests via Playwright or similar.
- **Tasks:**
  - [ ] Create `tests/e2e/test_daily_macro.py` (load ECO panel → see macro data)
  - [ ] Create `tests/e2e/test_backtest_workflow.py` (login → backtest → results)
  - [ ] Create `tests/e2e/test_factor_analysis.py` (portfolio → factor exposures)
  - [ ] Create `tests/e2e/test_llm_agent.py` (ask question → get analysis)
  - [ ] Run E2E as final check before production deploy
- **Acceptance:** 5+ critical workflows pass end-to-end
- **Owner:** You
- **Depends on:** 6.1

**Story 6.3: API Documentation (Phase 6)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** OpenAPI schema; interactive docs; examples.
- **Tasks:**
  - [ ] Update all endpoint docstrings (description, params, responses)
  - [ ] Ensure OpenAPI schema valid: `curl http://localhost:8000/openapi.json`
  - [ ] Update `API_DOCUMENTATION.md` (map all endpoints by domain)
  - [ ] Add examples (curl, Python, TypeScript) for key workflows
  - [ ] Tests: OpenAPI schema validates; examples run
- **Acceptance:** `/docs` and `/redoc` fully populated; 100% endpoint coverage
- **Owner:** You
- **Depends on:** All API stories

**Story 6.4: User & Operator Guides (Phase 6)**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** User workflows, operator runbook, troubleshooting.
- **Tasks:**
  - [ ] Update `WORKFLOWS.md` (5+ example user tasks)
  - [ ] Create `DEPLOYMENT_GUIDE.md` (how to deploy to prod)
  - [ ] Create `TROUBLESHOOTING.md` (common issues, logs, debugging)
  - [ ] Create `SECURITY.md` (auth, secrets, best practices)
  - [ ] Create `.env.example` (all required env vars)
- **Acceptance:** New dev can read docs and deploy system
- **Owner:** You
- **Depends on:** All phases

---

## HIGH Priority

**Story H1: CI/CD Pipeline Enhancement**
- **Effort:** L (1 week)
- **Status:** not-started
- **Description:** GitHub Actions for lint, type check, test, build, push to registry.
- **Tasks:**
  - [ ] Create `.github/workflows/ci.yml` (lint, type, test, coverage)
  - [ ] Create `.github/workflows/build.yml` (build Docker image, push to registry)
  - [ ] Add badge to README
  - [ ] Tests: CI passes on every push
- **Acceptance:** CI pipeline passes; Docker image pushed on merge to main

**Story H2: Performance Benchmarking**
- **Effort:** M (1 week)
- **Status:** not-started
- **Description:** Establish performance baseline; track regressions.
- **Tasks:**
  - [ ] Measure page load time (< 2s target)
  - [ ] Measure API latency (< 200ms p95 target)
  - [ ] Measure backtest speed (1 year of data < 5s target)
  - [ ] Create performance dashboard (track over time)
- **Acceptance:** Benchmarks documented; targets met or planned improvements

---

## Summary

| Priority | Epic | Stories | Effort | Phase |
|----------|------|---------|--------|-------|
| CRITICAL | Data | 1.1-1.5 | 5 stories, XL total | 1 |
| CRITICAL | Quant | 2.1-2.5 | 5 stories, L-XL total | 2 |
| CRITICAL | AI/ML | 3.1-3.4 | 4 stories, L-XL total | 3 |
| CRITICAL | Terminal | 4.1-4.4 | 4 stories, L total | 4 |
| CRITICAL | Security | 5.1-5.5 | 5 stories, M-L total | 5 |
| CRITICAL | Testing | 6.1-6.4 | 4 stories, L-M total | 6 |
| HIGH | Infrastructure | H1-H2 | 2 stories | 5-6 |

**Total CRITICAL: 27 stories | Total HIGH: 2 stories | Estimated: 16 weeks**
