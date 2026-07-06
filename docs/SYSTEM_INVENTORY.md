# System Inventory - Archived Baseline

**Generated:** February 9, 2026  
**Status:** Historical Phase 0 baseline. Not the current source of truth.
**Location:** /Users/ajaiupadhyaya/Documents/Models/

This document is retained as an audit snapshot from the pre-v1 cleanup. For current project status, use `README.md`, `GETTING_STARTED.md`, `ARCHITECTURE.md`, `API_DOCUMENTATION.md`, and `docs/FEATURE_BACKLOG.md`.

---

## A. DATA LAYER

### Operational (Production)

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **yfinance Session** | `core/yfinance_session.py` | ✅ Active | curl_cffi with chrome110 impersonation; fallback to None |
| **FRED Integration** | `core/data_fetcher.py` | ✅ Active | 1000+ macro indicators (unemployment, GDP, rates, etc.) |
| **Alpha Vantage** | `core/data_fetcher.py` | ✅ Active | Equity OHLCV + technical indicators |
| **UnifiedDataFetcher** | `core/data_fetcher.py` | ✅ Active | Routes to yfinance/FRED/Alpha Vantage; backoff retry logic |
| **Data Caching** | `core/data_fetcher.py` + Redis | ✅ Active | 5min TTL prices, 1hr TTL company info; 315x speedup |
| **Cache Stats** | `api/monitoring.py` | ✅ Active | Cache hit rates tracked & reported |

### Missing / TODO

| Component | Phase | Priority | Effort |
|-----------|-------|----------|--------|
| Polygon.io Provider | 1 | CRITICAL | 3 days |
| IEX Cloud Provider | 1 | CRITICAL | 3 days |
| OANDA FX Provider | 1 | HIGH | 3 days |
| CoinGecko Crypto | 1 | HIGH | 2 days |
| SEC EDGAR Fundamentals | 1 | HIGH | 4 days |
| NewsAPI Integration | 1 | HIGH | 2 days |
| World Bank + IMF Data | 1 | HIGH | 3 days |
| Point-in-Time Snapshots | 1 | CRITICAL | 3 days |
| Cold Storage (Parquet) | 1 | CRITICAL | 5 days |
| Historical Backfill (10yr) | 1 | HIGH | 3 days |

---

## B. QUANT ENGINE

### Operational (Production)

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Risk Analytics** | `models/risk/` | ✅ Active | VaR, CVaR, volatility, Sharpe, max drawdown |
| **Backtesting Engine** | `core/backtesting.py` | ✅ Active | Institutional costs (slippage, commissions, market impact) |
| **Black-Scholes** | `models/options_pricing.py` | ✅ Implemented | Call/put valuation (Greeks not exposed) |
| **Technical Indicators** | `models/technical/` | ✅ Implemented | SMA, EMA, RSI, MACD, Bollinger Bands, etc. |
| **Sentiment Analysis** | `models/sentiment.py` | ⚠️ Basic | Bag-of-words; not integrated to terminal |

### Partial / In-Development

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Factor Models** | `models/factor_models.py` | 🔴 Stub | Structure exists; Fama-French not wired |
| **Regime Detection** | `models/regime_detection.py` | 🔴 Stub | HMM concept; not integrated |
| **Stress Testing** | N/A | 🔴 Not started | Shock scenarios, correlation shifts |
| **Cross-Asset Backtest** | `core/backtesting.py` | 🔴 Equity only | Options, futures, crypto not supported |

---

## C. AI / ML / RL

### Operational (Production)

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **OpenAI Integration** | `core/ai_analysis.py` | ✅ Active | GPT-4 stock analysis, research reports |
| **Ensemble Predictor** | `models/ensemble_predictor.py` | ✅ Active | 3-model ensemble for return prediction |
| **LSTM Predictor** | `models/lstm_predictor.py` | ✅ Active | Sequence modeling on technical indicators |
| **AutoML (AutoGluon)** | `models/automl.py` | ✅ Active | Multi-task model training |

### Missing / TODO

| Component | Phase | Priority | Effort |
|-----------|-------|----------|--------|
| MLflow Integration | 3 | CRITICAL | 5 days |
| Model Registry | 3 | CRITICAL | 3 days |
| Feature Store | 3 | CRITICAL | 5 days |
| LLM Agent (Tools) | 3 | CRITICAL | 7 days |
| RL Env + PPO | 3 | CRITICAL | 7 days |
| Hyperparameter Tuning | 3 | HIGH | 5 days |
| Cross-validation Framework | 3 | HIGH | 3 days |
| Model Evaluation Metrics | 3 | HIGH | 3 days |

---

## D. API LAYER

### Operational (Production)

| Component | File | Endpoints | Status | Details |
|-----------|------|-----------|--------|---------|
| **Data API** | `api/data_api.py` | 12 | ✅ Active | OHLCV, macro, news, fundamentals |
| **Backtesting API** | `api/backtesting_api.py` | 8 | ✅ Active | Backtest, results, stats, export |
| **Risk API** | `api/risk_api.py` | 9 | ✅ Active | VaR, CVaR, portfolio risk, stress |
| **AI Analysis API** | `api/ai_analysis_api.py` | 6 | ✅ Active | Stock analysis, sentiment, research |
| **Paper Trading API** | `api/paper_trading_api.py` | 8 | ✅ Active | Virtual trading, P&L, positions |
| **Screener API** | `api/screener_api.py` | 7 | ✅ Active | Multi-factor screening, export |
| **Models API** | `api/models_api.py` | 5 | ✅ Active | Prediction, train, evaluate |
| **Company Analysis API** | `api/company_analysis_api.py` | 4 | ✅ Active | Fundamentals, peers, valuation |
| **News API** | `api/news_api.py` | 3 | ✅ Active | Fetch news, sentiment |
| **Predictions API** | `api/predictions_api.py` | 5 | ✅ Active | ML predictions, confidence |
| **Monitroing API** | `api/monitoring.py` | 2 | ✅ Active | Cache stats, system health |
| **WebSocket API** | `api/websocket_api.py` | 3 | ✅ Active | Real-time prices, predictions, live log |
| **Automation API** | `api/automation_api.py` | 4 | ✅ Active | Scheduled tasks, triggers |
| **Rate Limit** | `api/rate_limit.py` | 1 | ⚠️ Basic | Fallback plan exists; not wired |
| **Auth API** | `api/auth_api.py` | 1 | 🔴 Stub | Not implemented |
| **Orchestrator API** | `api/orchestrator_api.py` | 4 | ✅ Active | Automation engine, PPO/DQN concepts |

**Total: 99 routes across 16 routers**

### Missing / TODO

| Component | Phase | Priority |
|-----------|-------|----------|
| OAuth2 + JWT | 5 | CRITICAL |
| Token Management | 5 | CRITICAL |
| Rate Limiting Enforcement | 5 | CRITICAL |
| API Documentation (OpenAPI) | 6 | HIGH |
| Input Validation (Pydantic) | Throughout | HIGH |

---

## E. FRONTEND / TERMINAL

### Operational (Production)

| Component | Location | Status | Tech Stack | Details |
|-----------|----------|--------|-----------|---------|
| **Command Bar** | `frontend/src/components/CommandBar.tsx` | ✅ Active | React + TypeScript | 12+ commands (GP, FA, FLDS, QUANT, ECO, etc.) |
| **Candlestick Chart** | `frontend/src/components/charts/CandlestickChart.tsx` | ✅ Active | D3.js | OHLCV with volume subplot |
| **Primary Instrument Panel** | `frontend/src/components/panels/PrimaryInstrumentPanel.tsx` | ✅ Active | React | Candlestick + info |
| **Fundamental Analysis Panel** | `frontend/src/components/panels/FundamentalAnalysisPanel.tsx` | ✅ Active | React | P/E, book value, growth rates |
| **Technical Analysis Panel** | `frontend/src/components/panels/TechnicalAnalysisPanel.tsx` | ✅ Active | React | Indicators (SMA, RSI, MACD) |
| **Quantitative Panel** | `frontend/src/components/panels/QuantitativePanel.tsx` | ✅ Active | React | Risk metrics, factor exposures |
| **Economic Panel** | `frontend/src/components/panels/EconomicPanel.tsx` | ✅ Active | React | Macro indicators (unemployment, rates) |
| **News Panel** | `frontend/src/components/panels/NewsPanel.tsx` | ✅ Active | React | News headlines + sentiment |
| **Portfolio Panel** | `frontend/src/components/panels/PortfolioPanel.tsx` | ✅ Active | React | Holdings, allocation, P&L |
| **Screening Panel** | `frontend/src/components/panels/ScreeningPanel.tsx` | ✅ Active | React | Multi-factor filters, results table |
| **Workspace Context** | `frontend/src/context/WorkspaceContext.tsx` | ⚠️ Basic | React Context | Holds state; persistence not wired |

### Missing / TODO

| Component | Phase | Priority | Details |
|-----------|-------|----------|---------|
| Volatility Surface Chart | 4 | CRITICAL | D3 + 3D surface |
| Correlation Heatmap | 4 | HIGH | Color-coded correlation matrix |
| Factor Exposure Chart | 4 | HIGH | Stacked bar + contribution |
| Regime Shift Timeline | 4 | HIGH | Colored timeline with regimes |
| Options Greeks Display | 4 | HIGH | Greeks surface, Greeks sensitivity |
| Advanced Time Series | 4 | MEDIUM | Candlestick with more overlays |
| Workspace Persistence | 4 | MEDIUM | localStorage + sync |
| Dark Mode | 4 | MEDIUM | CSS theme toggle |
| Real-time Updates | 4 | MEDIUM | WebSocket integration |

---

## F. INFRASTRUCTURE & DEVOPS

### Operational (Production)

| Component | Location | Status | Details |
|-----------|----------|--------|---------|
| **Docker Image** | `Dockerfile` | ✅ Active | Python 3.10, FastAPI, Redis, Postgres |
| **Docker Compose** | `docker-compose.yml` | ✅ Active | API, Redis, Postgres, Prometheus (local) |
| **Render Deployment** | `render.yaml` | ✅ Active | Auto-deployed on push to main |
| **Environment Config** | `config/settings.py` | ✅ Active | `.env` loaded; overrides per env |
| **Logging** | `core/logging.py` | ⚠️ Basic | File + console; not structured JSON |
| **Monitoring** | `api/monitoring.py` | ⚠️ Basic | Cache stats, health check |
| **Redis Cache** | docker-compose | ✅ Active | In-memory cache (5min/1hr TTLs) |
| **PostgreSQL DB** | docker-compose | ✅ Active | Paper trading state, backtest results |

### Missing / TODO

| Component | Phase | Priority | Effort |
|-----------|-------|----------|--------|
| Prometheus Metrics | 5 | CRITICAL | 5 days |
| Grafana Dashboards | 5 | CRITICAL | 3 days |
| Structured JSON Logging | 5 | CRITICAL | 3 days |
| Distributed Tracing (Jaeger) | 5 | HIGH | 5 days |
| Alert Rules | 5 | HIGH | 3 days |
| Security Scanning (Bandit) | 5 | HIGH | 2 days |
| Dependency Audit | 5 | HIGH | 1 day |
| Secrets Management | 5 | MEDIUM | 3 days |

---

## G. TESTING

### Existing Tests

| Path | Status | Count | Coverage |
|------|--------|-------|----------|
| `tests/test_data_fetcher.py` | ✅ Exists | ~15 | Data fetching edge cases |
| `tests/test_backtesting.py` | ✅ Exists | ~10 | Backtest logic, PnL calc |
| `tests/test_risk.py` | ✅ Exists | ~8 | VaR, CVaR, Sharpe |
| `tests/api/test_*.py` | ✅ Exists | ~30 | Static API validation (61/61 pass) |
| `test_live_api.py` | ✅ Exists | ~20 | Live endpoint integration |

**Estimated Coverage:** ~25% of critical paths

### Missing / TODO

| Component | Phase | Priority |
|-----------|-------|----------|
| Unit Tests (all modules) | 6 | CRITICAL |
| Integration Tests | 6 | CRITICAL |
| E2E Tests (Playwright) | 6 | CRITICAL |
| Performance Tests | 6 | HIGH |
| Load Tests (k6) | 6 | MEDIUM |
| Security Tests | 5 | CRITICAL |
| CI/CD Pipeline | 5 | CRITICAL |

**Target Coverage:** 75%+ critical, 85%+ for core components

---

## H. DOCUMENTATION

### Existing Docs

| Document | Status | Details |
|----------|--------|---------|
| `README.md` | ✅ Current | Project overview, quick start |
| `GETTING_STARTED.md` | ✅ Current | Local setup, run, test, and build commands |
| `API_DOCUMENTATION.md` | ✅ Current | All 99 endpoints documented |
| `ARCHITECTURE.md` | ✅ Current | System design, modules, data flow |
| `DEPLOYMENT_GUIDE.md` | ✅ Current | Free-tier deploy paths and verification checklist |
| `TROUBLESHOOTING.md` | ✅ Current | Common deploy and runtime failure modes |
| `SECURITY.md` | ✅ Current | Auth model, secret handling, and disclosure |
| `docs/FEATURE_BACKLOG.md` | ✅ Current | v1 scope, release status, and v2 roadmap |
| `BACKTEST_METHODOLOGY.md` | ✅ Current | Backtest engine details |
| `DOCKER.md` | ✅ Current | Docker build/run |

### Missing / TODO

| Document | Phase | Priority |
|-----------|-------|----------|
| Comprehensive API Guide (all endpoints) | 6 | CRITICAL |
| Operator Runbook | 6 | HIGH |
| Troubleshooting Guide | 6 | HIGH |
| Security Best Practices | 5 | HIGH |
| ML Model Governance | 3 | HIGH |
| Data Dictionary | 1 | HIGH |
| Configuration Reference | Throughout | MEDIUM |

---

## I. DEPENDENCY INVENTORY

### Core Dependencies

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| FastAPI | 1.0+ | Web framework | ✅ Current |
| Pydantic | 2.x | Data validation | ✅ Current |
| yfinance | 0.2+ | Equity data | ✅ Current |
| pandas | 2.x | Data manipulation | ✅ Current |
| numpy | 1.x | Numerical computing | ✅ Current |
| scikit-learn | 1.3+ | ML models | ✅ Current |
| tensorflow/keras | 2.13+ | Deep learning | ✅ Current |
| openai | 1.x | GPT integration | ✅ Current |
| redis | 5.x | Caching | ✅ Current |
| psycopg2 | 2.9+ | PostgreSQL | ✅ Current |
| pytest | 7.x | Testing | ✅ Current |
| requests | 2.x | HTTP | ✅ Current |
| curl_cffi | Latest | yfinance support | ✅ Current |
| stable-baselines3 | 2.x | RL models | ✅ Current |
| gymnasium | 0.x | RL environment | ✅ Current |
| autogluon | Latest | AutoML | ✅ Current |

### Frontend Dependencies

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| React | 18+ | UI framework | ✅ Current |
| TypeScript | 5.x | Type safety | ✅ Current |
| D3.js | 7.x | Visualization | ✅ Current |
| Vite | Latest | Build tool | ✅ Current |
| Tailwind CSS | Latest | Styling | ✅ Current |

### Dev/Test Dependencies

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| black | Latest | Code formatting | ✅ Current |
| pylint | Latest | Linting | ✅ Current |
| mypy | Latest | Type checking | ✅ Current |
| pytest-cov | Latest | Coverage | ✅ Current |
| bandit | Latest | Security analysis | ⚠️ Installed, not in CI |

---

## J. DEPLOYMENT STATUS

### Production (Render)

- **URL:** https://models-app.onrender.com
- **Last Deploy:** [See git log]
- **Status:** ✅ Running
- **Environment:** Python 3.10, gunicorn, Redis, Postgres
- **Auto-Deploy:** Yes (on push to main)

### Local Development

- **Setup:** `python -m venv venv` + `pip install -r requirements.txt`
- **Run:** `python api/main.py` or `uvicorn api.main:app --reload`
- **Test:** `pytest tests/`
- **Docker:** `docker-compose up -d`

---

## K. PHASE 0 COMPLETION CHECKLIST

**Objective:** Lock in requirements and create working artifact list

- [x] Create FEATURE_BACKLOG.md (27 stories)
- [x] Create SYSTEM_INVENTORY.md (this doc)
- [x] Identify all gaps vs context.md
- [ ] Create CONTEXT_COMPLIANCE_MATRIX.md (map requirements to features)
- [ ] Validate no critical bugs in production
- [ ] Get team sign-off on backlog + timelines
- [ ] Set up tracking system (GitHub Projects, Jira, or linear)
- [ ] Schedule kickoff for Phase 1 (Week 2)

---

## Summary

**Production-Ready Components:** 60+ (Data, APIs, Core ML models, Infrastructure)

**Partial Components:** 8 (Factor models, regime detection, cross-asset backtest, testing, etc.)

**Missing Components:** 40+ (Mostly Phase 1-6 roadmap items)

**Current Test Coverage:** ~25% | **Target:** 75%+

**Deployment:** Render (auto-deploy on main)

**Next Steps:** Approve backlog → Begin Phase 1 (Week 2)
