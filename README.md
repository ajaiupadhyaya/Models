# Financial Research & Trading Platform

A professional-grade financial research and trading platform with **zero hardcoded data**: all prices, fundamentals, and macro data come from live or cached real sources (yfinance, Polygon, FMP, FRED, NewsAPI, SEC EDGAR). Built with React, FastAPI, PostgreSQL (TimescaleDB), Celery, and Redis.

## Run with Docker Compose

1. **Copy environment and set API keys**
   ```bash
   cp .env.example .env
   # Edit .env: set at least FRED_API_KEY and FMP_API_KEY for full functionality.
   # Optional: POLYGON_API_KEY, NEWSAPI_KEY, ANTHROPIC_API_KEY, etc.
   ```

2. **Start all services**
   ```bash
   docker compose up --build
   ```
   This starts: **backend** (API + optional SPA on port 8000), **postgres** (TimescaleDB, 5432), **redis** (6379), **celery_worker**, **celery_beat**, and **prometheus** (9090). On first run, the backend seeds the DB with 20 large-cap tickers (AAPL, MSFT, GOOGL, etc.) and 5 years of OHLCV plus latest fundamentals. Backend waits for postgres to be healthy before starting.

3. **Open the app**
   - API: http://localhost:8000  
   - Docs: http://localhost:8000/docs  
   - For local frontend dev: `cd frontend && npm install && npm run dev` (Vite on 5173, proxies /api to backend).

4. **Database migrations** (run once; also run automatically on backend startup)
   ```bash
   alembic -c db/alembic.ini upgrade head
   ```

## Required API keys (see .env.example)

| Key | Purpose |
|-----|---------|
| `FRED_API_KEY` | Macro indicators, yield curve |
| `FMP_API_KEY` | Fundamentals, DCF inputs, comps |
| `DATABASE_URL` | PostgreSQL (default: trader@localhost/trading_metrics) |
| `REDIS_URL` | Celery broker (default: redis://localhost:6379/0) |
| `SEC_USER_AGENT` | Required by SEC EDGAR |
| `ANTHROPIC_API_KEY` | AI Research Assistant (Claude) |
| `FINNHUB_API_KEY` | News & sentiment (per-ticker news feed) |
| `NEWSAPI_KEY` | News ingestion (optional; alternative to Finnhub) |
| `POLYGON_API_KEY` | OHLCV/options (optional; yfinance used as fallback) |

## Folder structure

- **frontend/** — React + Tailwind + D3/Recharts; Bloomberg-style terminal UI  
- **backend/** — FastAPI entrypoint (`backend/main.py`); app logic in **api/**, **core/**, **models/**  
- **workers/** — Celery app and ingestion tasks (OHLCV, macro, news, fundamentals)  
- **db/** — Alembic migrations for PostgreSQL + TimescaleDB  
- **docker-compose.yml** — backend, postgres, redis, celery_worker, celery_beat  

## Modules

- **Data pipeline** — Celery jobs refresh OHLCV (daily), macro (weekly), news (hourly), fundamentals (quarterly); data status dashboard in UI. DB seed script pre-loads 20 tickers (5y OHLCV + fundamentals) on first run.
- **Equity research** — Ticker search, company overview, financial statements, DCF, comparable comps, LBO.
- **Quant lab** — Factor ranker (momentum, value, quality, low-vol, size from real DB data), strategy backtester (MA cross, RSI, factor momentum), pairs trading (spread chart, z-score, signals), options chain (yfinance live). **POST /api/v1/quant/backtest** — equity curve, Sharpe, Sortino, CAGR, max drawdown, win rate, alpha/beta vs SPY.
- **Portfolio & risk** — Portfolio valuation, VaR/CVaR, correlation matrix. **POST /api/v1/risk/optimize** — mean-variance optimization, efficient frontier, optimal weights. **POST /api/v1/risk/stress-test** — historical crisis scenarios (2008, COVID, Dot-com, 2022) using real OHLCV.
- **News & sentiment** — **GET /api/v1/news/{symbol}** — per-ticker news (Finnhub/NewsAPI) with VADER sentiment, 7-day aggregate gauge.
- **Macro dashboard** — Yield curve (FRED), bond pricer, key indicators.
- **AI Research Assistant** — Claude-powered chat (**POST /api/v1/ai/chat**) with tool-use: run_dcf, screen_stocks, get_company_overview, run_backtest, get_macro_snapshot. Multi-turn with history.
- **Screener** — Fundamental + price filters (P/E, P/B, sector, market cap), sparklines, sortable table.

### New Phase 2 Panels

| Panel | Description |
|-------|-------------|
| **Backtest** | Strategy config (sma_cross, rsi_mean_reversion, factor_momentum), equity curve, trade log, metrics |
| **Optimizer** | Mean-variance optimization, optimal weights chart, efficient frontier chart with tooltips |
| **Stress Test** | Historical crisis scenarios with drawdown bars; real historical data |
| **News & Sentiment** | Per-ticker news feed with VADER sentiment score and 7-day gauge (no key required) |
| **AI Assistant** | Chat with Claude; tools: DCF, screener, company overview, backtest, macro snapshot |
| **Data Status** | Per-source timestamps; "Refresh Now" triggers Celery tasks for ohlcv/macro/news/fundamentals |  

---

# Financial Models Workspace (Legacy)

A comprehensive, institutional-grade financial modeling framework for quantitative analysis, machine learning, and investment research. This project combines sophisticated models with publication-quality visualizations and a production-ready API.

**Now with high-performance C++ implementations for computationally intensive operations (10-100x faster)!**

## ✅ Status: Deployment Ready (Feb 9, 2026)

**Awesome Quant Integration Complete:**
- ✅ Phase 1: Time-Series + Portfolio Optimization (4/4 tests)
- ✅ Phase 2: Sentiment Analysis + ML Features (7/7 tests)
- ✅ Phase 3: Options Pricing + Deep RL Trading (7/7 tests)

**Deployment Validation: 6/6 tests passed**
- ✅ All APIs operational (16 routers)
- ✅ All models imported successfully
- ✅ Dependencies verified (11/12 installed, 1 optional)
- ✅ Mathematical accuracy validated (Black-Scholes error: 0.000000)

**📖 Quick Start:** See [QUICK_START_FIXED.md](QUICK_START_FIXED.md) for setup  
**🚀 Deployment Report:** [DEPLOYMENT_READINESS_REPORT_2026-02-09.md](DEPLOYMENT_READINESS_REPORT_2026-02-09.md)  
**📊 Phase 3 Summary:** [PHASE_3_SUMMARY.md](PHASE_3_SUMMARY.md)

## 🎯 What's New: Company Analysis System

**Search and analyze any public company with comprehensive automated analysis:**

```bash
# Interactive search and analysis
python analyze_company.py

# Direct ticker analysis
python analyze_company.py TSLA --full

# Search by company name
python analyze_company.py --search "Apple"

# Export detailed report
python analyze_company.py AAPL --full --export report.json
```

**Features:**
- 🔍 **Smart Search**: Fuzzy matching for company names and tickers
- 📊 **Fundamental Analysis**: Complete financial metrics, ratios, efficiency
- 💰 **DCF Valuation**: Intrinsic value with upside/downside calculation
- ⚠️ **Risk Metrics**: VaR, CVaR, volatility, Sharpe ratio, max drawdown
- 📈 **Technical Analysis**: Moving averages, RSI, trend identification
- 🎓 **Automated Grading**: Letter grades (A+ to F) for all metrics
- 💡 **Investment Recommendations**: Buy/Hold/Sell with confidence levels

**See [COMPANY_ANALYSIS_GUIDE.md](COMPANY_ANALYSIS_GUIDE.md) for complete documentation.**

**→ [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** — Step-by-step instructions to get the API and web terminal running.

---

## 🏗️ Project Structure

```
Models/
├── core/                    # Core utilities and data fetching
│   ├── data_fetcher.py      # API integrations (FRED, Alpha Vantage, etc.)
│   ├── visualizations.py   # Advanced charting (D3.js, Plotly)
│   └── utils.py            # Helper functions
├── models/                  # Model templates
│   ├── valuation/          # DCF, multiples, etc.
│   ├── options/            # Options pricing models
│   ├── portfolio/          # Portfolio optimization
│   ├── risk/               # VaR, CVaR, stress testing
│   ├── macro/              # Macroeconomic models
│   └── trading/            # Trading strategies
├── templates/               # Report and presentation templates
│   ├── reports/            # Markdown/LaTeX templates
│   └── presentations/      # Slide deck templates
├── notebooks/              # Example notebooks
├── data/                   # Local data storage
└── config/                 # Configuration files
```

## 🚀 Quick Start

For **full run and deploy steps** (env, two terminals, Docker, production), use **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** as the single place for setup and launch.

### 1. Set Up Environment

```bash
# Create virtual environment (Python 3.12+ required)
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements-api.txt

# Install additional dependencies (optional)
pip install -r requirements.txt

# Build high-performance C++ extensions (optional but recommended)
./build_cpp.sh
# Or on Windows: python setup_cpp.py build_ext --inplace
```

**Note:** Python 3.12+ is required for all dependencies to work correctly.

### 2. Configure API Keys

Create a `.env` file in the root directory:

```env
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

Get free API keys:
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key

### 3. Run Examples

```bash
jupyter lab
```

Navigate to `notebooks/` for example models.

## 📊 Features

### Core Capabilities
- **High-Performance C++ Core**: 10-100x faster calculations for options pricing, Monte Carlo, and risk analytics
- **Real-time Data**: Automatic fetching from FRED, Alpha Vantage, Yahoo Finance with intelligent caching
- **Publication-Quality Visualizations**: NYT/Bloomberg/FT-inspired interactive charts
- **Bloomberg-Style Terminal**: React + D3 web terminal with market overview, charts, and AI assistant
- **Institutional Models**: DCF, Black-Scholes, Portfolio Optimization, Risk Models
- **Macro Analysis**: Economic indicators, unemployment, GDP, inflation, yield curves
- **Report Generation**: Automated report and slide deck creation
- **Trading Strategies**: Professional backtesting with transaction costs and slippage

### API & AI/ML/RL/DL
- **REST API**: FastAPI at `/api/v1` — models, predictions, backtest, AI, monitoring, WebSocket. See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) and `http://localhost:8000/docs` when running.
- **AI**: OpenAI-powered stock analysis, sentiment, and trading insight via `/api/v1/ai/*`.
- **ML/DL**: Ensemble and LSTM on-the-fly predictions; `GET /api/v1/predictions/quick-predict` for ML signal without pre-loaded models.
- **RL**: Orchestrator and automation endpoints use Stable-Baselines3 (PPO/DQN) when `schedule` is installed.

### Advanced Features
- **Machine Learning**: Time series forecasting, regime detection, anomaly detection
- **Advanced Macro Models**: Nelson-Siegel yield curves, Phillips Curve, Okun's Law, business cycle analysis
- **Walk-Forward Optimization**: Professional parameter optimization
- **Data Caching**: Intelligent caching system for performance
- **Advanced Risk Models**: VaR, CVaR, stress testing, scenario analysis
- **Multiple Visualization Types**: Waterfall, Sankey, small multiples, correlation networks, radar charts, treemaps

## 📚 Model Categories

### Valuation Models
- Discounted Cash Flow (DCF)
- Comparable Company Analysis
- Precedent Transactions
- Sum of Parts

### Options & Derivatives
- Black-Scholes-Merton
- Binomial Tree
- Monte Carlo Simulation
- Greeks Calculation

### Portfolio Management
- Mean-Variance Optimization
- Risk Parity
- Black-Litterman
- Factor Models

### Risk Management
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Stress Testing
- Scenario Analysis

### Macro Economics
- GDP Forecasting
- Unemployment Analysis
- Inflation Models
- Yield Curve Analysis

### Trading Strategies
- Momentum
- Mean Reversion
- Pairs Trading
- Factor Investing

## 🚀 Awesome Quant Integration (NEW!)

**We've created a comprehensive integration guide to upgrade your project with best-in-class libraries from the Awesome Quant ecosystem!**

### 📖 Read the Integration Guides

1. **[AWESOME_QUANT_INTEGRATION_GUIDE.md](AWESOME_QUANT_INTEGRATION_GUIDE.md)** — Master guide covering:
   - Gap analysis: What you have vs. what's in Awesome Quant
   - 3-phase roadmap for integration (Immediate, Medium-term, Advanced)
   - ROI assessment for each library
   - Implementation examples

2. **[PHASE1_IMPLEMENTATION_GUIDE.md](PHASE1_IMPLEMENTATION_GUIDE.md)** — Ready-to-implement code for:
   - ✅ **Advanced Time-Series** (auto-ARIMA forecasting with pmdarima)
   - ✅ **CVaR Portfolio Optimization** (tail-risk aware portfolios with riskfolio-lib)
   - ✅ **Trading Calendar** (NYSE/NASDAQ awareness via exchange-calendars)
   - ✅ **Enhanced Metrics** (Sortino, Calmar ratios via empyrical)
   - ✅ **Factor Analysis** (alphalens tearsheets and IC analysis)

### 🚀 Quick Start: Phase 1 (This Week)

```bash
# Install Phase 1 advanced quant libraries
pip install -r requirements-quant-phase1.txt

# This adds:
# - pmdarima (auto-ARIMA forecasting)
# - riskfolio-lib (CVaR, entropy pooling)
# - empyrical (comprehensive risk metrics)
# - alphalens (factor analysis)
# - exchange-calendars (trading day awareness)
```

Then follow [PHASE1_IMPLEMENTATION_GUIDE.md](PHASE1_IMPLEMENTATION_GUIDE.md) to:
1. Copy implementation files
2. Add new API endpoints
3. Test with provided examples
4. Deploy enhanced capabilities

### 📊 Impact Summary

| Capability | Current | After Phase 1 | Benefit |
|-----------|---------|---------------|---------|
| Portfolio optimization | Mean-variance, risk parity | + CVaR, entropy pooling | Tail-risk aware |
| Time-series forecasting | Custom ARIMA | Auto-ARIMA + SARIMAX | Better accuracy |
| Factor analysis | Manual | Alphalens framework | Systematic alpha |
| Risk metrics | Sharpe, Sortino | + Calmar, capture ratios | Comprehensive view |
| Backtesting | Calendar-agnostic | NYSE/NASDAQ aware | Fewer errors |

### 🎓 Phase 2 & 3 Complete

**✅ Phase 2: Advanced Sentiment & Machine Learning**
- Sentiment Analysis: FinBERT-powered headlines and news analysis
- Multi-Factor Models: Fama-French style factor analysis with alpha estimates
- ML Label Generation: Fixed horizon, triple-barrier, and meta-labels
- Feature Engineering: Fractional differencing, time-decay weighting, stationary transformations

**✅ Phase 3: Institutional Options Desk & Deep RL Trading**
- Black-Scholes European Options: Pricing, Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied Volatility: Brent's method solver with <0.0001 tolerance
- Option Analysis: Intrinsic value, time value, moneyness classification
- Deep RL Trading: Custom Gymnasium environments with Stable-Baselines3 (PPO, A2C, DQN)

**See [PHASE_3_SUMMARY.md](PHASE_3_SUMMARY.md) for complete Phase 3 implementation details.**

---

## 🧰 Quant Research Resources

We maintain a curated catalog of quant and CS tools, research, and learning resources in
[awesomequantreadme.md](awesomequantreadme.md). Use it to extend this project responsibly:

- **Core analytics stack**: Prioritize numpy, pandas, scipy, statsmodels, and arch for reliable baselines.
- **Backtesting & research**: Evaluate vectorbt, backtrader, QSTrader, quantstats, and pyfolio-reloaded to expand experiments.
- **Portfolio & risk**: Consider PyPortfolioOpt and Riskfolio-Lib for optimization and risk modeling.
- **Data tooling**: Explore pandas-datareader, yfinance, and vendor SDKs as alternate data sources.

**See [AWESOME_QUANT_INTEGRATION_GUIDE.md](AWESOME_QUANT_INTEGRATION_GUIDE.md) for how we incorporated these into your project.**

If you adopt a library from the list, document why it was chosen and add minimal examples in notebooks.

## 🔧 Quick Examples

### High-Performance C++ Quant Library
```python
from quant_accelerated import BlackScholesAccelerated, MonteCarloAccelerated

# 10x faster Black-Scholes options pricing
price = BlackScholesAccelerated.call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

# 100x faster Monte Carlo simulation
mc = MonteCarloAccelerated()
option_price = mc.price_european_option(100, 100, 1.0, 0.05, 0.2, True, 1000000)
```

### Bloomberg-Style Terminal (Web UI)
```bash
# Terminal 1: Start the API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start the React frontend
cd frontend && npm install && npm run dev
# Opens at http://localhost:5173
```
See [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) for full setup.

### Advanced Visualizations
```python
from core.advanced_visualizations import PublicationCharts

# Publication-quality waterfall chart
fig = PublicationCharts.waterfall_chart(data, title="Valuation Breakdown")
fig.show()
```

### Machine Learning Forecasting
```python
from models.ml.forecasting import TimeSeriesForecaster

forecaster = TimeSeriesForecaster(model_type='random_forest')
forecaster.fit(prices, n_lags=20)
forecast = forecaster.predict(prices, n_periods=30)
```

### Professional Backtesting
```python
from models.trading.backtesting import BacktestEngine

engine = BacktestEngine(commission=0.001, slippage=0.0005)
results = engine.run_backtest(prices, signals)
```

See `notebooks/` for detailed examples and [ARCHITECTURE.md](ARCHITECTURE.md) for methodology and data validation.

## 🚀 High-Performance C++ Library

This project includes a high-performance C++ quantitative finance library, similar to what Jane Street and other top quant firms use. The C++ library provides:

- **10-100x faster** calculations for options pricing, Monte Carlo simulations, and risk analytics
- **Production-grade** numerical algorithms with careful attention to precision and stability
- **Seamless integration** with Python via pybind11
- **Automatic fallback** to pure Python if not built

### Quick Start with C++

```bash
# Build the C++ library (one-time setup)
./build_cpp.sh

# Use in Python (same API as pure Python)
from quant_accelerated import BlackScholesAccelerated, MonteCarloAccelerated
```

For complete documentation, see **[CPP_QUANT_GUIDE.md](CPP_QUANT_GUIDE.md)**

### Performance Comparison

| Operation | Pure Python | C++ | Speedup |
|-----------|------------|-----|---------|
| Black-Scholes (10k calls) | 500 ms | 5 ms | 100x |
| Monte Carlo (1M paths) | 30 s | 0.3 s | 100x |
| Portfolio analytics | 100 μs | 5 μs | 20x |

## 🧪 Running tests

### Backend (pytest)
```bash
# Create and activate a virtual environment first (see Quick Start)
source venv/bin/activate   # or: .venv/bin/activate, or: uv venv && source .venv/bin/activate
pip install -r requirements.txt   # includes pytest, pytest-cov

# Run all backend tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=config --cov=api --cov=core --cov=models/risk --cov-report=term-missing
```

### Frontend (Vitest)
```bash
cd frontend
npm install
npm run test
```

See [WORKFLOWS.md](WORKFLOWS.md) for step-by-step terminal workflows and [ARCHITECTURE.md](ARCHITECTURE.md) for backtest methodology and data validation.

## Deployment

To run in production or with Docker: copy `.env.example` to `.env`, set your API keys, then follow **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** (sections 4–5 for Docker and production). See [DOCKER.md](DOCKER.md) for Docker-specific options.

## 📝 License

For personal and professional use.
