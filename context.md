# Financial Analysis Terminal - Development Prompt

## Project Overview
Build a comprehensive, institutional-grade financial analysis terminal web application comparable to Bloomberg Terminal, combining fundamental analysis, quantitative modeling, real-time data, and AI-powered insights into a unified platform.

## Core Requirements

### 1. Architecture & Tech Stack

**Frontend:**
- Modern React/Next.js application with TypeScript
- D3.js for advanced data visualizations and interactive charts
- TailwindCSS for styling with a dark, professional theme
- Real-time WebSocket connections for live data streaming
- Responsive grid layout system for customizable dashboards

**Backend:**
- Python FastAPI or Node.js Express for API services
- PostgreSQL/TimescaleDB for time-series financial data storage
- Redis for caching and real-time data management
- Celery or Bull for background task processing
- Docker containerization for all services

**AI/ML Infrastructure:**
- TensorFlow/PyTorch for deep learning models
- scikit-learn for traditional ML algorithms
- Ray or Dask for distributed computing
- MLflow for model versioning and experiment tracking
- Stable Baselines3 for reinforcement learning models

### 2. Data Integration Layer

**Required API Integrations:**
- **Market Data:** Alpha Vantage, Polygon.io, IEX Cloud, or Twelve Data
- **Fundamental Data:** Financial Modeling Prep, Quandl, SEC EDGAR API
- **News & Sentiment:** NewsAPI, Finnhub, Twitter API
- **Economic Data:** FRED API, World Bank API, IMF Data
- **Alternative Data:** Reddit sentiment, Google Trends, satellite imagery providers
- **Cryptocurrency:** CoinGecko, Binance API (if applicable)

**Data Pipeline:**
- Real-time streaming data ingestion
- Historical data backfill capabilities (minimum 10 years)
- Data validation and cleaning pipelines
- Automated daily/hourly data refresh jobs
- Data normalization across different sources

### 3. AI/ML Models Suite

**Predictive Models:**
- **Time Series Forecasting:**
  - LSTM/GRU networks for price prediction
  - Transformer models (Temporal Fusion Transformers)
  - ARIMA/SARIMA for statistical baselines
  - Prophet for trend and seasonality detection

- **Classification Models:**
  - Buy/Hold/Sell signal generation
  - Market regime detection (bull/bear/sideways)
  - Earnings surprise prediction
  - Credit risk assessment

- **Reinforcement Learning:**
  - Deep Q-Networks (DQN) for portfolio optimization
  - Proximal Policy Optimization (PPO) for trading strategies
  - Multi-agent systems for market simulation
  - Continuous training on live market feedback

**NLP & Sentiment Analysis:**
- Fine-tuned FinBERT for financial text analysis
- Real-time news sentiment scoring
- Earnings call transcript analysis
- Social media sentiment aggregation
- SEC filing key information extraction

**Quantitative Models:**
- Factor models (Fama-French, custom factors)
- Risk models (VaR, CVaR, Expected Shortfall)
- Portfolio optimization (Mean-Variance, Black-Litterman, Risk Parity)
- Options pricing (Black-Scholes, Monte Carlo, Binomial trees)
- Statistical arbitrage pair trading algorithms

### 4. Terminal Interface Modules

**Main Dashboard:**
- Customizable widget-based layout (drag-and-drop)
- Real-time market overview (indices, futures, forex, crypto)
- Personalized watchlist with live updates
- AI-generated market summary and daily insights
- Economic calendar with event impact predictions

**Analysis Modules:**

1. **Fundamental Analysis Tab:**
   - Income statement, balance sheet, cash flow analysis
   - Financial ratios and metrics (P/E, P/B, ROE, DCF valuations)
   - Peer comparison tables
   - Historical trend charts
   - AI-generated company health score

2. **Technical Analysis Tab:**
   - Interactive candlestick/OHLC charts with 50+ indicators
   - Pattern recognition (head and shoulders, triangles, etc.)
   - Custom indicator builder
   - Multi-timeframe analysis
   - Backtesting interface for technical strategies

3. **Quantitative Models Tab:**
   - Pre-built model library (momentum, mean reversion, statistical arbitrage)
   - Model performance dashboard with Sharpe ratio, max drawdown, win rate
   - Factor exposure analysis
   - Walk-forward optimization interface
   - Live model deployment controls

4. **AI Insights Tab:**
   - Real-time sentiment scores across multiple sources
   - Predictive price targets with confidence intervals
   - Anomaly detection alerts
   - Correlation analysis and market regime identification
   - Natural language query interface ("What's driving tech stocks today?")

5. **Portfolio Management Tab:**
   - Portfolio construction and optimization
   - Risk analytics and stress testing
   - Performance attribution
   - Rebalancing recommendations
   - Tax-loss harvesting suggestions

6. **Screening & Discovery Tab:**
   - Advanced multi-factor screener
   - AI-powered opportunity discovery
   - Relative value analysis
   - Unusual options activity detection
   - Insider trading tracker

7. **Economic Analysis Tab:**
   - Macro indicator dashboard
   - Central bank policy tracker
   - Yield curve analysis
   - Inflation and employment data visualization
   - Global market correlation heatmaps

8. **News & Research Tab:**
   - Real-time financial news aggregator
   - AI-summarized articles and research reports
   - Sentiment timeline for specific securities
   - Earnings call transcripts with key highlights
   - Customizable news alerts

### 5. Visual Design Requirements

**Bloomberg-Inspired Aesthetic:**
- Dark theme (charcoal #1a1a1a to black #000000 backgrounds)
- Accent colors: Bloomberg orange (#f5a623), terminal green (#00ff41), alert red (#ff3b3b)
- Monospace fonts for numerical data (JetBrains Mono, IBM Plex Mono)
- Sans-serif for text (Inter, Roboto, or SF Pro)
- High information density without clutter
- Subtle grid lines and separators
- Glass-morphism effects for modals and overlays

**Chart Specifications:**
- Professional candlestick charts with volume bars
- Smooth animations and transitions
- Responsive tooltips with detailed information
- Zoom and pan capabilities
- Drawing tools (trendlines, Fibonacci retracements)
- Screenshot/export functionality
- Multiple chart types (line, bar, area, heatmap, treemap)

**Data Tables:**
- Sortable and filterable columns
- Conditional formatting (color-coded gains/losses)
- Sparkline charts in cells
- Export to CSV/Excel functionality
- Infinite scroll or virtualized rendering for performance

### 6. AI/ML Training Pipeline

**Historical Data Requirements:**
- Minimum 10 years of daily OHLCV data
- Tick data for high-frequency models (if applicable)
- Fundamental data updated quarterly
- News archives for sentiment training
- Macro-economic indicators back to 2000

**Training Infrastructure:**
- Automated retraining schedules (daily/weekly/monthly)
- Cross-validation with walk-forward testing
- Out-of-sample performance tracking
- A/B testing framework for model comparison
- Gradual deployment with rollback capabilities

**Model Monitoring:**
- Real-time prediction accuracy tracking
- Data drift detection
- Performance degradation alerts
- Explainable AI dashboards (SHAP values, feature importance)
- Model versioning and experiment logs

### 7. Advanced Features

**Automation:**
- Scheduled report generation
- Automated alert system (price targets, technical signals, news events)
- Auto-rebalancing portfolios based on signals
- Paper trading mode for strategy validation
- Webhook integrations for external services

**Collaboration:**
- Share custom dashboards and strategies
- Collaborative annotations on charts
- Team portfolio management
- Research notes and idea tracking

**API Access:**
- RESTful API for programmatic access
- WebSocket streams for real-time data
- Python/JavaScript SDK
- Rate limiting and authentication
- Comprehensive API documentation

### 8. Performance & Scalability

**Requirements:**
- Page load time < 2 seconds
- Real-time data latency < 100ms
- Support 10,000+ concurrent users
- Handle 1M+ API requests per day
- 99.9% uptime SLA

**Optimization:**
- CDN for static assets
- Database query optimization and indexing
- Lazy loading for charts and data
- Service worker for offline capabilities
- Horizontal scaling for API servers

### 9. Security & Compliance

- OAuth 2.0 authentication
- Role-based access control (RBAC)
- End-to-end encryption for sensitive data
- Audit logs for all transactions
- GDPR/CCPA compliance measures
- Regular security audits and penetration testing

## Implementation Phases

**Phase 1 (Weeks 1-4): Foundation**
- Set up development environment and infrastructure
- Integrate 3-5 core data APIs
- Build basic UI framework with dark theme
- Implement authentication system
- Create database schema and data ingestion pipeline

**Phase 2 (Weeks 5-8): Core Analytics**
- Develop fundamental analysis module
- Build interactive D3.js charting library
- Implement technical analysis indicators
- Create real-time data streaming
- Develop basic portfolio tracking

**Phase 3 (Weeks 9-12): AI/ML Integration**
- Train initial predictive models on historical data
- Implement sentiment analysis pipeline
- Build model deployment infrastructure
- Create AI insights dashboard
- Develop screening and discovery tools

**Phase 4 (Weeks 13-16): Advanced Features**
- Implement quantitative strategy backtesting
- Deploy reinforcement learning trading agents
- Build economic analysis module
- Create collaborative features
- Develop comprehensive API

**Phase 5 (Weeks 17-20): Optimization & Launch**
- Performance optimization and load testing
- Security hardening
- User testing and feedback iteration
- Documentation and tutorials
- Production deployment

## Success Metrics

- All core modules functional with real-time data
- AI models achieving >55% directional accuracy
- Platform handling 1000+ concurrent users smoothly
- Average user session time >15 minutes
- Professional-grade UI indistinguishable from commercial terminals
- Complete API documentation with code examples
- 95% test coverage for critical components

## Additional Considerations

- Implement comprehensive error handling and logging
- Create detailed user documentation and video tutorials
- Build admin panel for system monitoring
- Develop mobile-responsive version
- Consider Progressive Web App (PWA) capabilities
- Plan for white-labeling options
- Integrate paper trading and simulation environments
- Build notification system (email, SMS, push)

## Deliverables

1. Fully functional web application
2. Docker compose setup for local development
3. Kubernetes manifests for production deployment
4. API documentation (OpenAPI/Swagger)
5. User guide and technical documentation
6. ML model documentation with performance metrics
7. Database schema documentation
8. Testing suite (unit, integration, E2E)
9. CI/CD pipeline configuration
10. Monitoring and alerting setup (Prometheus, Grafana)

SYSTEM ROLE & QUALITY BAR
You are an elite financial systems architect, quant researcher, and full-stack engineer who has built internal platforms for top hedge funds, banks, prop shops, and systematic trading firms.

You operate at Bloomberg / Citadel / Two Sigma / Renaissance quality levels.

This is not a demo, not a portfolio toy, not a dashboard. This must feel like a real financial terminal built for daily professional use.

If any design, model, or engineering choice feels sub-institutional, you must elevate it.

🎯 MISSION

Build a personal Bloomberg-terminal-level financial intelligence platform with a modern, younger aesthetic, while preserving the information density, speed, and seriousness of Bloomberg.

The terminal must unify:

Finance

Economics

Trading

Quantitative research

Risk

AI / ML / RL / DL / LLMs

APIs and real-time data

This is for my personal daily use as a serious quant / trader / researcher.

🧱 CORE REQUIREMENTS (NON-NEGOTIABLE)
1️⃣ ARCHITECTURE

Modular, production-grade system architecture

Clear separation between:

Data ingestion

Data storage

Analytics & math engine

Modeling & strategy layer

AI agents

UI / visualization

Async, event-driven where appropriate

Designed for extensibility, performance, and correctness

Everything API-first

2️⃣ DATA (INSTITUTIONAL-LEVEL)

Support deep historical + live data for:

Equities, ETFs, Options, Futures, FX, Rates, Crypto

Macro (CPI, GDP, NFP, yields, curves, spreads)

Corporate fundamentals & filings

News & text data

Alternative / sentiment data

Data standards:

Point-in-time correctness

Survivorship-bias free datasets

Versioned datasets

Reproducible research states

Unified schemas across asset classes

Cached hot data + cold historical storage

3️⃣ QUANT / MATH / ANALYTICS ENGINE

This must support real quant workflows, including:

Time-series analysis

Cross-sectional analysis

Factor modeling

Signal research

Risk modeling:

Volatility surfaces

Correlation regimes

Drawdowns

Stress testing

Portfolio construction & optimization

Backtesting engine with:

Slippage

Transaction costs

Latency

Market impact

Survivorship bias protection

Statistical rigor:

Hypothesis testing

Regime detection

Overfitting controls

Walk-forward validation

No simplified or “educational” assumptions.

4️⃣ AI / ML / RL / DL / LLMs (FIRST-CLASS)

AI is core infrastructure, not a bolt-on.

Requirements:

ML pipelines for prediction, classification, clustering

Deep learning for time-series & sequence modeling

Reinforcement learning for strategy optimization

LLM agents that can:

Query internal datasets

Run analysis

Generate charts

Explain models

Summarize markets

Assist research workflows

Models must be trained using:

The highest-quality historical data available

Advanced math, statistics, and data science techniques

Industry-grade feature engineering

Architecture must allow easy model swapping and experimentation

5️⃣ VISUALIZATION (MANDATORY)

All charts and graphs must be built with D3.js

No basic plotting libraries

Visuals must be:

High-performance

Interactive

Dense with information

Drill-down capable

Supported visuals:

Time-series

Candlesticks

Heatmaps

Correlation matrices

Volatility surfaces

Factor exposures

Risk dashboards

Regime shifts

6️⃣ UI / UX (BLOOMBERG-INSPIRED, MODERNIZED)

The interface must:

Look literally like a Bloomberg terminal, but:

More modern

Younger

Cleaner typography

Subtle modern color accents

Prioritize:

Speed

Information density

Keyboard-first interaction

Features:

Terminal-style command bar (Bloomberg-like functions)

Multi-panel layout

Persistent workspaces

Real-time updates

Minimal mouse dependence

Aesthetic: professional, sharp, serious — not flashy

7️⃣ ENGINEERING QUALITY BAR

Strong typing

Config-driven design

Logging & observability

Testing for core logic

Clean repo structure

Internal-style documentation

No magic numbers

No shortcuts

No “toy” abstractions

📦 DELIVERABLES (STEP-BY-STEP)

You must deliver this in clear, structured phases:

High-level architecture + diagrams

Tech stack justification

Data ingestion & storage foundation

Quant & analytics engine

AI / ML / RL integration

Terminal-style UI with D3.js visuals

Example real workflows:

Daily macro snapshot

Single-asset deep dive

Factor research

Strategy backtest

Portfolio risk analysis

AI-assisted research loop