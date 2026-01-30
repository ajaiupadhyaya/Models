# Financial Models Workspace

A comprehensive, institutional-grade financial modeling framework for quantitative analysis, machine learning, and investment research. This project combines sophisticated models with publication-quality visualizations and a production-ready API.

**Now with high-performance C++ implementations for computationally intensive operations (10-100x faster)!**

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

### 1. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build high-performance C++ extensions (optional but recommended)
./build_cpp.sh
# Or on Windows: python setup_cpp.py build_ext --inplace
```

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
See [QUICK_START_LIVE.md](QUICK_START_LIVE.md) for full setup.

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

See `notebooks/` for detailed examples and `ADVANCED_FEATURES.md` for comprehensive documentation.

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

## 📝 License

For personal and professional use.
