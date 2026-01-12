# Financial Models Workspace

A comprehensive financial modeling framework for quantitative analysis, trading strategies, and economic research. Built with institutional-grade tools and practices.

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
- **Real-time Data**: Automatic fetching from FRED, Alpha Vantage, Yahoo Finance with intelligent caching
- **Publication-Quality Visualizations**: NYT/Bloomberg/FT-inspired interactive charts
- **Interactive Dashboards**: Real-time web-based analysis dashboards
- **Institutional Models**: DCF, Black-Scholes, Portfolio Optimization, Risk Models
- **Macro Analysis**: Economic indicators, unemployment, GDP, inflation, yield curves
- **Report Generation**: Automated report and slide deck creation
- **Trading Strategies**: Professional backtesting with transaction costs and slippage

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

### Interactive Dashboard
```bash
python run_dashboard.py
# Opens at http://localhost:8050
```

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

## 📝 License

For personal and professional use.
