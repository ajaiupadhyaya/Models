# Getting Started Guide

## 🎯 Welcome!

This is your **one-stop shop for all things finance/economic models** - from basics to high-level quant analysis. Everything is designed to work together seamlessly.

## 🚀 First Time Setup

### Step 1: Launch the Unified Launcher

The easiest way to get started:

```bash
cd /Users/ajaiupadhyaya/Documents/Models
python launch.py
```

The launcher will:
- ✅ Automatically check if dependencies are installed
- ✅ Offer to install missing packages
- ✅ Validate your environment
- ✅ Provide an interactive menu to explore everything

### Step 2: Install Dependencies (if needed)

If the launcher detects missing dependencies, it will offer to install them. You can also install manually:

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys (Optional)

Many features work without API keys, but for full functionality:

1. Create a `.env` file in the project root
2. Add your API keys:

```env
FRED_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

**Where to get keys:**
- **FRED** (free): https://fred.stlouisfed.org/docs/api/api_key.html
- **Alpha Vantage** (free): https://www.alphavantage.co/support/#api-key

## 🎮 What Can You Do?

### 1. Interactive Dashboard
Launch a web-based dashboard to explore data and models visually:
```bash
python launch.py
# Select option 1
```

### 2. API Server
Start a production-ready API server with all endpoints:
```bash
python launch.py
# Select option 2
# Then visit http://localhost:8000/docs
```

### 3. Jupyter Notebooks
Explore interactive notebooks with examples:
```bash
python launch.py
# Select option 3
```

### 4. Run Tests
Validate everything is working:
```bash
python launch.py
# Select option 4 (Quick Start Test)
# Or option 5 (Comprehensive Audit)
```

## 📚 Exploring the Project

### Core Modules (`core/`)
- **data_fetcher.py**: Fetch data from FRED, Alpha Vantage, Yahoo Finance
- **visualizations.py**: Create publication-quality charts
- **dashboard.py**: Interactive web dashboard
- **backtesting.py**: Test trading strategies
- **paper_trading.py**: Simulate trading with real data

### Financial Models (`models/`)
- **valuation/**: DCF models, comparable analysis
- **options/**: Black-Scholes, Greeks
- **portfolio/**: Portfolio optimization
- **risk/**: VaR, CVaR, stress testing
- **macro/**: Economic indicators, yield curves
- **trading/**: Trading strategies, backtesting
- **ml/**: Machine learning forecasting

### API (`api/`)
- RESTful API endpoints for all functionality
- WebSocket support for real-time data
- Model training and prediction endpoints
- Backtesting and paper trading APIs

### Notebooks (`notebooks/`)
- **01_getting_started.ipynb**: Introduction
- **02_dcf_valuation.ipynb**: DCF examples
- **03_fundamental_analysis.ipynb**: Company analysis
- **04_macro_sentiment_analysis.ipynb**: Macro analysis
- **05_advanced_visualizations.ipynb**: Charting examples
- **06_ml_forecasting.ipynb**: ML models
- And more...

## 🔍 Quick Examples

### Fetch Stock Data
```python
from core.data_fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.get_stock_data('AAPL', period='1y')
print(data.head())
```

### Calculate DCF Valuation
```python
from models.valuation.dcf_model import DCFModel

dcf = DCFModel(
    free_cash_flows=[100, 120, 140, 160, 180],
    terminal_growth_rate=0.03,
    wacc=0.10
)
enterprise_value = dcf.calculate_enterprise_value()
print(f"Enterprise Value: ${enterprise_value:,.2f}")
```

### Price an Option
```python
from models.options.black_scholes import BlackScholes

call_price = BlackScholes.call_price(
    S=100,  # Stock price
    K=100,  # Strike price
    T=0.25, # Time to expiration (years)
    r=0.05, # Risk-free rate
    sigma=0.20  # Volatility
)
print(f"Call Option Price: ${call_price:.2f}")
```

### Optimize Portfolio
```python
from models.portfolio.optimization import MeanVarianceOptimizer
import pandas as pd

# Get stock data
from core.data_fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.get_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])

# Optimize
optimizer = MeanVarianceOptimizer(data['Close'])
weights = optimizer.optimize(target_return=0.15)
print(weights)
```

## 🛠️ Troubleshooting

### Dependencies Not Installing
If you encounter issues:
```bash
# Try upgrading pip first
pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

### Import Errors
Run the validation script:
```bash
python validate_all.py
```

This will show exactly what's missing.

### API Keys Not Working
- Check that your `.env` file is in the project root
- Verify API keys are correct (no extra spaces)
- Some features work without API keys (Yahoo Finance data)

### Port Already in Use
If port 8000 or 8050 is already in use:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
lsof -ti:8050 | xargs kill -9
```

## 📖 Next Steps

1. **Explore Notebooks**: Start with `notebooks/01_getting_started.ipynb`
2. **Try the Dashboard**: Launch it and explore interactively
3. **Read Documentation**: Check `PROJECT_OVERVIEW.md` for architecture details
4. **API Documentation**: Visit http://localhost:8000/docs when API is running

## 🎯 Your End Goal: Automation

This project is designed to be automated:
- ✅ Train models via API
- ✅ Run backtests programmatically
- ✅ Generate reports automatically
- ✅ Paper trade with real-time data
- ✅ Monitor and alert on conditions

Everything is API-accessible and can be scripted!

## 💡 Tips

- **Start Simple**: Begin with the dashboard or notebooks
- **Use the Launcher**: `python launch.py` is your friend
- **Read the Docs**: Each module has docstrings
- **Experiment**: All models are designed to be tweaked
- **API Keys Optional**: Many features work without them

## 🆘 Need Help?

1. Run `python validate_all.py` to diagnose issues
2. Check `PROJECT_OVERVIEW.md` for architecture
3. Review `API_DOCUMENTATION.md` for API details
4. Look at notebook examples in `notebooks/`

---

**You're all set! Start exploring with `python launch.py`** 🚀
