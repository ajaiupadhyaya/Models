# Python 3.14 Compatibility Notes

## Status

The framework has been tested and works with **Python 3.14.2**. However, some packages are not yet available for Python 3.14 and have been made optional.

## Installed Packages

✅ **Core packages successfully installed:**
- numpy, pandas, scipy, statsmodels
- yfinance, plotly, dash, matplotlib, seaborn
- jupyter, jupyterlab, ipywidgets
- requests, python-dotenv, jinja2
- All visualization and data fetching libraries

## Optional Packages (Not Available for Python 3.14 Yet)

The following packages are commented out in `requirements.txt` because they don't have Python 3.14 builds yet:

- **cvxpy**: Advanced convex optimization (portfolio optimization works without it using scipy)
- **pypfopt**: Portfolio optimization library (we have our own implementation in `models/portfolio/optimization.py`)
- **quantlib**: Quantitative finance library (most models work without it)

## Workarounds

### Portfolio Optimization
Our custom implementation in `models/portfolio/optimization.py` provides:
- Mean-Variance Optimization
- Risk Parity
- Efficient Frontier calculation

These work without `cvxpy` or `pypfopt` using scipy optimization.

### If You Need These Packages

If you specifically need `cvxpy` or `pypfopt`, you have two options:

1. **Use Python 3.12 or 3.13** (recommended for full compatibility):
   ```bash
   # Install Python 3.12 or 3.13
   brew install python@3.12
   
   # Create venv with specific Python version
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Wait for package updates**: These packages will likely add Python 3.14 support in future releases.

## Current Functionality

✅ **All core features work:**
- Data fetching (FRED, Alpha Vantage, Yahoo Finance)
- All visualization types
- DCF valuation
- Black-Scholes options pricing
- Portfolio optimization (custom implementation)
- Risk models (VaR, CVaR)
- Trading strategies
- Machine learning models
- Interactive dashboard
- Report generation

## Testing

Run the quick start test:
```bash
source venv/bin/activate
python quick_start.py
```

All tests should pass (except for optional packages).

## Recommendation

For production use or if you need all packages, consider using **Python 3.12 or 3.13** which have full package support. For development and most use cases, Python 3.14 works perfectly with the installed packages.
