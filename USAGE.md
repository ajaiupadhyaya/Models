# Usage Guide

## Installation

1. **Set up Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys (optional but recommended):**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - FRED API: https://fred.stlouisfed.org/docs/api/api_key.html
     - Alpha Vantage: https://www.alphavantage.co/support/#api-key

## Quick Start

### Test Installation
```bash
python quick_start.py
```

### Run Jupyter Lab
```bash
jupyter lab
```

Then open notebooks in the `notebooks/` directory.

## Common Use Cases

### 1. Fetch Stock Data
```python
from core.data_fetcher import DataFetcher

fetcher = DataFetcher()
stock_data = fetcher.get_stock_data('AAPL', period='1y')
```

### 2. Fetch Economic Data
```python
unemployment = fetcher.get_unemployment_rate()
gdp = fetcher.get_gdp()
cpi = fetcher.get_cpi()
```

### 3. DCF Valuation
```python
from models.valuation.dcf_model import DCFModel

dcf = DCFModel(
    free_cash_flows=[100, 120, 140, 160, 180],
    terminal_growth_rate=0.03,
    wacc=0.10
)

ev = dcf.calculate_enterprise_value()
```

### 4. Options Pricing
```python
from models.options.black_scholes import BlackScholes

call_price = BlackScholes.call_price(
    S=100,  # Stock price
    K=100,  # Strike price
    T=0.25,  # Time to expiration (years)
    r=0.05,  # Risk-free rate
    sigma=0.20  # Volatility
)

greeks = BlackScholes.get_all_greeks(100, 100, 0.25, 0.05, 0.20)
```

### 5. Portfolio Optimization
```python
from models.portfolio.optimization import optimize_portfolio_from_returns

# Assuming you have a returns DataFrame
result = optimize_portfolio_from_returns(
    returns_df,
    method='sharpe',  # or 'min_vol', 'risk_parity'
    risk_free_rate=0.02
)

print(result['weights'])
```

### 6. Risk Analysis
```python
from models.risk.var_cvar import VaRModel, CVaRModel

var = VaRModel.calculate_var(returns, method='historical')
cvar = CVaRModel.calculate_cvar(returns, method='historical')
```

### 7. Trading Strategies
```python
from models.trading.strategies import MomentumStrategy

strategy = MomentumStrategy(lookback_period=20)
signals = strategy.generate_signals(prices)
results = strategy.backtest(prices, initial_capital=100000)
```

### 8. Create Visualizations
```python
from core.visualizations import ChartBuilder

# Candlestick chart
fig = ChartBuilder.candlestick_chart(stock_data, title="Stock Price")
fig.show()

# Risk-return scatter
fig = ChartBuilder.risk_return_scatter(returns_df)
fig.show()

# Efficient frontier
fig = ChartBuilder.efficient_frontier(expected_returns, cov_matrix)
fig.show()
```

### 9. Generate Reports
```python
from templates.reports.report_generator import ReportGenerator

generator = ReportGenerator()
data = {
    'analyst_name': 'Your Name',
    'project_name': 'Analysis Project',
    'executive_summary': '...',
    # ... other fields
}
generator.generate_report(data, 'outputs/report.md')
```

### 10. Create Presentations
```python
from templates.presentations.presentation_generator import PresentationGenerator

gen = PresentationGenerator()
gen.add_title_slide('Financial Analysis', 'Date: 2024-01-01')
gen.add_content_slide('Key Findings', ['Finding 1', 'Finding 2'])
gen.save('outputs/presentation.pptx')
```

## Model Templates Available

### Valuation Models
- **DCF Model** (`models/valuation/dcf_model.py`)
  - Discounted cash flow valuation
  - Sensitivity analysis
  - Terminal value calculation

### Options & Derivatives
- **Black-Scholes** (`models/options/black_scholes.py`)
  - European call/put pricing
  - Greeks calculation
  - Implied volatility

### Portfolio Management
- **Mean-Variance Optimization** (`models/portfolio/optimization.py`)
  - Maximum Sharpe ratio
  - Minimum volatility
  - Target return optimization
- **Risk Parity** (`models/portfolio/optimization.py`)
  - Equal risk contribution

### Risk Management
- **VaR/CVaR** (`models/risk/var_cvar.py`)
  - Historical simulation
  - Parametric method
  - Monte Carlo simulation
- **Stress Testing** (`models/risk/var_cvar.py`)
  - Scenario analysis
  - Historical stress periods

### Macro Economics
- **GDP Forecasting** (`models/macro/economic_models.py`)
- **Yield Curve Analysis** (`models/macro/economic_models.py`)
- **Economic Indicators** (`models/macro/economic_models.py`)

### Trading Strategies
- **Momentum** (`models/trading/strategies.py`)
- **Mean Reversion** (`models/trading/strategies.py`)
- **Pairs Trading** (`models/trading/strategies.py`)
- **Factor Investing** (`models/trading/strategies.py`)

## Tips

1. **API Keys**: While optional, API keys enable real-time economic data from FRED and additional features from Alpha Vantage.

2. **Data Caching**: Consider caching frequently used data to avoid repeated API calls.

3. **Notebooks**: Start with the example notebooks in `notebooks/` to learn the framework.

4. **Customization**: All models are designed to be easily customizable. Modify parameters and extend classes as needed.

5. **Visualization**: Charts are interactive by default. Use `fig.show()` in Jupyter or save with `fig.write_html()`.

## Troubleshooting

### Import Errors
- Ensure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.8+

### API Errors
- Check that API keys are set in `.env` file
- Verify API keys are valid
- Some APIs have rate limits - add delays if needed

### Data Fetching Issues
- Check internet connection
- Verify ticker symbols are correct
- Some data sources may require authentication

## Support

For issues or questions:
1. Check the example notebooks
2. Review the README.md
3. Examine the code documentation in each module
