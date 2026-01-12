"""
Example configuration file for financial models.
Copy this to config.py and customize for your needs.
"""

# API Keys (set in .env file instead for security)
FRED_API_KEY = None  # Get from: https://fred.stlouisfed.org/docs/api/api_key.html
ALPHA_VANTAGE_API_KEY = None  # Get from: https://www.alphavantage.co/support/#api-key

# Default Parameters
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
DEFAULT_WACC = 0.10  # 10% weighted average cost of capital
DEFAULT_TERMINAL_GROWTH = 0.03  # 3% terminal growth rate

# Trading Parameters
DEFAULT_LOOKBACK_PERIOD = 20  # Days
DEFAULT_HOLDING_PERIOD = 5  # Days

# Risk Parameters
DEFAULT_CONFIDENCE_LEVEL = 0.05  # 95% VaR
DEFAULT_MONTE_CARLO_SIMULATIONS = 10000

# Visualization
CHART_THEME = "plotly_dark"  # Options: plotly, plotly_white, plotly_dark, ggplot2, seaborn, simple_white, none
CHART_HEIGHT = 600
CHART_WIDTH = 800

# Data Storage
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
REPORTS_DIR = "reports"
PRESENTATIONS_DIR = "presentations"
