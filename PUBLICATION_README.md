# Quantitative Financial Platform - Publication Ready

**A comprehensive, institutional-grade platform for quantitative analysis, machine learning, deep learning, reinforcement learning, and automated trading.**

## 🎯 Overview

This platform represents a complete, production-ready system for:
- **Fundamental Analysis**: DCF models, comparable analysis, financial statement analysis
- **Quantitative Analysis**: Portfolio optimization, risk management, options pricing
- **Machine Learning**: Time series forecasting, regime detection, anomaly detection
- **Deep Learning**: LSTM networks for price prediction
- **Reinforcement Learning**: Trading agents using DQN, PPO, and other RL algorithms
- **Transformer Models**: GPT-style models for financial text analysis, sentiment analysis
- **Political/Economic Analysis**: Geopolitical risk, policy impact, central bank analysis
- **Automated Trading**: End-to-end automation with ML signals, risk management, execution
- **Advanced Visualizations**: Plotly and D3.js interactive charts
- **REST APIs**: Comprehensive FastAPI-based API with 30+ endpoints
- **Real-time Streaming**: WebSocket support for live data

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install API dependencies (if running API server)
pip install -r requirements-api.txt
```

### Launch Dashboard

```bash
python start.py
# Dashboard available at http://localhost:8050
```

### Start API Server

```bash
python api/main.py
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Run Validation

```bash
python validate_publication_ready.py
```

## 📊 Key Features

### 1. Machine Learning & AI

#### Time Series Forecasting
```python
from models.ml.forecasting import TimeSeriesForecaster

forecaster = TimeSeriesForecaster(model_type='random_forest')
forecaster.fit(prices, n_lags=20)
forecast = forecaster.predict(prices, n_periods=30)
```

#### Deep Learning (LSTM)
```python
from models.ml.advanced_trading import LSTMPredictor

lstm = LSTMPredictor(lookback_window=20, hidden_units=64)
lstm.train(data, epochs=50, batch_size=32)
prediction = lstm.predict(data.tail(50))
```

#### Reinforcement Learning
```python
from models.ml.advanced_trading import RLReadyEnvironment
from stable_baselines3 import DQN

env = RLReadyEnvironment(df, initial_capital=100000)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

#### Transformer Models (Financial Text Analysis)
```python
from models.ml.transformer_models import FinancialSentimentAnalyzer, EarningsCallAnalyzer

# Sentiment analysis
analyzer = FinancialSentimentAnalyzer()
sentiment = analyzer.analyze_text("Company reports strong earnings growth")

# Earnings call analysis
call_analyzer = EarningsCallAnalyzer()
analysis = call_analyzer.analyze_transcript(transcript_text)
```

### 2. Political & Economic Analysis

#### Geopolitical Risk
```python
from models.macro.geopolitical_risk import GeopoliticalRiskAnalyzer

analyzer = GeopoliticalRiskAnalyzer()
risk_assessment = analyzer.assess_risk_level('trade_tensions', severity=0.7)
current_risks = analyzer.track_key_risks()
```

#### Policy Impact Analysis
```python
from models.macro.geopolitical_risk import PolicyImpactAssessor

assessor = PolicyImpactAssessor()
impact = assessor.assess_policy_impact('monetary_policy', 'tightening', 'medium_term')
```

### 3. Advanced Visualizations

#### Plotly Charts
```python
from core.visualizations import ChartBuilder

fig = ChartBuilder.candlestick_chart(df, title="Price Action")
fig.show()
```

#### D3.js Visualizations
```python
from core.advanced_viz.d3_visualizations import D3Visualizations

d3_viz = D3Visualizations()
html = d3_viz.candlestick_chart_d3(df, width=1200, height=600)
d3_viz.save_html(html, 'chart.html')
```

### 4. Automated Trading

#### ML Pipeline
```python
from automation.ml_pipeline import MLPipeline

pipeline = MLPipeline()
results = pipeline.train_lstm_model('AAPL', epochs=50)
```

#### Trading Automation
```python
from automation.trading_automation import TradingAutomation

trading = TradingAutomation(trading_enabled=False, initial_capital=100000)
signal = trading.generate_ml_signal('AAPL')
```

#### Orchestrator (Full Automation)
```python
from automation.orchestrator import AutomationOrchestrator

orchestrator = AutomationOrchestrator()
orchestrator.start_all()  # Starts data pipeline, ML training, monitoring
```

### 5. API Endpoints

#### Models API
- `GET /api/v1/models/` - List all models
- `POST /api/v1/models/register` - Register new model
- `POST /api/v1/models/{model_id}/train` - Train model
- `GET /api/v1/models/{model_id}/predict` - Get predictions

#### Predictions API
- `POST /api/v1/predictions/single` - Single prediction
- `POST /api/v1/predictions/batch` - Batch predictions

#### Backtesting API
- `POST /api/v1/backtest/run` - Run backtest
- `GET /api/v1/backtest/{backtest_id}/results` - Get results

#### WebSocket API
- `WS /api/v1/ws/stream` - Real-time data streaming

## 📁 Project Structure

```
Models/
├── api/                    # FastAPI REST API
│   ├── main.py            # API server
│   ├── models_api.py      # Model management
│   ├── predictions_api.py # Predictions
│   ├── backtesting_api.py # Backtesting
│   └── ...
├── automation/            # Automation pipelines
│   ├── orchestrator.py   # Central orchestrator
│   ├── ml_pipeline.py     # ML training automation
│   ├── data_pipeline.py  # Data pipeline
│   └── trading_automation.py
├── core/                  # Core utilities
│   ├── data_fetcher.py   # Data fetching
│   ├── backtesting.py    # Backtesting engine
│   ├── paper_trading.py  # Paper trading
│   ├── dashboard.py      # Dashboard
│   └── advanced_viz/     # Visualizations
│       ├── d3_visualizations.py  # D3.js charts
│       └── ...
├── models/                # Financial models
│   ├── ml/               # ML/DL/RL models
│   │   ├── forecasting.py
│   │   ├── advanced_trading.py  # LSTM, RL
│   │   └── transformer_models.py  # GPT-style models
│   ├── macro/            # Macro/political models
│   │   ├── geopolitical_risk.py
│   │   └── ...
│   ├── portfolio/        # Portfolio optimization
│   ├── risk/             # Risk models
│   └── ...
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Dependencies
└── requirements-api.txt  # API dependencies
```

## 🔧 Dependencies

### Core
- Python 3.8+
- NumPy, Pandas, SciPy
- Scikit-learn

### Machine Learning
- TensorFlow 2.13+ (for LSTM)
- PyTorch 2.0+ (for transformers)
- Transformers (HuggingFace)
- Stable-Baselines3 (for RL)

### Visualization
- Plotly 5.17+
- Dash 2.14+
- D3.js (via js2py bridge)
- Matplotlib, Seaborn

### API
- FastAPI 0.104+
- Uvicorn
- WebSockets

### Financial Data
- yfinance
- FRED API
- Alpha Vantage

## 📈 Use Cases

### 1. Research & Analysis
- Academic research
- Market analysis
- Economic research
- Strategy development

### 2. Automated Trading
- Strategy backtesting
- ML signal generation
- Paper trading
- Live trading (with broker integration)

### 3. Risk Management
- VaR/CVaR calculations
- Stress testing
- Portfolio optimization
- Scenario analysis

### 4. Reporting
- Investor reports
- Performance analysis
- Risk disclosure
- Automated report generation

## 🎨 Visualizations

### Plotly Charts
- Candlestick charts
- Correlation heatmaps
- Efficient frontier
- Time series plots
- Distribution plots

### D3.js Charts
- Interactive candlestick charts
- Force-directed networks
- Sankey diagrams
- Treemaps
- Custom D3 visualizations

## 🤖 Automation

### Data Pipeline
- Automated data fetching
- Data quality monitoring
- Scheduled updates

### ML Pipeline
- Automated model training
- Model versioning
- Performance tracking

### Trading Automation
- Signal generation
- Risk management
- Order execution
- Performance monitoring

## 📚 Documentation

- `README.md` - Main documentation
- `PROJECT_OVERVIEW.md` - Architecture overview
- `API_DOCUMENTATION.md` - API reference
- `ADVANCED_FEATURES.md` - Advanced features guide
- `DEPLOYMENT.md` - Deployment guide
- `notebooks/` - Interactive examples

## ✅ Validation

Run comprehensive validation:

```bash
python validate_publication_ready.py
```

This checks:
- All dependencies
- Core modules
- Financial models
- API endpoints
- Visualizations
- Automation components

## 🚢 Deployment

### Docker

```bash
docker-compose up -d
```

### Production

```bash
# Using Gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Or Uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🔐 Security Notes

- Configure CORS for production
- Use environment variables for API keys
- Implement authentication for API endpoints
- Secure WebSocket connections
- Validate all inputs

## 📝 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]

## 📧 Support

[Support Information]

---

**Status**: ✅ **PRODUCTION READY**

All components validated and operational. Platform ready for publication and deployment.
