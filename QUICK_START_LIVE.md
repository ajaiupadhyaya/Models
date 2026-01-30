# 🚀 TRADING SYSTEM - QUICK START GUIDE

## ✅ System Status: OPERATIONAL

**Validation Results:**
- ✓ Python 3.11 environment configured
- ✓ All dependencies installed (pandas, numpy, sklearn, TensorFlow, FastAPI, OpenAI)
- ✓ Data fetching works (FRED macro, stock data)
- ✓ ML models trained and predicting (Ensemble, LSTM on GPU)
- ✓ AI analysis service active (OpenAI integration verified)
- ✓ Alpaca trading integration ready
- ✓ API endpoints configured
- ✓ React terminal frontend (Vite + D3)

---

## 🎯 GETTING STARTED

### 1. Activate the Virtual Environment

```bash
cd <project_root>   # e.g. /Users/ajaiupadhyaya/Documents/Models
source venv/bin/activate
# or use the venv Python directly:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# or use the provided script:
bash start-api.sh
```

**API Documentation**: http://localhost:8000/docs

**Important:** The frontend proxies `/api` to `http://localhost:8000`. Start the API first, then the frontend. For automation and orchestrator (RL, scheduled jobs), ensure `schedule` is installed: `pip install schedule` (included in `requirements.txt`).

### 3. Start the Bloomberg-Style Terminal (React Frontend)

In a **new terminal**, from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown (typically **http://localhost:5173**) in your browser. You will see the terminal with Market Overview, Primary Instrument chart, Portfolio panel, and AI Assistant.

---

## 🤖 NEW AI ANALYSIS FEATURES

### Market Summary
```bash
curl "http://localhost:8000/api/v1/ai/market-summary?symbols=AAPL,MSFT,GOOGL"
```

### Stock Deep Dive Analysis
```bash
curl "http://localhost:8000/api/v1/ai/stock-analysis/AAPL?include_prediction=true"
```

### Trading Recommendations
```bash
curl -X POST "http://localhost:8000/api/v1/ai/trading-insight?symbol=AAPL&current_price=150.00&predicted_price=155.00&confidence=0.75"
```

### Sentiment Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/ai/sentiment?text=Market+is+bullish+strong+fundamentals"
```

### Explain Financial Metrics
```bash
curl -X POST "http://localhost:8000/api/v1/ai/explain-metrics" \
  -H "Content-Type: application/json" \
  -d '{"Sharpe Ratio": 1.5, "Max Drawdown": -0.15, "Sortino Ratio": 2.1}'
```

---

## ⚡ AUTOMATED TRADING ORCHESTRATION

### Run Full Automated Trading Loop

```bash
# Dry run (predictions only, no actual trades)
curl -X POST "http://localhost:8000/api/v1/automation/predict-and-trade?symbols=AAPL,MSFT,GOOGL&use_lstm=true&execute_trades=false"

# Live trading on Alpaca (use with caution!)
curl -X POST "http://localhost:8000/api/v1/automation/predict-and-trade?symbols=AAPL,MSFT,GOOGL&use_lstm=true&execute_trades=true"
```

This endpoint:
1. ✓ Fetches macro data (FRED: unemployment, GDP, CPI)
2. ✓ Runs ensemble + LSTM predictions
3. ✓ Generates AI trading recommendations
4. ✓ Executes trades on Alpaca (if enabled)
5. ✓ Creates OpenAI narrative summary

### Check Automation Status
```bash
curl "http://localhost:8000/api/v1/automation/status"
```

### Get Current Positions
```bash
curl "http://localhost:8000/api/v1/automation/positions"
```

### Get Account Info
```bash
curl "http://localhost:8000/api/v1/automation/account"
```

---

## 📊 API ENDPOINTS

### Health & Status
- `GET /` — API status
- `GET /health` — Health check with models count
- `GET /info` — Detailed system info

### AI Analysis (NEW)
- `GET /api/v1/ai/market-summary` — Multi-stock AI analysis
- `GET /api/v1/ai/stock-analysis/{symbol}` — Deep dive with ML predictions
- `POST /api/v1/ai/trading-insight` — AI trading recommendations
- `POST /api/v1/ai/sentiment` — Sentiment analysis
- `POST /api/v1/ai/explain-metrics` — Financial metric explanations

### Automation (NEW)
- `POST /api/v1/automation/predict-and-trade` — Full trading orchestration
- `GET /api/v1/automation/status` — Automation system status
- `GET /api/v1/automation/positions` — Current Alpaca positions
- `GET /api/v1/automation/account` — Alpaca account info

### Predictions
- `POST /api/v1/predictions/predict` — Get next-period predictions
- `POST /api/v1/predictions/ensemble` — Ensemble model predictions
- `POST /api/v1/predictions/lstm` — LSTM model predictions

### Company Analysis
- `GET /api/v1/company/analysis/{symbol}` — DCF valuation + risk metrics

### Paper Trading
- `GET /api/v1/paper-trading/account` — Account status
- `POST /api/v1/paper-trading/place-order` — Place a trade order
- `GET /api/v1/paper-trading/positions` — View current positions

### Backtesting
- `POST /api/v1/backtest/run` — Backtest a strategy

### Investor Reports
- `GET /api/v1/investor-reports/{symbol}` — Generate investor report

---

## 🔧 CONFIGURATION

### Environment Variables (.env)

Required keys:
```
FRED_API_KEY=your_fred_key
ALPHA_VANTAGE_API_KEY=your_av_key
OPENAI_API_KEY=your_openai_key
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

Optional:
```
ALPACA_API_BASE=https://paper-api.alpaca.markets  # or live trading URL
WEBSOCKET_ENABLED=true
AI_ANALYSIS_ENABLED=true
```

---

## 🧪 VALIDATION

Run the complete system validation:

```bash
python automation/validate_live.py
```

Expected output:
```
✓ PASS     Environment
✓ PASS     Dependencies
✓ PASS     API Keys
✓ PASS     Data Fetching
✓ PASS     ML Models
✓ PASS     AI Analysis
✓ PASS     Alpaca Integration
✓ PASS     API Endpoints
✓ PASS     (Frontend: build with cd frontend && npm run build)

Result: 9/9 checks passed
✅ ALL SYSTEMS OPERATIONAL - Ready to trade!
```

---

## 🎓 EXAMPLE: END-TO-END TRADING FLOW

### 1. Fetch Market Data
```python
from core.data_fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.get_stock_data("AAPL", period="3mo")
unemployment = fetcher.get_economic_indicator("UNRATE")
```

### 2. Generate Predictions
```python
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor

ensemble = EnsemblePredictor(lookback_window=20)
ensemble.train(df)
pred_ensemble = ensemble.predict(df)

lstm = LSTMPredictor(lookback_window=20, hidden_units=16)
lstm.train(df, epochs=5)
pred_lstm = lstm.predict(df)
```

### 3. Get AI Analysis
```python
from core.ai_analysis import get_ai_service

ai = get_ai_service()
insight = ai.generate_trading_insight(
    symbol="AAPL",
    current_price=150.0,
    prediction=155.0,
    confidence=0.75,
    market_context="Unemployment low, GDP strong"
)
# Returns: {"action": "BUY", "reasoning": "...", "risk_level": "medium", ...}
```

### 4. Execute Trade
```python
from core.paper_trading import AlpacaAdapter

alpaca = AlpacaAdapter(api_key, api_secret)
if insight["action"] == "BUY":
    order = alpaca.submit_order(
        symbol="AAPL",
        qty=10,
        side="buy",
        type="market"
    )
```

### 5. Generate Report
```python
from core.investor_reports import InvestorReportGenerator

reporter = InvestorReportGenerator()
report = reporter.generate_executive_summary(
    symbol="AAPL",
    metrics={"Sharpe": 1.5, "MaxDD": -0.12},
    ai_enabled=True
)
```

---

## 🚨 TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'pandas'"
```bash
# Ensure you're using the right Python (from project root with venv activated):
python -c "import pandas; print(pandas.__version__)"

# If not working, reinstall dependencies:
pip install -r requirements.txt
# For full API/ML stack including LSTM: pip install -r requirements-api.txt
```

### "TensorFlow errors on M2 Mac"
```bash
# Optional: use TensorFlow macOS build for Apple Silicon:
pip install tensorflow-macos tensorflow-metal
```

### "OpenAI API rate limit"
- Check rate limits: https://platform.openai.com/account/rate-limits
- Add delays between requests
- Upgrade API plan if needed

### "Alpaca authentication failed"
- Verify ALPACA_API_KEY and ALPACA_API_SECRET in .env
- Test with curl:
  ```bash
  curl -H "APCA-API-KEY-ID: your_key" https://paper-api.alpaca.markets/v2/account
  ```

---

## 📈 PERFORMANCE TIPS

1. **Use LSTM for long predictions** (>5 days) — better at capturing trends
2. **Use Ensemble for short-term** (<3 days) — faster, lighter model
3. **Cache data** — data_fetcher uses 5-minute TTL cache automatically
4. **Batch predictions** — send multiple symbols in one request
5. **Monitor GPU** — LSTM automatically uses M2 GPU if available

---

## 🔐 SECURITY NOTES

- ✓ Never commit .env file
- ✓ Rotate API keys regularly
- ✓ Use paper trading first before live
- ✓ Alpaca paper trading uses sand box — safe to test
- ✓ AI API calls are logged; monitor usage

---

## 📚 NEXT STEPS

1. **Customize prediction thresholds** — Modify BUY/SELL signals in automation_api.py
2. **Add more stock symbols** — Pass comma-separated list to predict-and-trade
3. **Implement risk management** — Add stop-loss and take-profit limits
4. **Create scheduled tasks** — Use APScheduler to run predict-and-trade hourly
5. **Monitor P&L** — Build dashboard widget for position tracking

---

**Questions?** Check the API docs at `http://localhost:8000/docs` once the server is running.

**Last Updated**: 2026-01-21  
**Status**: ✅ PRODUCTION READY
