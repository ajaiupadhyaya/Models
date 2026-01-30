# Bloomberg Terminal - Automated Trading Platform Guide

## 🎯 Overview

This is a fully automated, AI/ML/DL/RL-powered Bloomberg Terminal-like platform for quantitative trading and research. The system combines:

- **Advanced ML/DL Models**: Ensemble predictors, LSTM networks, and RL agents
- **Real-time Data Streaming**: WebSocket-based live market data
- **Automated Trading**: Continuous model training, signal generation, and trade execution
- **Modern UI**: Bloomberg Terminal-inspired multi-panel dashboard
- **Production-Ready**: Optimized for server deployment with monitoring and alerting

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd /path/to/Models

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# Install PyTorch (if needed)
# For CPU: pip install torch
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Data APIs
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Trading (optional - for paper/live trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_API_BASE=https://paper-api.alpaca.markets

# AI Analysis (optional)
OPENAI_API_KEY=your_openai_key
```

### 3. Start the Platform

**Option A: API Server Only (FastAPI)**
```bash
cd api
python main.py
# Or use uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Option B: Full Bloomberg-Style Web Terminal (Recommended)**
```bash
# 1. Start API server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. In another terminal, start the React + D3 UI
cd ../frontend
npm install
npm run dev
```
Opens at: http://localhost:5173

## 📊 Features

### 1. Automated Trading Orchestrator

The orchestrator coordinates all models and executes automated trading:

```python
from core.automated_trading_orchestrator import AutomatedTradingOrchestrator

# Initialize
orchestrator = AutomatedTradingOrchestrator(
    symbols=["AAPL", "MSFT", "GOOGL"],
    use_rl=True,
    use_lstm=True,
    use_ensemble=True,
    retrain_frequency="daily"
)

# Initialize models
orchestrator.initialize_models()

# Run one cycle
result = orchestrator.run_cycle(execute_trades=False)

# Start continuous trading
orchestrator.start_automated_trading(interval_minutes=60, execute=False)
```

### 2. Advanced RL Agents

Multiple RL algorithms available:

```python
from models.ml.rl_agents import DQNAgent, PPOAgent, StableBaselines3Wrapper
from models.ml.advanced_trading import RLReadyEnvironment

# Create environment
env = RLReadyEnvironment(df, initial_capital=100000)

# DQN Agent
dqn = DQNAgent(state_dim=len(env.reset()), action_dim=4)
# Train...
dqn.save("models/dqn_agent.pth")

# PPO Agent (via stable-baselines3)
ppo = StableBaselines3Wrapper(agent_type="PPO")
ppo.create_agent(env)
ppo.train(total_timesteps=10000)
ppo.save("models/ppo_agent")
```

### 3. Real-time Data Streaming

```python
from core.realtime_streaming import RealTimeDataStreamer

streamer = RealTimeDataStreamer(symbols=["AAPL", "MSFT"])

def handle_message(message):
    print(f"{message.symbol}: ${message.data['price']}")

streamer.subscribe(handle_message)
streamer.start()
```

### 4. Model Performance Monitoring

```python
from core.model_monitor import ModelPerformanceMonitor

monitor = ModelPerformanceMonitor()

# Record predictions
monitor.record_prediction("ensemble", "AAPL", prediction=150.0, actual=151.0)

# Check if retraining needed
should_retrain, reason = monitor.should_retrain("ensemble", "AAPL")
if should_retrain:
    print(f"Retrain needed: {reason}")
```

### 5. Alerting System

```python
from core.alerting_system import AlertingSystem, AlertSeverity

alerts = AlertingSystem()

# Check risk thresholds
alerts.check_risk_thresholds(
    portfolio_value=100000,
    daily_pnl=-5000,
    max_drawdown=-0.12,
    positions={"AAPL": 20000}
)

# Get alerts
critical_alerts = alerts.get_alerts(severity=AlertSeverity.CRITICAL)
```

## 🌐 API Endpoints

### Orchestrator API

- `POST /api/v1/orchestrator/initialize` - Initialize orchestrator
- `POST /api/v1/orchestrator/run-cycle` - Run one trading cycle
- `POST /api/v1/orchestrator/start-automated` - Start continuous trading
- `GET /api/v1/orchestrator/status` - Get orchestrator status
- `GET /api/v1/orchestrator/signals` - Get latest trading signals
- `GET /api/v1/orchestrator/trades` - Get trade history
- `POST /api/v1/orchestrator/retrain` - Manually retrain models

### WebSocket Streaming

- `ws://localhost:8000/api/v1/ws/prices/{symbol}` - Real-time price updates
- `ws://localhost:8000/api/v1/ws/predictions/{model}/{symbol}` - Real-time predictions
- `ws://localhost:8000/api/v1/ws/live` - General live feed

### Other APIs

See `API_DOCUMENTATION.md` for complete API reference.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              BLOOMBERG TERMINAL PLATFORM                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Data       │    │   ML/DL/RL   │    │   Trading    │ │
│  │   Layer      │───▶│   Models     │───▶│   Execution  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │          │
│         └────────────────────┼────────────────────┘          │
│                              │                              │
│                    ┌─────────▼─────────┐                   │
│                    │   Orchestrator     │                   │
│                    │  (Coordination)     │                   │
│                    └─────────┬─────────┘                   │
│                              │                              │
│         ┌────────────────────┼────────────────────┐         │
│         │                    │                    │         │
│  ┌──────▼──────┐    ┌───────▼──────┐   ┌───────▼──────┐  │
│  │   Real-time │    │   Monitoring │   │   Alerting   │  │
│  │   Streaming │    │   & Metrics  │   │   System     │  │
│  └─────────────┘    └──────────────┘   └──────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │   React + D3 Bloomberg Terminal Web UI               │ │
│  │  - Watchlist  - Charts  - Signals  - Portfolio       │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t bloomberg-terminal .

# Run container
docker run -d \
  -p 8000:8000 \
  -p 8050:8050 \
  -e FRED_API_KEY=your_key \
  -e ALPACA_API_KEY=your_key \
  bloomberg-terminal
```

### Server Deployment

1. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

2. **Set environment variables**:
```bash
export FRED_API_KEY=your_key
export ALPACA_API_KEY=your_key
# etc.
```

3. **Run with process manager** (e.g., systemd, supervisor):
```ini
[program:bloomberg-terminal]
command=/path/to/venv/bin/python start_bloomberg_terminal.py
directory=/path/to/Models
autostart=true
autorestart=true
```

4. **Use reverse proxy** (nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8050;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

## 📈 Model Training

### Training RL Agents

```python
from models.ml.rl_agents import StableBaselines3Wrapper
from models.ml.advanced_trading import RLReadyEnvironment
from core.data_fetcher import DataFetcher

# Fetch data
fetcher = DataFetcher()
df = fetcher.get_stock_data("AAPL", period="2y")

# Create environment
env = RLReadyEnvironment(df, initial_capital=100000)

# Train PPO agent
agent = StableBaselines3Wrapper(agent_type="PPO")
agent.create_agent(env)
agent.train(total_timesteps=50000)
agent.save("models/ppo_aapl")
```

### Training LSTM Models

```python
from models.ml.advanced_trading import LSTMPredictor

lstm = LSTMPredictor(lookback_window=20, hidden_units=64)
lstm.train(df, epochs=50, batch_size=32)
predictions = lstm.predict(df)
```

## 🎨 Customization

### Custom Symbols

Edit `start_bloomberg_terminal.py`:
```python
symbols = ["YOUR", "SYMBOLS", "HERE"]
```

### Custom Risk Limits

```python
orchestrator = AutomatedTradingOrchestrator(
    symbols=symbols,
    risk_limit=0.01  # 1% risk per trade
)
```

### Custom Retraining Schedule

```python
orchestrator = AutomatedTradingOrchestrator(
    symbols=symbols,
    retrain_frequency="weekly"  # daily, weekly, monthly
)
```

## 🔍 Monitoring & Debugging

### View Logs

```bash
tail -f logs/api.log
tail -f logs/orchestrator.log
```

### Check Model Performance

```python
from core.model_monitor import ModelPerformanceMonitor

monitor = ModelPerformanceMonitor()
performance = monitor.get_performance("ensemble", "AAPL")
print(performance)
```

### View Alerts

```python
from core.alerting_system import AlertingSystem

alerts = AlertingSystem()
critical = alerts.get_alerts(severity=AlertSeverity.CRITICAL)
for alert in critical:
    print(alert.title, alert.message)
```

## 🚨 Important Notes

1. **Paper Trading First**: Always test with `execute=False` before live trading
2. **Risk Management**: Set appropriate risk limits based on your capital
3. **Model Validation**: Monitor model performance and retrain regularly
4. **API Limits**: Be aware of API rate limits (FRED, Alpha Vantage, etc.)
5. **Server Resources**: RL training is resource-intensive; ensure adequate CPU/RAM

## 📚 Additional Resources

- `API_DOCUMENTATION.md` - Complete API reference
- `PROJECT_ARCHITECTURE.md` - System architecture details
- `notebooks/` - Example notebooks and tutorials
- `COMPANY_ANALYSIS_GUIDE.md` - Company analysis features

## 🆘 Troubleshooting

### Models not training
- Check data availability: `df = fetcher.get_stock_data("AAPL", period="2y")`
- Verify dependencies: `pip install torch stable-baselines3`

### WebSocket connection issues
- Check firewall settings
- Verify port availability: `netstat -an | grep 8000`

### Performance issues
- Reduce number of symbols
- Increase retraining interval
- Use GPU for RL training: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## 📝 License

For personal and professional use.

---

**Built with institutional-grade quantitative finance techniques and modern AI/ML/DL/RL technologies.**
