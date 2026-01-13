# Trading ML API - Complete Documentation

## Overview

Production-grade FastAPI server for machine learning trading models with real-time predictions, backtesting, WebSocket streaming, and performance monitoring.

## Features

✅ **Model Management**: Train, load, save, and manage ML models
✅ **Real-Time Predictions**: Single, batch, and ensemble predictions
✅ **Backtesting**: Run historical backtests with walk-forward analysis
✅ **WebSocket Streaming**: Live price and prediction updates
✅ **Monitoring**: Comprehensive metrics and performance tracking
✅ **Production-Ready**: Full error handling, logging, and CORS support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Start the Server

```bash
# Development mode (with auto-reload)
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Health & System

#### `GET /`
Root endpoint - API information

```bash
curl http://localhost:8000/
```

#### `GET /health`
Health check with system metrics

```bash
curl http://localhost:8000/health
```

#### `GET /info`
Detailed system information

```bash
curl http://localhost:8000/info
```

---

### Model Management (`/api/v1/models`)

#### `GET /api/v1/models/`
List all available models

```bash
curl http://localhost:8000/api/v1/models/
```

**Response:**
```json
{
  "models": [
    {
      "name": "ensemble_spy",
      "type": "ensemble",
      "symbol": "SPY",
      "trained_date": "2024-01-13T10:30:00",
      "parameters": {},
      "status": "loaded"
    }
  ],
  "count": 1
}
```

#### `POST /api/v1/models/train`
Train a new model

```bash
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "ensemble",
    "model_name": "ensemble_spy",
    "symbol": "SPY",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01"
  }'
```

**Parameters:**
- `model_type`: "simple", "ensemble", or "lstm"
- `model_name`: Unique name for the model
- `symbol`: Stock symbol (e.g., "SPY", "AAPL")
- `start_date`: Training start date (YYYY-MM-DD)
- `end_date`: Training end date (optional)

**Response:**
```json
{
  "status": "success",
  "model_name": "ensemble_spy",
  "message": "Model trained in 45.32s",
  "training_time": 45.32
}
```

#### `GET /api/v1/models/{model_name}`
Get model details

```bash
curl http://localhost:8000/api/v1/models/ensemble_spy
```

#### `DELETE /api/v1/models/{model_name}`
Delete a model

```bash
curl -X DELETE http://localhost:8000/api/v1/models/ensemble_spy
```

#### `POST /api/v1/models/{model_name}/reload`
Reload a model from disk

```bash
curl -X POST http://localhost:8000/api/v1/models/ensemble_spy/reload
```

---

### Predictions (`/api/v1/predictions`)

#### `POST /api/v1/predictions/predict`
Generate a single prediction

```bash
curl -X POST http://localhost:8000/api/v1/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_spy",
    "symbol": "SPY",
    "days_lookback": 60
  }'
```

**Response:**
```json
{
  "model_name": "ensemble_spy",
  "symbol": "SPY",
  "timestamp": "2024-01-13T15:30:00",
  "signal": 0.65,
  "confidence": 0.75,
  "current_price": 450.23,
  "recommendation": "BUY",
  "metadata": {
    "model_type": "ensemble",
    "data_points": 60,
    "latest_date": "2024-01-13"
  }
}
```

**Signal Values:**
- `> 0.2`: BUY signal
- `-0.2 to 0.2`: HOLD signal
- `< -0.2`: SELL signal

#### `POST /api/v1/predictions/predict/batch`
Generate predictions for multiple symbols

```bash
curl -X POST http://localhost:8000/api/v1/predictions/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_spy",
    "symbols": ["SPY", "QQQ", "IWM"],
    "days_lookback": 60
  }'
```

**Response:**
```json
{
  "model_name": "ensemble_spy",
  "timestamp": "2024-01-13T15:30:00",
  "predictions": [
    {
      "symbol": "SPY",
      "signal": 0.65,
      "confidence": 0.75,
      "recommendation": "BUY",
      "current_price": 450.23
    }
  ],
  "summary": {
    "total": 3,
    "successful": 3,
    "buy_signals": 2,
    "sell_signals": 0,
    "hold_signals": 1
  }
}
```

#### `POST /api/v1/predictions/predict/ensemble`
Ensemble prediction from multiple models

```bash
curl -X POST http://localhost:8000/api/v1/predictions/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["model1", "model2", "model3"],
    "symbol": "SPY",
    "weights": [0.5, 0.3, 0.2]
  }'
```

**Response:**
```json
{
  "models_used": ["model1", "model2", "model3"],
  "symbol": "SPY",
  "timestamp": "2024-01-13T15:30:00",
  "ensemble_signal": 0.58,
  "ensemble_confidence": 0.72,
  "individual_predictions": [
    {
      "model_name": "model1",
      "signal": 0.65,
      "confidence": 0.75,
      "weight": 0.5,
      "weighted_signal": 0.325
    }
  ],
  "recommendation": "BUY"
}
```

#### `GET /api/v1/predictions/models/{model_name}/signals/{symbol}?days=30`
Get historical signals

```bash
curl "http://localhost:8000/api/v1/predictions/models/ensemble_spy/signals/SPY?days=30"
```

---

### Backtesting (`/api/v1/backtest`)

#### `POST /api/v1/backtest/run`
Run a backtest

```bash
curl -X POST http://localhost:8000/api/v1/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_spy",
    "symbol": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000,
    "commission": 0.001,
    "position_size": 1.0
  }'
```

**Response:**
```json
{
  "model_name": "ensemble_spy",
  "symbol": "SPY",
  "period": {
    "start": "2023-01-01",
    "end": "2024-01-01"
  },
  "metrics": {
    "total_return": 0.201,
    "annual_return": 0.201,
    "sharpe_ratio": 1.62,
    "max_drawdown": -0.065,
    "num_trades": 45,
    "win_rate": 0.58
  },
  "equity_curve": [...],
  "trades": [...],
  "status": "success"
}
```

#### `POST /api/v1/backtest/compare`
Compare multiple strategies

```bash
curl -X POST http://localhost:8000/api/v1/backtest/compare \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["model1", "model2", "model3"],
    "symbol": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }'
```

#### `POST /api/v1/backtest/walk-forward`
Run walk-forward analysis

```bash
curl -X POST http://localhost:8000/api/v1/backtest/walk-forward \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_spy",
    "symbol": "SPY",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "train_window": 252,
    "test_window": 63
  }'
```

#### `GET /api/v1/backtest/metrics`
Get available metrics

```bash
curl http://localhost:8000/api/v1/backtest/metrics
```

---

### WebSocket Streaming (`/api/v1/ws`)

#### `WS /api/v1/ws/prices/{symbol}`
Stream real-time prices

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/prices/SPY');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  // {
  //   "type": "price_update",
  //   "symbol": "SPY",
  //   "price": 450.23,
  //   "timestamp": "2024-01-13T15:30:00",
  //   "volume": 50000000
  // }
};
```

#### `WS /api/v1/ws/predictions/{model_name}/{symbol}`
Stream real-time predictions

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/predictions/ensemble_spy/SPY');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  // {
  //   "type": "prediction_update",
  //   "model_name": "ensemble_spy",
  //   "symbol": "SPY",
  //   "signal": 0.65,
  //   "confidence": 0.75,
  //   "recommendation": "BUY",
  //   "timestamp": "2024-01-13T15:30:00"
  // }
};
```

#### `WS /api/v1/ws/live`
General-purpose live feed with commands

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/live');

// Subscribe to a symbol
ws.send(JSON.stringify({
  action: "subscribe",
  symbol: "SPY"
}));

// Request a prediction
ws.send(JSON.stringify({
  action: "predict",
  model: "ensemble_spy",
  symbol: "SPY"
}));

// Unsubscribe
ws.send(JSON.stringify({
  action: "unsubscribe",
  symbol: "SPY"
}));
```

---

### Monitoring (`/api/v1/monitoring`)

#### `GET /api/v1/monitoring/system`
Get system metrics

```bash
curl http://localhost:8000/api/v1/monitoring/system
```

**Response:**
```json
{
  "total_predictions": 1523,
  "total_api_calls": 3042,
  "total_errors": 5,
  "recent_api_calls_1h": 125,
  "recent_errors_24h": 2,
  "avg_response_time": 0.245,
  "uptime": "N/A"
}
```

#### `GET /api/v1/monitoring/models/{model_name}`
Get model-specific metrics

```bash
curl http://localhost:8000/api/v1/monitoring/models/ensemble_spy
```

**Response:**
```json
{
  "model_name": "ensemble_spy",
  "total_predictions": 523,
  "avg_confidence": 0.72,
  "avg_signal": 0.15,
  "buy_signals": 312,
  "sell_signals": 89,
  "hold_signals": 122
}
```

#### `GET /api/v1/monitoring/dashboard`
Get comprehensive dashboard data

```bash
curl http://localhost:8000/api/v1/monitoring/dashboard
```

#### `GET /api/v1/monitoring/predictions/recent?limit=100`
Get recent predictions

```bash
curl "http://localhost:8000/api/v1/monitoring/predictions/recent?limit=100"
```

#### `GET /api/v1/monitoring/errors/recent?limit=50`
Get recent errors

```bash
curl "http://localhost:8000/api/v1/monitoring/errors/recent?limit=50"
```

#### `POST /api/v1/monitoring/save`
Save metrics to disk

```bash
curl -X POST http://localhost:8000/api/v1/monitoring/save
```

#### `GET /api/v1/monitoring/history`
Get saved metrics files

```bash
curl http://localhost:8000/api/v1/monitoring/history
```

---

## Python Client Examples

### Training a Model

```python
import requests

url = "http://localhost:8000/api/v1/models/train"
data = {
    "model_type": "ensemble",
    "model_name": "my_ensemble_spy",
    "symbol": "SPY",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01"
}

response = requests.post(url, json=data)
print(response.json())
```

### Getting a Prediction

```python
import requests

url = "http://localhost:8000/api/v1/predictions/predict"
data = {
    "model_name": "my_ensemble_spy",
    "symbol": "SPY",
    "days_lookback": 60
}

response = requests.post(url, json=data)
result = response.json()

print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']}")
print(f"Recommendation: {result['recommendation']}")
print(f"Current Price: ${result['current_price']:.2f}")
```

### Running a Backtest

```python
import requests

url = "http://localhost:8000/api/v1/backtest/run"
data = {
    "model_name": "my_ensemble_spy",
    "symbol": "SPY",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000,
    "commission": 0.001
}

response = requests.post(url, json=data)
result = response.json()

print(f"Total Return: {result['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result['metrics']['max_drawdown']:.2%}")
print(f"Win Rate: {result['metrics']['win_rate']:.2%}")
```

### WebSocket Client

```python
import asyncio
import websockets
import json

async def stream_predictions():
    uri = "ws://localhost:8000/api/v1/ws/predictions/my_ensemble_spy/SPY"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "prediction_update":
                print(f"Signal: {data['signal']:.2f}")
                print(f"Recommendation: {data['recommendation']}")
                print(f"Price: ${data['current_price']:.2f}")
                print("-" * 40)

# Run the client
asyncio.run(stream_predictions())
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error message here",
  "status_code": 400
}
```

**Common Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (model doesn't exist)
- `500`: Internal Server Error
- `503`: Service Unavailable (metrics collector not ready)

---

## Configuration

### Environment Variables

```bash
# Optional configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export LOG_LEVEL="INFO"
```

### CORS Configuration

Edit `api/main.py` to configure CORS for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

---

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Using Docker (coming next)

```bash
docker build -t trading-ml-api .
docker run -p 8000:8000 trading-ml-api
```

---

## Monitoring & Logging

### Logs

All API activity is logged with timestamps:

```
2024-01-13 15:30:00 - api.predictions - INFO - Fetching data for SPY
2024-01-13 15:30:01 - api.predictions - INFO - Generated prediction: signal=0.65
```

### Metrics Storage

Metrics are automatically saved to `data/metrics/` directory:
- Predictions tracked in memory
- API calls logged with duration
- Errors recorded with context
- Periodic saves to disk

---

## Architecture

```
api/
├── main.py              # FastAPI application & routing
├── models_api.py        # Model management endpoints
├── predictions_api.py   # Prediction endpoints
├── backtesting_api.py   # Backtesting endpoints
├── websocket_api.py     # WebSocket streaming
└── monitoring.py        # Metrics & monitoring
```

**Key Components:**
- **ConnectionManager**: WebSocket connection pooling
- **MetricsCollector**: Real-time metrics tracking
- **BacktestEngine**: Historical backtesting
- **Model Loaders**: Automatic model loading on startup

---

## Performance

**Typical Response Times:**
- Model prediction: 200-500ms
- Backtest (1 year): 2-5 seconds
- Walk-forward analysis: 10-30 seconds
- WebSocket latency: <50ms

**Scalability:**
- Horizontal scaling via load balancer
- Model caching for fast predictions
- Async WebSocket for real-time updates
- Background task processing

---

## Security Considerations

**For Production:**
1. Add API key authentication
2. Rate limiting per endpoint
3. HTTPS/WSS only
4. Input validation & sanitization
5. SQL injection protection (if using DB)
6. CORS restricted to specific domains

---

## Support & Troubleshooting

### Common Issues

**1. Model Not Found**
```bash
# List all models
curl http://localhost:8000/api/v1/models/

# Train a new model first
curl -X POST http://localhost:8000/api/v1/models/train -H "Content-Type: application/json" -d '{"model_type": "ensemble", "model_name": "test_model", "symbol": "SPY", "start_date": "2022-01-01"}'
```

**2. WebSocket Connection Refused**
- Check server is running on correct port
- Verify firewall settings
- Use `ws://` for local, `wss://` for HTTPS

**3. Slow Predictions**
- Reduce `days_lookback` parameter
- Use simple model type for faster results
- Cache predictions for repeated requests

---

## Next Steps

1. ✅ **Test the API** - Try the examples above
2. ⏳ **Docker Deployment** - Containerize for production
3. ⏳ **Paper Trading** - Integrate with broker API
4. ⏳ **Authentication** - Add API keys
5. ⏳ **Database** - Persist predictions & metrics
6. ⏳ **Dashboard UI** - Build web interface

---

## License

Part of the Trading ML Platform project.

## Contact

For questions or issues, refer to the main project documentation.
