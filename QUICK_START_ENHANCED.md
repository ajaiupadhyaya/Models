# Quick Start - Enhanced Bloomberg Terminal Platform

## 🚀 Ultra-Fast Setup (5 Minutes)

### 1. Install Backend Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 2. Configure API Keys
Create `.env` file:
```env
FRED_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
OPENAI_API_KEY=your_openai_key   # optional, for AI analysis
LLM_PROVIDER=openai              # or leave unset to disable
```

### 3. Start FastAPI Backend
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Start React + D3 Bloomberg Terminal UI
```bash
cd ../frontend
npm install
npm run dev
```

**That's it!** Open http://localhost:5173

## ✨ What's New & Enhanced

### 🎯 Advanced Quantitative Features
- **Factor Models**: Multi-factor analysis (Fama-French style)
- **Regime Detection**: Market regime identification (Bull/Bear/Sideways)
- **Portfolio Optimization**: Risk parity and max Sharpe optimization
- **Alternative Data**: Sentiment analysis, volume profiles, anomaly detection

### ⚡ Performance Optimizations
- **Smart Caching**: Intelligent caching with TTL and size limits
- **Parallel Processing**: Multi-threaded data fetching
- **DataFrame Optimization**: Memory-efficient data structures
- **Lazy Loading**: Models load only when needed

### 🎨 Enhanced UI Features
- **Real-time Updates**: Live REST/WS data from FastAPI
- **Regime Indicators**: Market regime display from enhanced orchestrator
- **Factor Exposure**: Factor loadings for each asset
- **Portfolio Analytics**: Advanced portfolio metrics (planned React views)

## 📊 Usage Examples

### Enhanced Orchestrator
```python
from core.enhanced_orchestrator import EnhancedOrchestrator

orchestrator = EnhancedOrchestrator(
    symbols=["AAPL", "MSFT", "GOOGL"],
    use_rl=True,
    use_lstm=True,
    use_ensemble=True
)

# Analyze market regime
regime = orchestrator.analyze_market_regime()
print(f"Current regime: {regime['regime_label']}")

# Get factor exposure
factors = orchestrator.get_factor_exposure("AAPL")
print(f"Factor exposure: {factors}")

# Generate enhanced signals (with regime adjustment)
signals = orchestrator.generate_enhanced_signals("AAPL")
```

### Factor Models
```python
from models.quant.advanced_models import FactorModel
from core.data_fetcher import DataFetcher

fetcher = DataFetcher()
returns = fetcher.get_stock_data("AAPL", period="1y")['Close'].pct_change()

factor_model = FactorModel(n_factors=3)
results = factor_model.fit(returns)
print(f"Explained variance: {results['explained_variance']}")
```

### Regime Detection
```python
from models.quant.advanced_models import RegimeDetector

detector = RegimeDetector(n_regimes=3)
regimes = detector.detect_regimes(returns)
current = detector.get_current_regime()
print(f"Current regime: {current}")
```

### Portfolio Optimization
```python
from models.quant.advanced_models import PortfolioOptimizerAdvanced

optimizer = PortfolioOptimizerAdvanced()
weights = optimizer.optimize_risk_parity(returns_df)
print(f"Optimal weights: {weights}")
```

## 🔥 Performance Tips

1. **Use Caching**: Results are automatically cached
2. **Batch Operations**: Process multiple symbols together
3. **Lazy Loading**: Models initialize on first use
4. **Parallel Processing**: Data fetching uses multiple threads

## 🎯 Key Features

✅ **Fully Automated** - End-to-end orchestration  
✅ **AI/ML/DL/RL** - Ensemble, LSTM, RL agents  
✅ **Advanced Quant** - Factor models, regime detection  
✅ **Real-time** - WebSocket streaming  
✅ **Optimized** - Smart caching, parallel processing  
✅ **Production Ready** - Error handling, monitoring  

## 📚 Next Steps

- Read `BLOOMBERG_TERMINAL_GUIDE.md` for detailed documentation
- Check `API_DOCUMENTATION.md` for API endpoints
- Explore `notebooks/` for examples

---

**Ready to trade like a quant!** 🚀
