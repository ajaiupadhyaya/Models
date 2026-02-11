# Deployment Fix Plan - Making All Features Operational

## Problem Statement

Most features on the deployed application are giving errors or timing out. The project claims many features but they're not operational in production.

## Root Causes Identified

### 1. **Timeout Issues** 🔴 Critical
- External API calls (yfinance, FRED, OpenAI, Alpha Vantage) have no timeout limits
- Requests can hang indefinitely, causing user-facing timeouts
- No retry logic with exponential backoff
- **Impact**: 60-70% of features timeout or hang

### 2. **Missing API Keys** 🔴 Critical  
- Many features require API keys not documented as required
- No graceful degradation when keys missing
- Features return 500 errors instead of informative messages
- **Impact**: AI, Macro data, News, Paper trading completely broken

### 3. **Heavy Dependencies** 🟡 High
- TensorFlow, Stable-Baselines3 cause 60-90s cold start times
- Free tier Render/cloud hosting times out before app loads
- Optional features loaded eagerly instead of lazily
- **Impact**: App never finishes starting on free hosting

### 4. **Poor Error Handling** 🟡 High
- Errors bubble up as generic 500s
- No user-friendly error messages
- No fallback behavior when services unavailable
- **Impact**: Poor user experience, hard to debug

### 5. **Data Fetching Issues** 🟡 Medium
- yfinance rate limiting not handled
- Multi-level column indices cause crashes (partially fixed)
- No caching for expensive data fetches
- **Impact**: Charts show "No data" or crash

### 6. **Resource Constraints** 🟡 Medium
- No request throttling or rate limiting per user
- No connection pooling
- No memory limits on data processing
- **Impact**: One heavy request can crash entire service

## Implementation Plan

### Phase 1: Critical Fixes (Deploy Immediately) ⚡

#### 1.1 Add Timeout Configuration
**Goal**: No request should hang indefinitely

**Files to modify:**
- `api/data_api.py` - Add timeout to yfinance, FRED calls
- `api/ai_analysis_api.py` - Add timeout to OpenAI calls
- `api/news_api.py` - Add timeout to Finnhub calls
- `api/company_analysis_api.py` - Add timeout to all data fetches
- `core/data_fetcher.py` - Add timeout parameter to all fetch methods

**Implementation:**
```python
# Before
data = yf.download(ticker, start, end)

# After
import asyncio
try:
    data = await asyncio.wait_for(
        asyncio.to_thread(yf.download, ticker, start, end),
        timeout=10.0  # 10 second timeout
    )
except asyncio.TimeoutError:
    raise HTTPException(
        status_code=504,
        detail="Data fetch timed out. Try again or use a shorter period."
    )
```

**Timeouts by service:**
- yfinance: 10s for quotes, 20s for historical data
- FRED: 10s for single series, 15s for multiple
- OpenAI: 30s for analysis, 60s for reports
- Finnhub: 5s for news
- Internal computation: 30s max

#### 1.2 Graceful Degradation for Missing Keys
**Goal**: Features work with reduced functionality when keys missing

**Files to modify:**
- `api/data_api.py` - Return mock/cached data when FRED_API_KEY missing
- `api/ai_analysis_api.py` - Return basic analysis when OPENAI_API_KEY missing
- `api/news_api.py` - Return empty list with message when FINNHUB_API_KEY missing
- All routers - Check keys at router load, not request time

**Implementation:**
```python
# At module level
FRED_AVAILABLE = bool(os.environ.get("FRED_API_KEY"))
OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))

# In endpoint
@router.get("/macro")
async def get_macro():
    if not FRED_AVAILABLE:
        return {
            "status": "limited",
            "message": "FRED API key not configured. Showing cached data.",
            "series": get_cached_macro_data(),  # Last known good data
            "last_updated": "2026-02-01"
        }
    # Normal flow...
```

#### 1.3 Optimize Cold Start Time
**Goal**: App starts in <30s on free tier hosting

**Changes:**
1. Make heavy dependencies optional in Dockerfile:
```dockerfile
# Install core deps (always)
RUN pip install fastapi uvicorn pandas numpy yfinance

# Install ML deps (best effort, don't fail build)
RUN pip install tensorflow scikit-learn stable-baselines3 || echo "ML dependencies skipped"
```

2. Lazy load heavy modules:
```python
# Before: Import at top
import tensorflow as tf
from stable_baselines3 import PPO

# After: Import when needed
def get_lstm_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(...)
    except ImportError:
        raise HTTPException(503, "ML features not available in this deployment")
```

3. Remove preloading of models in lifespan:
```python
# Remove from lifespan startup
# loaded_models = await load_saved_models()  # Takes 20-30s

# Load on-demand instead
@router.post("/predictions/predict")
async def predict(...):
    model = get_or_load_model(model_name)  # Lazy load
```

#### 1.4 Better Error Responses
**Goal**: Users see helpful error messages, not 500s

**Implementation:**
```python
# Wrap all endpoints with error handling
@router.get("/analysis/{symbol}")
async def analyze(symbol: str):
    try:
        # Business logic
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Analysis of {symbol} timed out. Try again later."
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {symbol} not found."
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please contact support if this persists."
        )
```

### Phase 2: Performance Improvements (Deploy within 24h) 🚀

#### 2.1 Aggressive Caching
**Goal**: Reduce external API calls by 80%

**Changes:**
- Increase cache TTL for static data (yield curve: 1h → 6h)
- Cache OpenAI responses (24h for company analysis)
- Cache yfinance OHLCV data (15m for recent, 24h for historical)
- Implement Redis for multi-instance caching (optional)

**Implementation:**
```python
from api.cache import get_cached, set_cached, CACHE_TTL_LONG

@router.get("/company/{symbol}/analysis")
async def get_company_analysis(symbol: str):
    cache_key = f"company_analysis:{symbol}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    # Expensive operation
    result = await analyze_company(symbol)
    set_cached(cache_key, result, ttl=86400)  # 24h
    return result
```

#### 2.2 Rate Limiting
**Goal**: Prevent abuse and resource exhaustion

**Files to modify:**
- `api/rate_limit.py` - Implement per-IP rate limiting
- `api/main.py` - Add rate limit middleware

**Limits:**
- Anonymous: 10 req/min, 100 req/hour
- Authenticated: 30 req/min, 500 req/hour
- Expensive endpoints (AI, backtest): 2 req/min

#### 2.3 Request Throttling
**Goal**: Process requests efficiently without crashing

**Implementation:**
- Limit concurrent OpenAI calls to 3
- Limit concurrent yfinance fetches to 10
- Use semaphores to control concurrency

```python
import asyncio

openai_semaphore = asyncio.Semaphore(3)

async def call_openai_api(...):
    async with openai_semaphore:
        # Only 3 concurrent OpenAI calls
        return await openai.ChatCompletion.acreate(...)
```

### Phase 3: Monitoring & Reliability (Deploy within 48h) 📊

#### 3.1 Health Checks for All Services
**Goal**: Know what's working and what's broken

**New endpoint:**
```python
@router.get("/health/detailed")
async def detailed_health():
    return {
        "api": "healthy",
        "services": {
            "yfinance": test_yfinance(),
            "fred": test_fred(),
            "openai": test_openai(),
            "database": test_database(),
        },
        "dependencies": {
            "tensorflow": check_import("tensorflow"),
            "stable_baselines3": check_import("stable_baselines3"),
        },
        "timestamp": datetime.now().isoformat()
    }
```

#### 3.2 Metrics & Logging
**Goal**: Track what's failing and why

**Additions:**
- Log all timeout events with context
- Track success/failure rates per endpoint
- Alert when error rate >10%
- Log slow requests (>2s)

#### 3.3 Circuit Breaker Pattern
**Goal**: Don't retry failing services repeatedly

**Implementation:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time < self.timeout:
                raise ServiceUnavailableError("Circuit breaker is open")
            self.state = "half-open"
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

## Files to Modify

### Critical (Phase 1)
1. `api/data_api.py` - Timeouts, error handling, graceful degradation
2. `api/ai_analysis_api.py` - Timeouts for OpenAI, fallback responses
3. `api/news_api.py` - Timeouts for Finnhub, empty state handling
4. `api/company_analysis_api.py` - Timeouts for all data sources
5. `core/data_fetcher.py` - Add timeout parameter to all methods
6. `Dockerfile` - Make heavy deps optional, optimize build
7. `api/main.py` - Lazy load routers, better error handling

### Performance (Phase 2)
8. `api/cache.py` - Increase TTLs, add Redis support
9. `api/rate_limit.py` - Implement stricter limits
10. `api/main.py` - Add rate limit middleware

### Monitoring (Phase 3)
11. `api/monitoring.py` - Add detailed health checks
12. `api/main.py` - Add logging middleware, circuit breakers

## Testing Plan

### Before Deployment
1. **Local Testing**
   - Test with all API keys
   - Test without API keys (graceful degradation)
   - Test with network delays (artificial 5s delay)
   - Test concurrent requests (100 req/s)

2. **Staging Deployment**
   - Deploy to staging environment
   - Run automated test suite
   - Manual testing of all features
   - Load testing with k6 or locust

### After Deployment
1. **Smoke Tests**
   - Test all endpoints return 200 or expected error
   - Test health check shows all services
   - Test at least one request per feature

2. **Monitor**
   - Watch error logs for first hour
   - Check response times <2s for 95th percentile
   - Verify no timeouts in logs
   - Check memory usage stays <512MB

## Success Criteria

### Immediate (Phase 1)
- ✅ Zero indefinite hangs (all requests timeout within 60s)
- ✅ Clear error messages for missing API keys
- ✅ App starts in <30s on free tier
- ✅ All endpoints return JSON (no 500s)

### 24h (Phase 2)
- ✅ 95th percentile response time <2s
- ✅ Cache hit rate >60%
- ✅ Rate limiting prevents abuse
- ✅ No out-of-memory crashes

### 48h (Phase 3)
- ✅ Detailed health checks show service status
- ✅ Metrics dashboard shows all endpoints
- ✅ Circuit breakers prevent cascade failures
- ✅ Error rate <5%

## Rollback Plan

If deployment causes critical issues:
1. Revert to previous Docker image
2. Check Render logs for errors
3. Restore previous environment variables
4. Notify team

## Required Environment Variables

### Absolutely Required (App won't start without these)
- `PORT` - Set by Render automatically
- `TERMINAL_USER` - Username for login
- `TERMINAL_PASSWORD` - Password for login
- `AUTH_SECRET` - JWT secret (generate with `openssl rand -hex 32`)

### Required for Full Functionality
- `FRED_API_KEY` - Macro data, yield curves (get free key at fred.stlouisfed.org)
- `ALPHA_VANTAGE_API_KEY` - Stock fundamentals (get free key at alphavantage.co)

### Optional (Features degrade gracefully)
- `OPENAI_API_KEY` - AI analysis (falls back to basic analysis)
- `FINNHUB_API_KEY` - News headlines (returns empty list)
- `ALPACA_API_KEY` - Paper trading (feature disabled)
- `ALPACA_API_SECRET` - Paper trading
- `ALPACA_API_BASE` - Paper trading

### Should NOT Be Set
- `VITE_API_ORIGIN` - Only for split deployments; breaks same-origin

## Timeline

- **Phase 1 (Critical)**: Implement today, deploy tonight (4-6 hours)
- **Phase 2 (Performance)**: Implement tomorrow morning, deploy tomorrow night (6-8 hours)
- **Phase 3 (Monitoring)**: Implement day 3, deploy evening (4-6 hours)

## Next Steps

1. Review this plan with team
2. Create branch `fix/deployment-issues`
3. Implement Phase 1 changes
4. Test locally without API keys
5. Test locally with artificial delays
6. Deploy to staging
7. Deploy to production
8. Monitor for 1 hour
9. Proceed to Phase 2

---

**Last Updated**: 2026-02-11  
**Status**: Ready for Implementation  
**Priority**: 🔴 Critical - Deploy ASAP
