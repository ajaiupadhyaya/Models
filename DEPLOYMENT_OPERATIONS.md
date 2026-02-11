# Deployment Operations & Troubleshooting Guide

Complete guide to ensure all APIs, Webhooks, and features are fully functional and operational when deployed to Render.

## Table of Contents
1. [Pre-Deployment Checks](#pre-deployment-checks)
2. [Deployment Process](#deployment-process)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Webhook Integration](#webhook-integration)
7. [Performance Optimization](#performance-optimization)

---

## Pre-Deployment Checks

### 1. Run Deployment Validation

```bash
# Perform comprehensive deployment validation
python deployment_validation.py

# This checks:
# - Environment variables
# - Configuration files
# - API routers and imports
# - Dependencies
# - Frontend build
# - Data directories
```

Expected output: "All deployment checks passed!"

### 2. Run Deployment Fixes

```bash
# Prepare deployment and ensure all requirements are met
python deployment_fixes.py

# This:
# - Creates .env file if missing
# - Creates required data directories
# - Builds frontend if needed
# - Validates configuration
# - Creates production checklist
```

### 3. Essential Environment Variables

**Critical (MUST be set):**
```env
TERMINAL_USER=your_username
TERMINAL_PASSWORD=your_password
AUTH_SECRET=your_secret_key_at_least_32_chars
```

**Highly Recommended (for data features):**
```env
FRED_API_KEY=your_fred_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
```

**Optional (for enhanced features):**
```env
OPENAI_API_KEY=your_openai_key
FINNHUB_API_KEY=your_finnhub_key
ENABLE_PAPER_TRADING=true
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

### 4. Verify Local Deployment

```bash
# Start the API locally
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test critical endpoints:
curl http://localhost:8000/health
curl http://localhost:8000/info
curl "http://localhost:8000/api/v1/data/quotes"
```

---

## Deployment Process

### 1. Push to GitHub

```bash
# Ensure all changes are committed
git add .
git commit -m "Prepare for deployment: validation and fixes"
git push origin main
```

### 2. Trigger Render Deployment

**Option A: Via Render Dashboard**
1. Go to https://dashboard.render.com
2. Select your service (e.g., `terminal-api`)
3. Click **"Deploy latest commit"** or **"Manual Deploy"**
4. Monitor the build logs

**Option B: Via Render CLI**
```bash
# List services
render services -o json --confirm

# Trigger deployment for your service
render deploys create srv-YOUR_SERVICE_ID --wait

# Clear build cache if needed
render deploys create srv-YOUR_SERVICE_ID --clear-cache --wait

# Monitor logs
render logs -r srv-YOUR_SERVICE_ID --tail
```

### 3. Build Process Timeline

```
~30s   : Docker build starts
~1m    : Base image pulled
~2m    : npm build completes
~4m    : Python dependencies installed
~5m    : Build complete, deployment starts
~6m-7m : Container starts, API initializes
~8m+   : Health checks pass
```

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# Basic health check (CRITICAL)
curl https://your-service.onrender.com/health

# Expected response:
{
  "status": "healthy",
  "models_loaded": N,
  "active_connections": 0,
  "metrics_collector": true
}
```

### 2. System Information

```bash
# Check loaded routers and capabilities
curl https://your-service.onrender.com/info

# Expected response shows:
{
  "api_version": "1.0.0",
  "routers_loaded": ["models", "predictions", "backtesting", ...],
  "capabilities": ["ai", "ml", "dl", "rl"],
  "websocket": {"enabled": true, "connections": 0},
  "monitoring": {"enabled": true}
}
```

### 3. Data Endpoints

```bash
# Get quotes (requires yfinance working)
curl "https://your-service.onrender.com/api/v1/data/quotes?symbols=AAPL,MSFT"

# Get macro data (requires FRED_API_KEY)
curl "https://your-service.onrender.com/api/v1/data/macro"

# Get risk metrics
curl "https://your-service.onrender.com/api/v1/risk/metrics/AAPL"
```

### 4. Authentication

```bash
# Get auth token
curl -X POST https://your-service.onrender.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"TERMINAL_USER","password":"TERMINAL_PASSWORD"}'

# Get current user info (use returned token)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-service.onrender.com/api/v1/auth/me
```

### 5. WebSocket Connection

```bash
# Test WebSocket connection
wscat -c "wss://your-service.onrender.com/api/v1/ws/prices/AAPL"

# Send subscription message
{"action": "subscribe", "symbol": "AAPL"}

# Should receive price updates
```

### 6. Predictions & Backtesting

```bash
# Quick prediction
curl "https://your-service.onrender.com/api/v1/predictions/quick-predict?symbol=AAPL"

# Backtesting sample data
curl "https://your-service.onrender.com/api/v1/backtest/sample-data?symbol=AAPL&period=1y"
```

---

## Common Issues & Solutions

### Issue 1: `/health` returns 404

**Cause:** API not started or frontend is serving 404 fallback

**Solution:**
```bash
# Check server logs
render logs -r srv-YOUR_SERVICE_ID --tail

# Should see: "API server ready!"
```

### Issue 2: Endpoints return 200 but empty data

**Cause:** yfinance failing due to browser impersonation blocks

**Status:** ✓ FIXED in latest deployment (curl_cffi disabled)

**Verification:**
```bash
curl "https://service.onrender.com/api/v1/data/quotes"

# Should return actual prices, not empty array
```

### Issue 3: FRED data not loading

**Cause:** FRED_API_KEY not set or invalid

**Solution:**
1. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
2. Add to Render → Environment Variables
3. Redeploy: `render deploys create srv-ID --wait`

### Issue 4: "API unreachable" or WebSocket errors

**Cause:** VITE_API_ORIGIN set (should NOT be set for single service)

**Solution:**
1. In Render dashboard → Environment
2. DELETE `VITE_API_ORIGIN` variable
3. Redeploy the frontend

### Issue 5: WebSocket connections failing

**Cause:** WebSocket disabled or path wrong

**Solution:**
```env
WEBSOCKET_ENABLED=true  # Ensure set in Render environment
```

### Issue 6: Deployment timeout (>15 minutes)

**Cause:** Large dependencies taking too long to install

**Solution:**
```bash
# Clear Docker cache
render deploys create srv-ID --clear-cache --wait

# Or manually restart in dashboard
```

### Issue 7: "Too many requests" (429)

**Cause:** Rate limit exceeded (100 req/60s per IP)

**Solution:**
- If legitimate load, increase `RATE_LIMIT_REQUESTS` in api/rate_limit.py
- Deploy again

---

## Monitoring & Maintenance

### 1. Real-Time Logs

```bash
# Stream logs (follow mode)
render logs -r srv-YOUR_SERVICE_ID --tail

# Get last 200 lines
render logs -r srv-YOUR_SERVICE_ID --limit 200 -o text --confirm

# Search for errors
render logs -r srv-YOUR_SERVICE_ID --limit 500 -o text --confirm | grep ERROR
```

### 2. Critical Log Patterns

**✓ Healthy:**
```
API server ready!
Routers registered successfully
WebSocket connections enabled
Metrics collector initialized
```

**✗ Problems:**
```
Router {name} not available  -- Optional router failed
Failed to fetch data for {ticker}  -- yfinance issue
FRED API key not configured  -- Missing required key
```

### 3. Metrics Endpoint

```bash
# Get performance metrics
curl "https://service.onrender.com/api/v1/monitoring/dashboard"

# Returns:
{
  "predictions": [...],
  "api_performance": {...},
  "websocket_stats": {...},
  "cache_stats": {...},
  "rate_limit_stats": {...}
}
```

### 4. Set Up Monitoring Alerts

In Render Dashboard → Service → Notifications:
1. Enable "Failure on Deploy"
2. Enable "Daily Status Report"
3. Add PagerDuty or Slack webhook for critical errors

---

## Webhook Integration

### Incoming Webhooks (External → Your API)

**Example: Receive signals from external source**

```python
# Add to api/main.py or create new endpoint
@app.post("/api/v1/webhooks/signals")
async def receive_signal(payload: Dict) -> Dict:
    """
    Receive trading signals from external sources.
    
    Expected payload:
    {
        "symbol": "AAPL",
        "signal": 1.0,  # 1=BUY, -1=SELL, 0=HOLD
        "confidence": 0.95,
        "source": "external_model"
    }
    """
    # Validate signal
    # Send to WebSocket connected clients
    # Log to metrics
    return {"status": "received", "id": str(uuid.uuid4())}
```

### Outgoing Webhooks (Your API → External)

**Example: Notify external system on predictions**

```python
# In api/predictions_api.py
async def notify_external_webhook(prediction: Dict):
    """Send prediction to external webhook.
    
    Configure webhook URL in environment:
    WEBHOOK_URL=https://external-system.com/signals
    WEBHOOK_SECRET=signing_key
    """
    import httpx
    import hmac
    import hashlib
    
    webhook_url = os.getenv("WEBHOOK_URL")
    webhook_secret = os.getenv("WEBHOOK_SECRET")
    
    if not webhook_url:
        return
    
    # Create HMAC signature
    signature = hmac.new(
        webhook_secret.encode(),
        json.dumps(prediction).encode(),
        hashlib.sha256
    ).hexdigest()
    
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                webhook_url,
                json=prediction,
                headers={"X-Signature": signature},
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {e}")
```

---

## Performance Optimization

### 1. Caching Strategy

Currently implemented:
- Response cache: 5-15 min TTL
- Rate limiting: 100 req/60s per IP
- WebSocket connection reuse

Add Redis for multi-instance:
```python
# Replace TTLCache with Redis in api/cache.py
import aioredis

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
    
    async def get(self, key: str):
        if not self.redis:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
        return await self.redis.get(key)
```

### 2. Database Connection Pooling

For future database integration:
```python
# Use async database connection pooling
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@host/db",
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
)
```

### 3. Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py
# Run load test
locust -f locustfile.py --host=https://your-service.onrender.com

# Check /info endpoint can handle 100+ concurrent requests
```

### 4. Response Time Targets

- `/health` — <10ms
- `/info` — <20ms
- `/api/v1/data/quotes` — <500ms
- `/api/v1/predictions/quick-predict` — <1000ms
- `/api/v1/risk/metrics/{ticker}` — <2000ms

---

## Production Readiness Checklist

Before marking deployment as "production-ready":

- [ ] All `/health` checks pass
- [ ] `/info` shows all expected routers
- [ ] Data endpoints returning real prices (not nulls)
- [ ] Authentication working with configured credentials
- [ ] WebSocket connections established
- [ ] Render logs clean (no startup errors)
- [ ] Response times within targets
- [ ] No 5xx errors in logs
- [ ] Rate limiting working
- [ ] CORS allowing frontend origin
- [ ] SSL certificate valid (automatic on Render)
- [ ] Error alerting configured

---

## Maintenance Schedule

**Daily:**
- Check `/health` endpoint
- Review Render logs for errors

**Weekly:**
- Verify all data endpoints
- Check rate limit stats
- Review WebSocket connection stats

**Monthly:**
- Run deployment_validation.py script
- Update authentication credentials if needed
- Review performance metrics
- Check for deprecated dependencies

**As Needed:**
- Update API keys (FRED, Alpha Vantage, OpenAI)
- Patch security vulnerabilities
- Optimize slow endpoints
- Scale for increased load

---

## Emergency Procedures

**API Down:**
```bash
# Check service status
render services -o json --confirm

# Force redeploy
render deploys create srv-ID --clear-cache --wait

# Check logs
render logs -r srv-ID --tail
```

**High Error Rate:**
1. Check logs for specific error pattern
2. Check external API status (FRED, yfinance, etc.)
3. Check environment variables in Render dashboard
4. Rollback if recent deploy caused issue:
   ```bash
   render deploys list srv-ID  # Find previous deploy
   # Manually trigger that commit in GitHub
   ```

**OOM (Out of Memory):**
- Render free tier: 512 MB
- Increase plan if needed
- Clear cache: `python api/cache.py clear()`

---

## Support & Resources

- **Render Docs:** https://render.com/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **yfinance Issues:** https://github.com/ranaroussi/yfinance/issues
- **Check Deployment Status:** https://your-service.onrender.com/health

