# Multi-Provider Data System - Implementation Guide

**Status:** Phase 1.1 Complete (WIP)  
**Last Updated:** February 9, 2026  
**Components:** 5 providers registered, 18 unit tests passing

---

## Overview

The system now supports 5 data providers with automatic fallback chains:

### Providers

| Provider | Asset Types | API Key | Status | Rate Limit |
|----------|------------|---------|--------|-----------|
| **Polygon.io** | EQUITY, CRYPTO, FOREX | POLYGON_API_KEY | ✅ Registered | 5/min (free) |
| **IEX Cloud** | EQUITY | IEX_API_KEY | ✅ Registered | 100/sec (free) |
| **CoinGecko** | CRYPTO | None | ✅ Registered | 10/sec (free) |
| **NewsAPI** | NEWS | NEWSAPI_KEY | ✅ Registered | 500/day (free) |
| **SEC EDGAR** | FUNDAMENTALS | None | ✅ Registered | 10/sec (free) |
| **yfinance** | EQUITY, CRYPTO | None | ✅ Fallback | Unlimited |

---

## Installation & Setup

### 1. Add API Keys to `.env`

```bash
# Equity & Crypto
POLYGON_API_KEY=pk_...
IEX_API_KEY=pk_...

# News
NEWSAPI_KEY=sk_...

# Macro (existing)
FRED_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

All providers are implemented; dependencies already included:
- `requests` (HTTP)
- `fredapi` (FRED macro)
- `yfinance` (equity fallback)
- `pandas` (data handling)

### 3. Verify Installation

```bash
python -m pytest tests/test_data_providers.py -v
# Should see: 18 passed, 3 skipped (skipped = API keys not set)
```

---

## Basic Usage

### For Application Startup (api/main.py)

```python
from core.data_fetcher import DataFetcher

# In lifespan event:
fetcher = DataFetcher()
registry = fetcher.initialize_provider_registry()
# Output:
# ✓ Polygon.io registered
# ✓ IEX Cloud registered
# ✓ CoinGecko registered
# ✓ NewsAPI registered
# ✓ SEC EDGAR registered

# Store registry in app state for endpoints
app.state.provider_registry = registry
```

### Direct Usage in Models/Scripts

```python
from core.data_fetcher import DataFetcher
from core.data_providers import DataProviderRegistry, AssetType
from datetime import datetime, timedelta

fetcher = DataFetcher()
registry = fetcher.initialize_provider_registry()

# Fetch OHLCV with automatic fallback
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

ohlcv, provider_used = registry.fetch_ohlcv_with_fallback(
    symbol="AAPL",
    start_date=start_date,
    end_date=end_date,
    asset_type=AssetType.EQUITY
)
print(f"Fetched {len(ohlcv)} bars from {provider_used}")

# Fetch latest price
price, provider_used = registry.fetch_latest_price_with_fallback(
    symbol="BTC",
    asset_type=AssetType.CRYPTO
)
print(f"BTC price: ${price:.2f} (from {provider_used})")
```

### Convenience Methods

```python
# Crypto price (CoinGecko → yfinance fallback)
btc_price = fetcher.get_crypto_latest_price("BTC")

# News articles
articles = fetcher.get_news("AAPL", limit=10)
for article in articles:
    print(f"{article['title']} - {article['source']}")

# Company fundamentals (SEC EDGAR)
fundamentals = fetcher.get_fundamentals_from_sec("AAPL")
print(f"Market Cap: ${fundamentals['market_cap']:,.0f}")
print(f"Revenue: ${fundamentals['revenue']:,.0f}")
```

---

## Architecture

### Class Hierarchy

```
DataProvider (abstract)
├── PolygonProvider
├── IEXProvider
├── CoinGeckoProvider
├── NewsAPIProvider
└── SECEdgarProvider

DataProviderRegistry
├── register(provider)
├── set_fallback_chain(["primary", "secondary", ...])
├── fetch_ohlcv_with_fallback(symbol, dates, asset_type)
└── fetch_latest_price_with_fallback(symbol, asset_type)
```

### Data Structures

```python
# OHLCV bar
@dataclass
class OHLCV:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

# Company fundamentals
@dataclass
class FundamentalsData:
    symbol: str
    price: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    market_cap: Optional[float]
    # ... 10+ additional fields
```

---

## Fallback Chain Strategy

When fetching OHLCV or prices:

1. **Polygon.io** (primary) - Most comprehensive, real-time
2. **IEX Cloud** (secondary) - Good for US equities
3. **yfinance** (tertiary, app-level) - Free, reliable
4. **CoinGecko** (for crypto) - Free, good quality

If Polygon is down, system automatically tries IEX. If IEX is down, falls back to yfinance. This provides:
- ✅ High availability
- ✅ Cost optimization (cheap → expensive)
- ✅ Graceful degradation
- ✅ No user impact

---

## API Integration

### Example Endpoint (for api/data_api.py)

```python
from fastapi import APIRouter, Depends
from core.data_fetcher import DataFetcher
from core.data_providers import AssetType
from datetime import datetime, timedelta

router = APIRouter()
fetcher = DataFetcher()

@router.get("/data/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    days: int = 30,
    asset_type: str = "equity"
):
    """Fetch OHLCV with multi-provider fallback."""
    registry = app.state.provider_registry
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        ohlcv, provider = registry.fetch_ohlcv_with_fallback(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            asset_type=AssetType(asset_type)
        )
        
        return {
            "symbol": symbol,
            "bars": [o.to_dict() for o in ohlcv],
            "provider": provider,
            "count": len(ohlcv)
        }
    
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

@router.get("/data/news/{symbol}")
async def get_news(symbol: str, limit: int = 10):
    """Fetch news using NewsAPI."""
    articles = fetcher.get_news(symbol, limit)
    return {
        "symbol": symbol,
        "articles": articles,
        "count": len(articles)
    }

@router.get("/data/fundamentals/{symbol}")
async def get_fundamentals(symbol: str):
    """Fetch fundamentals from SEC EDGAR."""
    fundamentals = fetcher.get_fundamentals_from_sec(symbol)
    return {
        "symbol": symbol,
        "fundamentals": fundamentals
    }
```

---

## Configuration

### API Keys (Environment Variables)

```bash
# .env file or deployment secrets
POLYGON_API_KEY=pk_live_xxxxxxxx
IEX_API_KEY=pk_live_xxxxxxxx
NEWSAPI_KEY=xxxxxxxxxxxxxxxx
FRED_API_KEY=xxxxxxxxxxxxxxxx
```

### Programmatic Configuration

```python
from core.data_providers import PolygonProvider, IEXProvider

# Initialize with custom API key
polygon = PolygonProvider(api_key="custom_key")

# Check API key validity
if polygon.validate_api_key():
    print("✓ Polygon key is valid")
else:
    print("✗ Polygon key is invalid or expired")
```

---

## Testing

### Run Unit Tests

```bash
# All tests (mocked data)
python -m pytest tests/test_data_providers.py -v

# Specific test class
python -m pytest tests/test_data_providers.py::TestPolygonProvider -v

# With live API (requires keys)
POLYGON_API_KEY=... pytest tests/test_data_providers.py::TestProviderIntegration -v
```

### Test Categories

1. **Data Classes** (OHLCV, FundamentalsData) - ✅ 2 tests
2. **Registry & Fallback** - ✅ 3 tests
3. **Polygon Provider** - ✅ 2 tests
4. **IEX Provider** - ✅ 2 tests
5. **CoinGecko Provider** - ✅ 3 tests (includes live test)
6. **NewsAPI Provider** - ✅ 2 tests
7. **SEC EDGAR Provider** - ✅ 2 tests
8. **Integration Tests** - ⏭️ 3 skipped (require API keys)

**Total: 18 passing + 3 skipped**

---

## Error Handling

Each provider includes:

- **API key validation** - Check before fetching
- **Graceful degradation** - Continue on provider failure
- **Logging** - All errors logged with context
- **Timeout management** - 30s default timeout
- **Rate limit awareness** - Documented limits per provider

Example:

```python
try:
    ohlcv, provider = registry.fetch_ohlcv_with_fallback(...)
except ConnectionError as e:
    logger.error(f"All providers failed: {e}")
    # Fallback to cached data or raise to caller
    return cached_ohlcv
```

---

## Performance Considerations

### Caching

Existing caching still applies:
- Prices: `@cached(ttl=300)` (5 minutes)
- Company info: `@cached(ttl=3600)` (1 hour)
- Economic data: `@cached(ttl=3600)` (1 hour)

### Fallback Overhead

- First provider attempt: ~1-2s
- Fallback attempt (if first fails): ~1-2s additional
- Cached response: ~50ms

### Optimization Tips

```python
# Good: Use provider-specific method if you know the source
polygon = PolygonProvider(api_key=key)
ohlcv = polygon.fetch_ohlcv("AAPL", start, end)

# Better: Use registry for automatic failover
registry.fetch_ohlcv_with_fallback("AAPL", start, end)

# Best: Cache results with longer TTL
@cached(ttl=7200)  # 2 hours
def get_stock_data_cached(ticker, days):
    return registry.fetch_ohlcv_with_fallback(...)
```

---

## Next Steps (Phase 1.2-1.5)

### Story 1.2: Unified Data Fetcher V2
- [ ] Enhance DataFetcher to auto-select best provider per asset
- [ ] Add provider-specific parameters (interval, outputsize, etc.)
- [ ] Implement per-provider retry logic (backoff)

### Story 1.3: Point-in-Time Dataset Layer
- [ ] DatasetSnapshot class (save exact OHLCV + metadata)
- [ ] Hash-based reproducibility (sha256 verification)
- [ ] Load historical snapshot with timestamp guarantee

### Story 1.4: Cold Storage + Audit Trail
- [ ] Parquet-based historical storage (symbol/year/month)
- [ ] Audit logging (JSONL) for all fetches
- [ ] Query interface for historical data retrieval

### Story 1.5: 10-Year Backfill Script
- [ ] Batch download top-500 symbols
- [ ] Store in cold storage
- [ ] Scheduled weekly/monthly updates

---

## Troubleshooting

### API Key Not Working

```bash
# Check key is in .env
grep POLYGON_API_KEY .env

# Or set in terminal
export POLYGON_API_KEY=pk_...
python -c "from core.data_providers import PolygonProvider; \
           p = PolygonProvider(); \
           print('Valid' if p.validate_api_key() else 'Invalid')"
```

### Provider Not Registered

```python
registry.initialize_provider_registry()
# Check startup logs for ✓ or ⚠ status
# If ⚠ shown, check:
# 1. API key is set in .env
# 2. API key has correct permissions
# 3. API endpoint is accessible
```

### Slow Fallback

```python
# Polygon slow? Try directly:
from core.data_providers import IEXProvider
iex = IEXProvider()
ohlcv = iex.fetch_ohlcv(...)
```

---

## References

- **Polygon.io:** https://polygon.io/docs/stocks
- **IEX Cloud:** https://iexcloud.io/docs
- **CoinGecko:** https://www.coingecko.com/en/api
- **NewsAPI:** https://newsapi.org/docs
- **SEC EDGAR:** https://www.sec.gov/cgi-bin/browse-edgar
- **FRED:** https://fred.stlouisfed.org/docs
- **yfinance:** https://github.com/ranaroussi/yfinance

---

## Support

For issues or questions:

1. Check logs: `docker-compose logs api`
2. Run tests: `pytest tests/test_data_providers.py -v`
3. Validate API key: `python -c "from core.data_providers import PolygonProvider; print(PolygonProvider().validate_api_key())"`
4. Check GitHub Issues: ajaiupadhyaya/Models/issues

---

**Status:** Phase 1.1 ✅ Complete  
**Next:** Story 1.2 (UnifiedDataFetcher V2) - ETA Week 3
