# Production Completion Roadmap (16 Weeks)

**Status:** February 9, 2026  
**Goal:** Execute all remaining features from context.md to achieve Bloomberg-terminal-level institutional grade production system.  
**Quality Bar:** No demos, no toys; institutional rigor on data, quant, AI, UI, and infrastructure.

---

## Table of Contents

1. [Feature Gap Analysis](#feature-gap-analysis)
2. [Phase Timeline Overview](#phase-timeline-overview)
3. [Phase 0: Gap Lock & Baseline (Week 1)](#phase-0-gap-lock--baseline-week-1)
4. [Phase 1: Data Foundation (Weeks 2-4)](#phase-1-data-foundation-weeks-2-4)
5. [Phase 2: Quant Engine Completion (Weeks 5-7)](#phase-2-quant-engine-completion-weeks-5-7)
6. [Phase 3: AI/ML/RL Pipeline (Weeks 8-10)](#phase-3-aimlrl-pipeline-weeks-8-10)
7. [Phase 4: Terminal UI & D3 (Weeks 11-13)](#phase-4-terminal-ui--d3-weeks-11-13)
8. [Phase 5: Production Hardening (Weeks 14-15)](#phase-5-production-hardening-weeks-14-15)
9. [Phase 6: Testing & Documentation (Week 16)](#phase-6-testing--documentation-week-16)
10. [Deployment Gates & Acceptance](#deployment-gates--acceptance)

---

## Feature Gap Analysis

### Context.md vs Current State

| Requirement | Category | Current State | Gap Status | Priority |
|-------------|----------|---------------|-----------|----------|
| **Data: Equities/ETF/Options/Futures/FX/Crypto** | Data | yfinance + Alpha Vantage | Partial (equities/crypto only) | CRITICAL |
| **Data: Fundamentals & Filings** | Data | Basic via yfinance | Partial (no SEC EDGAR, limited ratios) | CRITICAL |
| **Data: News & Sentiment** | Data | Not implemented | Missing entirely | HIGH |
| **Data: Macro (FRED/WB/IMF)** | Data | FRED only | Partial (no World Bank/IMF) | HIGH |
| **Data: Alternative/Sentiment** | Data | Not implemented | Missing | MEDIUM |
| **Point-in-Time + Survivorship-safe datasets** | Data | Documented goal | Not implemented | HIGH |
| **Time-series + Cross-sectional analysis** | Quant | Partial (backtesting) | Incomplete framework | HIGH |
| **Factor modeling + Regime detection** | Quant | ML models exist | Not wired to terminal | MEDIUM |
| **Options surface + Greeks** | Quant | Black-Scholes available | Not exposed in API/terminal | MEDIUM |
| **ML training pipeline + MLflow** | AI/ML | Basic models exist | No MLflow/governance | HIGH |
| **RL agents + continuous training** | AI/ML | Orchestrator mentions RL | Not implemented | MEDIUM |
| **LLM agents with tool access** | AI | ChatGPT integration exists | Limited scope, not full agent | HIGH |
| **D3 suite (heatmaps, surfaces, regime)** | UI | Basic candlestick + SMA | Missing advanced charts | HIGH |
| **Keyboard-first, Bloomberg command bar** | UI | Implemented | Polish needed | MEDIUM |
| **OAuth2 + RBAC + API key auth** | Security | Not implemented | Missing | CRITICAL |
| **Rate limiting policy** | Security | Per-router limits exist | Not configurable/enforced | HIGH |
| **Structured logging + tracing** | Observability | Basic logging | Needs structure + tracing | HIGH |
| **CI/CD pipeline** | DevOps | Git + Docker | No automated tests/gates | CRITICAL |
| **Test coverage ≥80% critical paths** | QA | ~20% | Needs pytest infrastructure | CRITICAL |
| **Reproducible backtest validation** | Quant | Backtest works | No deterministic test suite | HIGH |

### Summary

- **CRITICAL (must complete):** Data breadth, point-in-time correctness, OAuth/auth, CI/CD, test coverage, reproducibility
- **HIGH (core system readiness):** Factor/regime, options, ML governance, LLM agents, D3 advanced charts, structured logging
- **MEDIUM (polish):** RL full integration, keyboard shortcuts refinement

---

## Phase Timeline Overview

```
Week 1:    Phase 0 — Gap lock + baseline (backlog, inventory, acceptance criteria)
Weeks 2-4: Phase 1 — Data foundation (multi-source, PIT, schemas, 10-year backfill)
Weeks 5-7: Phase 2 — Quant completion (factor, regime, options, risk, surface models)
Weeks 8-10: Phase 3 — AI/ML/RL (MLflow, model governance, LLM agents, RL training)
Weeks 11-13: Phase 4 — Terminal UI + D3 (advanced charts, keyboard polish, workspaces)
Weeks 14-15: Phase 5 — Production hardening (auth, rate limiting, logging, security)
Week 16: Phase 6 — Testing, validation, documentation, final deployment readiness
```

---

## Phase 0: Gap Lock & Baseline (Week 1)

**Goal:** Establish a single source of truth: what must be built, in what order, by whom, with what acceptance criteria.

### Deliverables

#### 0.1 Feature Backlog (Jira/GitHub Issues)

Create a master issue tracker with:
- **Epic per domain** (Data, Quant, AI, UI, Security, DevOps)
- **Story per feature** (e.g. "Fetch SEC EDGAR filings", "Implement OAuth2")
- **Task per story** (e.g. "Write EDGAR connector", "Test EDGAR endpoint")
- **Labels:** priority, phase, component, owner, acceptance criteria

**Artifact:** `docs/FEATURE_BACKLOG.md` (Markdown; can feed to Jira/Asana later)

#### 0.2 Current System Inventory

Document every module, endpoint, and capability:
- **Data layer:** DataFetcher (yfinance, FRED, Alpha Vantage), caching, refresh schedule
- **Quant modules:** Backtesting (standard + institutional), risk (VaR/CVaR), factors (if any), options (if any)
- **AI/ML:** Models trained, LLM integrations, sentiment (if any)
- **API routers:** 16 routers, 99 routes (from validation), what each does
- **Terminal UI:** Command bar, 8 modules, keyboard shortcuts, D3 charts (type/count)
- **Infrastructure:** Docker, Compose, Redis, Postgres, Prometheus, env vars

**Artifact:** `docs/SYSTEM_INVENTORY.md`

#### 0.3 Gap-to-Spec Mapping

For every requirement in context.md (23 in data, 18 in quant, 15 in AI/ML, 12 in UI, etc.):
- **Specification text** (from context.md)
- **Current implementation** (from inventory)
- **Gap description** (what's missing)
- **Acceptance test** (how we know it's done)
- **Estimated effort** (T-shirt size: S/M/L/XL)
- **Owner** (you; assign sprints later)
- **Phase** (0–6)

**Artifact:** `docs/CONTEXT_COMPLIANCE_MATRIX.md`

#### 0.4 Acceptance Criteria Per Phase

Define measurable, testable.

**Artifact:** `docs/PHASE_ACCEPTANCE_CRITERIA.md`

Example structure:
```
## Phase 1: Data Foundation — Acceptance Criteria

### 1a. Multi-Provider Connectors
- [ ] Equities/ETF: yfinance + Polygon.io (using API key)
- [ ] Futures: IEX Cloud or Polygon.io
- [ ] Options: Real data source (IEX, Intrinio, or mock for MVP)
- [ ] FX: Alpha Vantage or OANDA
- [ ] Crypto: CoinGecko or Binance (free tier)
- [ ] Test: Each source returns OHLCV with expected schema
- [ ] Test: Fallback to yfinance if primary fails

### 1b. Point-in-Time Correctness
- [ ] Dataset versioning: JSON metadata per download (timestamp, source, symbol, period)
- [ ] Reproducibility: Re-fetch same symbol/date → exact same OHLCV
- [ ] No lookahead: All indicators/models use price data up to or before signal date
- [ ] Test: Deterministic backtest run on snapshot data produces identical equity curve

### 1c. Cold Storage + Audit Trail
- [ ] SQLite or files in `data/historical/{symbol}/{date}.csv`
- [ ] `data/meta/{symbol}.json` logs source, last refresh, version
- [ ] TTL config for cache (hot) vs archival (cold)
- [ ] Test: Query 10-year backfill; data integrity check (no gaps > 1 day)

### 1d. 10-Year Backfill
- [ ] All tracked symbols (top 500 by volume + user watchlist) have 10Y OHLCV
- [ ] Macro data (FRED + 5 economic indicators) for same period
- [ ] Test: `pytest tests/data/test_backfill.py` passes for all symbols
```

### Acceptance Criteria

- ✅ All requirements from context.md mapped to a test case
- ✅ Every gap identified with owner + phase + effort estimate
- ✅ Backlog curated and prioritized (CRITICAL → HIGH → MEDIUM)
- ✅ Phase-by-phase acceptance tests defined and reviewed

---

## Phase 1: Data Foundation (Weeks 2-4)

**Goal:** Institutional-grade multi-source data layer with PIT correctness, survivorship safety, and 10-year backfill.

### Deliverables

#### 1.1 Multi-Provider Data Connectors

**Files to create/modify:**
- `core/data_providers/` (new package)
  - `__init__.py`
  - `base.py` — Abstract DataProvider class
  - `polygon_provider.py` — Equities, options, futures
  - `iex_provider.py` — Real-time + historical
  - `oanda_provider.py` — FX and commodities
  - `coingecko_provider.py` — Crypto (free tier)
  - `sec_edgar.py` — Fundamentals + filings
  - `newsapi_provider.py` — News feed
  - `alternative_provider.py` — Reddit sentiment, Google Trends stub

**API keys required:**
- Polygon.io (trial available)
- IEX Cloud (free tier has limits)
- OANDA (free demo account)
- CoinGecko (free; no key)
- NewsAPI (free; low rate limit)
- SEC EDGAR (no key)

**Specifications:**

```python
# core/data_providers/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any

class DataProvider(ABC):
    """Abstract base for all data sources."""
    
    name: str  # e.g., "Polygon", "OANDA", "CoinGecko"
    asset_types: list[str]  # ["equities", "options", "fx", "crypto"]
    lookback_limit: int  # Max days of history (e.g., 730 for ~2Y)
    rate_limit: int  # Requests per minute
    requires_api_key: bool
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        OHLCV data with schema:
        Index: DatetimeIndex
        Columns: Open, High, Low, Close, Volume (all float)
        """
        pass
    
    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Company metrics: P/E, market cap, industry, etc."""
        pass
    
    @abstractmethod
    def fetch_news(self, symbol: str, limit: int = 10) -> list[Dict]:
        """News items: [{"title": "...", "published": "...", "source": "..."}]"""
        pass
```

**Test structure:**
```python
# tests/data/test_providers.py
def test_polygon_fetch_ohlcv():
    provider = PolygonProvider(api_key=os.getenv("POLYGON_API_KEY"))
    df = provider.fetch_ohlcv("AAPL", "2015-01-01", "2025-01-01")
    assert len(df) > 2500  # ~10 years of trading days
    assert df.columns.tolist() == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert df.index.is_monotonic_increasing

def test_polygon_rate_limit():
    provider = PolygonProvider(api_key=...)
    # Test rate limiting: submit 10 requests, measure time
    # Should respect 5 req/sec or configured limit
```

#### 1.2 Unified Data Fetcher V2

**File:** `core/data_fetcher_v2.py` (or refactor existing)

```python
class UnifiedDataFetcher:
    """
    Routes requests to best provider based on asset type.
    Implements caching, fallback, and audit trail.
    """
    
    def __init__(self, config: DataConfig):
        self.providers = {
            "equities": PolygonProvider(...),  # primary
            "equities_fallback": YfinanceProvider(...),
            "options": PolygonProvider(...),
            "fx": OandaProvider(...),
            "crypto": CoinGeckoProvider(...),
            "fundamentals": SECProvider(...),
            "news": NewsAPIProvider(...),
            "sentiment": AlternativeProvider(...),
        }
        self.cache = CacheManager(ttl_hot=300, ttl_cold=3600)
        self.audit_log = AuditLogger("data_fetches.jsonl")
    
    def fetch(
        self,
        asset_type: str,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with caching and failover.
        Log all fetches to audit trail for reproducibility.
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(asset_type, symbol, start_date, end_date)
            if cached is not None:
                return cached
        
        # Try primary provider
        provider = self.providers.get(asset_type)
        try:
            df = provider.fetch_ohlcv(symbol, start_date, end_date)
            self._validate_ohlcv(df)
            self.cache.set(asset_type, symbol, start_date, end_date, df)
            self.audit_log.log(
                asset_type=asset_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider=provider.name,
                rows=len(df),
                timestamp=datetime.utcnow(),
            )
            return df
        except Exception as e:
            # Fallback
            fallback = self.providers.get(f"{asset_type}_fallback")
            if fallback:
                df = fallback.fetch_ohlcv(symbol, start_date, end_date)
                self.audit_log.log(..., provider=fallback.name, fallback=True)
                return df
            raise
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Ensure schema, no NaNs in OHLCV, monotonic index."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        assert df.columns.tolist() == required_cols
        assert df.isnull().sum().sum() == 0  # No missing data
        assert df.index.is_monotonic_increasing
        assert (df["High"] >= df["Low"]).all()  # High ≥ Low
```

#### 1.3 Point-in-Time Dataset Layer

**Files:**
- `core/datasets.py` (new)

```python
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

@dataclass
class DatasetSnapshot:
    """Immutable record of a data fetch."""
    symbol: str
    asset_type: str
    start_date: str
    end_date: str
    provider: str
    rows: int
    hash: str  # SHA256 of CSV content for reproducibility
    fetched_at: datetime
    version: str  # "1.0" for schema versioning

class DatasetManager:
    """
    Persist snapshots for reproducibility.
    Enable: "re-fetch this exact dataset from archive"
    """
    
    def __init__(self, storage_path: str = "data/snapshots"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_snapshot(
        self,
        df: pd.DataFrame,
        snapshot: DatasetSnapshot,
    ) -> str:
        """
        Save OHLCV to parquet + metadata to JSON.
        Return archive ID for reproducibility.
        """
        archive_id = f"{snapshot.symbol}_{snapshot.start_date}_{timestamp}"
        parquet_path = f"{self.storage_path}/{archive_id}.parquet"
        meta_path = f"{self.storage_path}/{archive_id}_meta.json"
        
        df.to_parquet(parquet_path)
        with open(meta_path, 'w') as f:
            json.dump(asdict(snapshot), f)
        
        return archive_id
    
    def load_snapshot(self, archive_id: str) -> tuple[pd.DataFrame, DatasetSnapshot]:
        """Load exact historical data for research reproducibility."""
        df = pd.read_parquet(f"{self.storage_path}/{archive_id}.parquet")
        with open(f"{self.storage_path}/{archive_id}_meta.json") as f:
            meta = json.load(f)
        snapshot = DatasetSnapshot(**meta)
        return df, snapshot
```

#### 1.4 Cold Storage & Audit Trail

**Files:**
- `core/storage.py` (new)

Architecture:
- **Hot:** In-memory cache (Redis) + DataFrames in process
- **Cold:** SQLite or Parquet in `data/archive/{symbol}/{year}/{month}.parquet`
- **Metadata:** `data/meta/{symbol}.json` with version, last refresh, data integrity hash

```python
class ColdStorageManager:
    """Archive historical data for long-term reproducibility."""
    
    def __init__(self, base_path: str = "data/archive"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def archive(self, symbol: str, df: pd.DataFrame, year: int, month: int) -> None:
        """
        Partition by symbol/year/month.
        Store as Parquet (efficient columnar format).
        """
        path = f"{self.base_path}/{symbol}/{year}/{month:02d}.parquet"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        
        # Update metadata
        meta_path = f"{self.base_path}/{symbol}/metadata.json"
        meta = self._load_meta(meta_path) or {
            "symbol": symbol,
            "first_date": df.index.min().strftime("%Y-%m-%d"),
            "last_date": df.index.max().strftime("%Y-%m-%d"),
            "rows": len(df),
            "refreshes": []
        }
        meta["refreshes"].append({
            "date": datetime.utcnow().isoformat(),
            "year": year,
            "month": month,
            "rows": len(df)
        })
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    
    def retrieve(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Query cold storage; reconstruct from parquets if range spans months."""
        # Logic: find all parquet files overlapping [start_date, end_date]
        # Load + concatenate + slice to requested range
        ...
```

#### 1.5 10-Year Backfill Script

**File:** `scripts/backfill_historical_data.py`

```python
#!/usr/bin/env python3
"""
One-off script: Download 10 years of OHLCV for all tracked symbols.
Run monthly or quarterly to refresh.
"""

def backfill_top500():
    """
    1. Get top 500 symbols by market cap (from Polygon or IEX)
    2. For each, fetch 2015-01-01 to 2025-01-01
    3. Store in cold storage
    4. Log progress
    """
    fetcher = UnifiedDataFetcher(config)
    symbols = fetch_top_500_by_market_cap()
    
    for symbol in symbols:
        try:
            df = fetcher.fetch(
                asset_type="equities",
                symbol=symbol,
                start_date="2015-01-01",
                end_date="2025-01-01",
            )
            cold_storage.archive(symbol, df, year=2025, month=1)
            print(f"✓ {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
    
    print(f"Backfill complete. Archive: {cold_storage.base_path}")

def backfill_macro():
    """Download FRED macro data (unemployment, GDP, etc.) for 10 years."""
    ...
```

### Acceptance Criteria for Phase 1

- ✅ All 5+ data providers implemented and tested (Polygon, IEX, OANDA, CoinGecko, NewsAPI, SEC EDGAR)
- ✅ UnifiedDataFetcher routes requests correctly; fallback tested
- ✅ Point-in-time snapshots saved and reproducible (load snapshot → same OHLCV)
- ✅ Cold storage working: 10 years of top-500 symbols + macro data archived
- ✅ Audit log shows all fetches with provider/timestamp/row count
- ✅ `pytest tests/data/ -v` passes 100% (provider tests, schema tests, fallback tests)
- ✅ No data gaps > 1 trading day in backfilled data
- ✅ Performance: Fetch 1-year OHLCV for 10 symbols in < 5s (using cache)

---

## Phase 2: Quant Engine Completion (Weeks 5-7)

**Goal:** Institutional-grade analytics with factor models, regime detection, options, and risk surfaces.

### Deliverables

#### 2.1 Factor Modeling Framework

**Files:**
- `models/factor_models.py` (new)

```python
class FactorModel:
    """Base class for factor models (Fama-French, custom)."""
    
    def __init__(self, name: str, factors: list[str]):
        self.name = name
        self.factors = factors  # e.g., ["market", "smb", "hml", "mom"]
    
    def compute_factor_returns(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,  # factor returns by date
    ) -> pd.DataFrame:
        """
        Regress asset returns on factors.
        Return: factor loadings, alpha, R², residuals.
        """
        ...
    
    def analyze_factor_exposure(
        self,
        portfolio: list[tuple[str, float]],  # [(symbol, weight), ...]
        factor_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute portfolio factor exposures.
        E.g., {"market": 1.05, "smb": 0.3, "hml": -0.1}
        """
        ...

class FamaFrenchModel(FactorModel):
    """Fama-French 3/5 factor model."""
    
    def __init__(self, frequency: str = "daily"):
        # frequency: "daily", "monthly"
        super().__init__(
            "Fama-French",
            ["market", "smb", "hml", "rmw", "cma"]
        )
        # Load FF factor data from Ken French's data library
        self.ff_data = self._load_ff_data()
    
    def _load_ff_data(self) -> pd.DataFrame:
        """
        Download Fama-French factor returns from web or local cache.
        Index: Date, Columns: [Mkt-RF, SMB, HML, RMW, CMA]
        """
        ...
```

#### 2.2 Regime Detection

**Files:**
- `models/regime_detection.py` (new)

```python
class RegimeDetector:
    """Identify market regimes: bull, bear, sideways, high-vol, etc."""
    
    @staticmethod
    def hmmstate_detection(returns: pd.Series, n_regimes: int = 3) -> pd.Series:
        """
        Use Gaussian HMM to classify regimes.
        Return: Series of regime IDs (0, 1, 2, ...).
        """
        from hmmlearn import gaussian_hmm
        X = returns.values.reshape(-1, 1)
        model = gaussian_hmm.GaussianHMM(n_components=n_regimes)
        hidden_states = model.fit(X).predict(X)
        return pd.Series(hidden_states, index=returns.index)
    
    @staticmethod
    def ewma_vol_regime(prices: pd.Series, vol_threshold: float = 0.02) -> pd.Series:
        """
        Simple regime based on rolling volatility.
        Return: Series("low_vol", "high_vol").
        """
        returns = prices.pct_change()
        vol = returns.ewm(span=20).std()
        regime = vol.apply(lambda x: "high_vol" if x > vol_threshold else "low_vol")
        return regime

class RegimeAnalyzer:
    """Analyze model/strategy performance per regime."""
    
    def analyze_strategy_by_regime(
        self,
        returns: pd.Series,
        regime: pd.Series,
    ) -> Dict[str, Dict]:
        """
        For each regime, compute Sharpe, max drawdown, win rate, etc.
        Return: {"bull": {"sharpe": 1.5, ...}, "bear": {...}}
        """
        results = {}
        for regime_id in regime.unique():
            mask = regime == regime_id
            regime_returns = returns[mask]
            results[regime_id] = {
                "sharpe": self._sharpe(regime_returns),
                "max_drawdown": self._max_drawdown(regime_returns),
                "win_rate": (regime_returns > 0).sum() / len(regime_returns),
                "return": regime_returns.sum(),
            }
        return results
```

#### 2.3 Options Pricing & Greeks

**Files:**
- `models/options_pricing.py` (new or enhance existing)

```python
class OptionsPricer:
    """Options pricing and Greeks."""
    
    @staticmethod
    def black_scholes_call(
        S: float,  # Spot price
        K: float,  # Strike
        T: float,  # Time to maturity (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
    ) -> Dict[str, float]:
        """
        Black-Scholes option price + Greeks.
        Return: {"price": ..., "delta": ..., "gamma": ..., "theta": ..., "vega": ..., "rho": ...}
        """
        import numpy as np
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
        
        return {
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
            "rho": float(rho),
        }
    
    @staticmethod
    def volatility_smile(strikes: list[float], spot: float, T: float, r: float) -> Dict:
        """
        Implied vol surface across strikes.
        (Simplified for MVP; real implementation uses market data or models.)
        """
        ...
```

#### 2.4 Risk Models: VaR, CVaR, Stress, Scenario

**Files:**
- `models/risk/var_cvar.py` (enhance existing)
- `models/risk/stress_testing.py` (new)

```python
class StressTestEngine:
    """Stress test a portfolio under extreme scenarios."""
    
    def shock_scenario(
        self,
        returns: pd.DataFrame,  # Asset returns
        weights: dict[str, float],  # Portfolio weights
        shock: Dict[str, float],  # Market shocks: {"SPY": -0.2, "VIX": +0.5}
    ) -> Dict:
        """
        Apply shocks to historical returns, recompute portfolio loss.
        Return: {"portfolio_loss": X%, "affected_assets": [...]}
        """
        # Apply shock to asset returns
        stressed_returns = returns.copy()
        for asset, shock_val in shock.items():
            if asset in stressed_returns.columns:
                stressed_returns[asset] *= (1 + shock_val)
        
        # Compute portfolio loss
        portfolio_return = (stressed_returns * pd.Series(weights)).sum(axis=1)
        loss = -portfolio_return.min()  # max loss
        
        return {
            "portfolio_max_loss": float(loss),
            "confidence": 0.95,
            "scenario": shock,
        }
    
    def correlation_regime_shift(
        self,
        returns: pd.DataFrame,
        correlation_multiplier: float = 1.5,  # Assets become more correlated
    ) -> Dict:
        """
        Stress test: increase all correlations by multiplier.
        Simulate "risk-off" events where assets move together.
        """
        ...
```

#### 2.5 Backtesting: Cross-Asset Support

**Files:**
- `core/backtesting.py` (enhance)

```python
class BacktestEngine:
    """Support equities, options, futures, crypto."""
    
    def backtest(
        self,
        signals: pd.DataFrame,  # Per-asset signals indexed by date
        price_data: Dict[str, pd.DataFrame],  # {asset: OHLCV}
        portfolio: Dict[str, float],  # {asset: position_size_pct}
        asset_types: Dict[str, str],  # {asset: "equity"/"option"/"future"}
        transaction_cost: float = 0.0005,
        slippage: float = 0.001,
    ) -> BacktestResult:
        """
        Backtest across multiple asset types.
        Calculate PnL accounting for transaction costs + slippage.
        """
        ...
```

### Acceptance Criteria for Phase 2

- ✅ Factor model: Fama-French 3/5 factors computed; portfolio exposures match benchmark
- ✅ Regime detection: HMM-based regimes identified; performance by regime computed
- ✅ Options: Black-Scholes pricing + Greeks; test against known benchmarks (e.g., Haug 2007)
- ✅ Stress testing: Scenario shocks applied; portfolio loss calculated deterministically
- ✅ Backtest: Works with equities + options + crypto; transaction costs applied
- ✅ `pytest tests/quant/ -v` passes 100%
- ✅ API endpoints wired: `/api/v1/risk/factors`, `/api/v1/risk/stress`, `/api/v1/quant/regime`

---

## Phase 3: AI/ML/RL Pipeline (Weeks 8-10)

**Goal:** Production-grade ML governance, LLM agents, and RL strategy training.

### Deliverables

#### 3.1 MLflow Integration

**Files:**
- `ml/mlflow_setup.py` (new)
- `ml/model_registry.py` (new)

```python
import mlflow
from mlflow.models import infer_signature
import os

def setup_mlflow():
    """Initialize MLflow tracking and artifact storage."""
    mlflow.set_tracking_uri("file:./mlruns")  # Or remote server
    mlflow.set_experiment("financial-models")

def train_and_register_model(
    model_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_class,
    hyperparams: Dict,
):
    """
    Train model, log metrics, artifacts, and register in MLflow registry.
    """
    mlflow.start_run(tags={"model": model_name, "phase": "development"})
    
    # Train
    model = model_class(**hyperparams)
    model.fit(train_data)
    
    # Evaluate
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)
    train_rmse = mean_squared_error(train_data["y"], train_preds) ** 0.5
    test_rmse = mean_squared_error(test_data["y"], test_preds) ** 0.5
    
    # Log to MLflow
    mlflow.log_params(hyperparams)
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    })
    mlflow.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(train_data, train_preds),
        registered_model_name=model_name,
    )
    
    mlflow.end_run()
    
    # Register in Model Registry
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Staging",  # or "Archived", "Production"
    )
    
    print(f"Model {model_name} v1 registered and staged.")

class ModelRegistry:
    """Manage model lifecycle: development → staging → production."""
    
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()
    
    def promote_to_production(self, model_name: str, version: int):
        """Move model version from Staging to Production."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
    
    def auto_rollback_on_degradation(self, model_name: str, metric_threshold: float):
        """
        If production model's test metric drops below threshold,
        rollback to previous version.
        """
        # Logic: compare latest production metric vs previous version
        # If degraded, promote previous version back to production
        ...
```

#### 3.2 Feature Store

**Files:**
- `ml/feature_store.py` (new)

```python
class FeatureStore:
    """Centralized feature engineering and versioning for ML models."""
    
    def __init__(self, storage_path: str = "data/features"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def compute_features(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Compute technical features (MA, RSI, volatility, etc.)
        and save to feature store.
        """
        features = pd.DataFrame(index=ohlcv.index)
        features["sma_20"] = ohlcv["Close"].rolling(20).mean()
        features["sma_50"] = ohlcv["Close"].rolling(50).mean()
        features["rsi_14"] = self._rsi(ohlcv["Close"], 14)
        features["volatility"] = ohlcv["Close"].pct_change().rolling(20).std()
        features["volume_ma"] = ohlcv["Volume"].rolling(20).mean()
        # Add many more features...
        
        # Save
        feature_path = f"{self.storage_path}/{symbol}_features.parquet"
        features.to_parquet(feature_path)
        
        return features
    
    def get_features(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve features from store, recompute if missing."""
        feature_path = f"{self.storage_path}/{symbol}_features.parquet"
        if os.path.exists(feature_path):
            return pd.read_parquet(feature_path).loc[start_date:end_date]
        else:
            ohlcv = fetcher.fetch("equities", symbol, start_date, end_date)
            return self.compute_features(ohlcv, symbol, start_date, end_date)
```

#### 3.3 LLM-Powered Research Agent

**Files:**
- `ai/llm_agent.py` (new or enhance)

```python
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.chat_models import ChatOpenAI
import json

class FinancialResearchAgent:
    """
    LLM agent that can:
    - Query data (fetch prices, fundamentals, macro)
    - Run analysis (backtest, risk metrics, sentiment)
    - Explain findings in natural language
    """
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0.2)
        self.fetcher = UnifiedDataFetcher(config)
        self.backtester = BacktestEngine()
        self.risk_analyzer = RiskAnalyzer()
        
        # Define tools for agent
        self.tools = [
            Tool(
                name="fetch_price_data",
                func=self._fetch_price,
                description="Fetch historical OHLCV data for a symbol.",
            ),
            Tool(
                name="fetch_fundamentals",
                func=self._fetch_fundamentals,
                description="Get company fundamentals: P/E, market cap, etc.",
            ),
            Tool(
                name="backtest_strategy",
                func=self._backtest,
                description="Backtest a trading strategy and return metrics.",
            ),
            Tool(
                name="analyze_risk",
                func=self._analyze_risk,
                description="Compute VaR, volatility, max drawdown, etc.",
            ),
            Tool(
                name="fetch_news_sentiment",
                func=self._fetch_sentiment,
                description="Get recent news and sentiment score for a symbol.",
            ),
        ]
        
        self.agent = initialize_agent(
            agent="zero-shot-react-description",
            tools=self.tools,
            llm=self.llm,
            max_iterations=5,
            verbose=True,
        )
    
    def run_query(self, query: str) -> str:
        """
        User asks question → agent decides which tools to call → returns analysis.
        Example: "Should I buy Microsoft? What's the risk?"
        """
        result = self.agent.run(query)
        return result
    
    def _fetch_price(self, symbol: str) -> str:
        """Tool: fetch price data."""
        df = self.fetcher.fetch("equities", symbol, "2024-01-01", "2025-01-01")
        return json.dumps({
            "symbol": symbol,
            "latest_close": float(df["Close"].iloc[-1]),
            "1y_return": float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100),
            "volatility": float(df["Close"].pct_change().std() * np.sqrt(252)),
        })
    
    def _fetch_fundamentals(self, symbol: str) -> str:
        """Tool: fetch company fundamentals."""
        info = self.fetcher.fetch_fundamentals(symbol)
        return json.dumps({
            "symbol": symbol,
            "pe_ratio": info.get("pe_ratio"),
            "market_cap": info.get("market_cap"),
            "industry": info.get("industry"),
        })
    
    def _backtest(self, strategy_name: str, symbol: str) -> str:
        """Tool: run backtest."""
        # Logic: load pre-defined strategy, run backtest
        result = self.backtester.backtest(...)
        return json.dumps({
            "strategy": strategy_name,
            "symbol": symbol,
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
        })
    
    def _analyze_risk(self, symbol: str) -> str:
        """Tool: risk analysis."""
        risk = self.risk_analyzer.analyze(symbol)
        return json.dumps({
            "symbol": symbol,
            "var_95": risk.var_95,
            "cvar_95": risk.cvar_95,
            "max_drawdown": risk.max_drawdown,
        })
    
    def _fetch_sentiment(self, symbol: str) -> str:
        """Tool: get news sentiment."""
        sentiment = self.fetcher.fetch_sentiment(symbol)
        return json.dumps({
            "symbol": symbol,
            "sentiment_score": sentiment.score,
            "recent_news": sentiment.headlines[:3],
        })
```

#### 3.4 Reinforcement Learning: Strategy Training

**Files:**
- `models/rl/trading_env.py` (new)
- `models/rl/ppo_trainer.py` (new)

```python
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

class TradingEnvironment(gym.Env):
    """
    Gymnasium environment for RL training.
    State: current portfolio + market data
    Actions: buy/hold/sell
    Reward: trading P&L
    """
    
    def __init__(self, prices: np.ndarray, initial_cash: float = 10000):
        self.prices = prices
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0  # 0: no position, 1: long, -1: short
        self.current_step = 0
        
        # Observation: [price, sma_20, rsi, cash, position]
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,))
        # Action: [0: hold, 1: buy, 2: sell]
        self.action_space = gym.spaces.Discrete(3)
    
    def reset(self, seed=None):
        self.cash = self.initial_cash
        self.position = 0
        self.current_step = 0
        return self._get_obs(), {}
    
    def step(self, action):
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            # Buy at current price
        elif action == 2 and self.position == 1:  # Sell
            profit = self.prices[self.current_step] * 100 - self.prices[self.current_step - 1] * 100
            self.cash += profit
            self.position = 0
        
        # Reward: change in portfolio value
        portfolio_value = self.cash + self.position * self.prices[self.current_step] * 100
        reward = portfolio_value - (self.initial_cash + self.cash)
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        price = self.prices[self.current_step]
        return np.array([price, price, 50, self.cash, self.position], dtype=np.float32)

def train_rl_agent(symbol: str, data: pd.DataFrame, total_timesteps: int = 100000):
    """
    Train PPO agent on trading data.
    Save model to registry.
    """
    env = TradingEnvironment(data["Close"].values)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    
    # Save and register
    model_name = f"rl_ppo_{symbol}"
    model.save(f"ml/models/{model_name}")
    mlflow.log_model(model, artifact_path=f"rl/{model_name}")
    print(f"RL model {model_name} trained and saved.")
    
    return model
```

### Acceptance Criteria for Phase 3

- ✅ MLflow: Models logged with metrics, artifacts, params; registry operational
- ✅ Feature store: Features computed and retrieved; versioning working
- ✅ LLM agent: Can run 5 tools (fetch, backtest, risk, sentiment, etc.); queries answered
- ✅ RL training: PPO agent trains on trading environment; policies improve over episodes
- ✅ Auto-rollback: Model degradation detected; previous version promoted
- ✅ `pytest tests/ml/ -v` passes 100%
- ✅ API endpoints: `/api/v1/ai/agent-query`, `/api/v1/ml/models`, `/api/v1/rl/train`

---

## Phase 4: Terminal UI & D3 Visualization (Weeks 11-13)

**Goal:** Complete D3 suite + Bloomberg-level terminal UX.

### Deliverables

#### 4.1 Advanced D3 Chart Library

**Files:**
- `frontend/src/components/charts/` (new/enhance)
  - `CandlestickChart.tsx` (exists; polish)
  - `HeatmapChart.tsx` (new)
  - `VolatilitySurfaceChart.tsx` (new)
  - `CorrelationMatrix.tsx` (new)
  - `FactorExposureChart.tsx` (new)
  - `RegimeShiftChart.tsx` (new)

```typescript
// frontend/src/components/charts/VolatilitySurfaceChart.tsx

import * as d3 from "d3";
import { useEffect, useRef } from "react";

interface VolatilitySurfaceProps {
  strikes: number[];
  maturities: string[];  // e.g., ["1M", "3M", "6M", "1Y"]
  volatilities: number[][];  // 2D array
  spotPrice: number;
}

export function VolatilitySurfaceChart({
  strikes,
  maturities,
  volatilities,
  spotPrice,
}: VolatilitySurfaceProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !volatilities.length) return;

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 30, bottom: 30, left: 60 };

    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Flatten 2D data for heatmap
    const data = [];
    for (let i = 0; i < strikes.length; i++) {
      for (let j = 0; j < maturities.length; j++) {
        data.push({
          strike: strikes[i],
          maturity: maturities[j],
          vol: volatilities[i][j],
        });
      }
    }

    const xScale = d3.scaleBand()
      .domain(maturities)
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleBand()
      .domain(strikes.map(String))
      .range([margin.top, height - margin.bottom]);

    const colorScale = d3.scaleSequential()
      .domain(d3.extent(data, (d) => d.vol) as [number, number])
      .interpolator(d3.interpolateYlOrRd);

    // Draw heatmap
    svg
      .selectAll(".cell")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "cell")
      .attr("x", (d) => xScale(d.maturity)!)
      .attr("y", (d) => yScale(String(d.strike))!)
      .attr("width", xScale.bandwidth())
      .attr("height", yScale.bandwidth())
      .attr("fill", (d) => colorScale(d.vol));

    // Axes
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    // Title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", margin.top - 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "bold")
      .text(`Volatility Surface (Spot: $${spotPrice})`);
  }, [strikes, maturities, volatilities, spotPrice]);

  return <svg ref={svgRef} />;
}
```

**Heatmap, correlation matrix, factor exposure, regime shift: similar structure.**

#### 4.2 Terminal Panels: Full D3 Integration

**Files:**
- `frontend/src/pages/Terminal.tsx` (enhance)

Update existing panels to use D3:

- **Primary Instrument:** Candlestick + volume (✅ exists; polish)
- **Fundamental:** Peer comparison table + D3 waterfall (valuation breakdown)
- **Technical:** Indicators on candlestick (✅ exists; add more indicators)
- **Quant:** Factor exposure heatmap, regime timeline, backtest metrics
- **Economic:** Macro indicators (FRED) as time-series; yield curve
- **News:** Sentiment timeline for symbol
- **Portfolio:** Pie/treemap for allocation, scatter for risk/return
- **Screening:** Results grid + multi-factor scatter

#### 4.3 Command Bar Enhancements

**Files:**
- `frontend/src/components/CommandBar.tsx` (enhance)

```typescript
// Support more advanced commands:
// GP AAPL      → Primary (candlestick)
// FA AAPL      → Fundamental (DCF, ratios, peer comp)
// FLDS AAPL    → Technical (indicators, chart tools)
// QUANT AAPL   → Models, backtest, regime
// ECO          → Economic dashboard (macro)
// NEWS AAPL    → News + sentiment for AAPL
// PORT AAPL MSFT GOOGL    → Portfolio analysis
// SCREEN       → Advanced screener with factors
// BACKTEST 50SMA.50EMA    → Run backtest with specific rules
// AI AAPL      → AI assistant (query agent)
// WORKSPACE    → Manage saved layouts
// ? or HELP    → Command help

interface CommandContext {
  module: ModuleType;
  primarySymbol: string;
  secondarySymbols: string[];  // For portfolio/screener
  parameters: Record<string, string>;  // e.g., { indicator: "RSI", period: "14" }
}

export function CommandBar() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const { setActiveModule, setPrimarySymbol } = useTerminalContext();

  const handleSubmit = (cmd: string) => {
    const context = parseCommand(cmd);
    if (context) {
      setActiveModule(context.module);
      setPrimarySymbol(context.primarySymbol);
      // Pass parameters to module for indicators, backtest params, etc.
    }
  };

  return (
    <div className="command-bar">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSubmit(input);
          if (e.key === "ArrowUp") setInput(history[0] || "");
        }}
        placeholder="Command: GP AAPL | FA | BACKTEST | AI | ?"
      />
    </div>
  );
}
```

#### 4.4 Workspace Persistence

**Files:**
- `frontend/src/context/WorkspaceContext.tsx` (enhance)

```typescript
interface WorkspaceState {
  name: string;
  activeModule: ModuleType;
  primarySymbol: string;
  layout: {
    primarySize: number;  // % of screen
    leftSize: number;
    rightSize: number;
  };
  panelSettings: Record<ModuleType, Record<string, any>>;  // Module-specific settings
  timestamp: number;
}

export function useWorkspace() {
  const [workspaces, setWorkspaces] = useState<WorkspaceState[]>(() => {
    const saved = localStorage.getItem("workspaces");
    return saved ? JSON.parse(saved) : [];
  });

  const saveWorkspace = (state: WorkspaceState) => {
    const updated = [
      ...workspaces.filter((w) => w.name !== state.name),
      state,
    ];
    setWorkspaces(updated);
    localStorage.setItem("workspaces", JSON.stringify(updated));
  };

  const loadWorkspace = (name: string) => {
    const workspace = workspaces.find((w) => w.name === name);
    return workspace;
  };

  return { workspaces, saveWorkspace, loadWorkspace };
}
```

### Acceptance Criteria for Phase 4

- ✅ All 6 D3 chart types implemented (candlestick, heatmap, vol surface, correlation, factor exposure, regime)
- ✅ 8 terminal modules fully functional with D3 visuals
- ✅ Command bar: parses 15+ commands correctly
- ✅ Keyboard shortcuts: Alt+Left/Right, Esc, ?, / all work
- ✅ Workspaces: Save/load layouts; persistent across sessions
- ✅ Performance: < 2s load time; zoom/pan smooth
- ✅ Frontend bundle: No >100KB chunks above the fold
- ✅ E2E test: Load terminal → type "FA AAPL" → fundamental panel renders with D3 charts

---

## Phase 5: Production Hardening (Weeks 14-15)

**Goal:** Security, reliability, observability, and deployment readiness.

### Deliverables

#### 5.1 Authentication & Authorization

**Files:**
- `api/auth.py` (new/enhance)
- `api/middleware.py` (enhance)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
import jwt
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-key-change-in-prod")
ALGORITHM = "HS256"

security = HTTPBearer()

class TokenManager:
    @staticmethod
    def create_token(user_id: str, expires_delta: timedelta = None):
        """Create JWT token."""
        if expires_delta is None:
            expires_delta = timedelta(hours=1)
        
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() + expires_delta,
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return token
    
    @staticmethod
    def verify_token(token: str):
        """Verify JWT and return user_id."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)):
    """Dependency: verify user from Bearer token."""
    user_id = TokenManager.verify_token(credentials.credentials)
    return user_id

@router.post("/auth/login")
async def login(username: str, password: str):
    """Authenticate and return JWT."""
    # Check credentials (use bcrypt + DB in production)
    if username == "demo" and password == "demo":  # Dummy for demo
        token = TokenManager.create_token(user_id=username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/api/v1/protected-endpoint")
async def protected(user: str = Depends(get_current_user)):
    """Example protected endpoint."""
    return {"user": user, "data": "sensitive"}
```

#### 5.2 Rate Limiting & Quota Management

**Files:**
- `api/rate_limit.py` (enhance)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# Define rate limits per endpoint
RATE_LIMITS = {
    "/api/v1/backtest/run": "10 per hour",  # Expensive
    "/api/v1/ai/agent-query": "20 per hour",
    "/api/v1/data/quotes": "100 per minute",
    "/api/v1/health": "1000 per minute",
}

@app.post("/api/v1/backtest/run")
@limiter.limit("10 per hour")
async def backtest_run(request: Request, ...):
    """Rate limited to 10 backtests per hour per IP."""
    ...
```

#### 5.3 Structured Logging & Tracing

**Files:**
- `api/logging_config.py` (new)
- `api/instrumentation.py` (new)

```python
import logging
import json
from datetime import datetime
import uuid

class StructuredLogger:
    """JSON structured logging for production."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # JSON formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_request(self, method: str, path: str, user: str, request_id: str):
        """Log incoming request."""
        self.logger.info(json.dumps({
            "event": "request",
            "method": method,
            "path": path,
            "user": user,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }))
    
    def log_response(self, status_code: int, duration_ms: float, request_id: str):
        """Log outgoing response."""
        self.logger.info(json.dumps({
            "event": "response",
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        }))

# Middleware to inject request tracing
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger = StructuredLogger(__name__)
    logger.log_request(request.method, request.url.path, "anonymous", request_id)
    
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    
    logger.log_response(response.status_code, duration_ms, request_id)
    return response
```

#### 5.4 Prometheus Metrics & Alerts

**Files:**
- `api/metrics.py` (new)
- `prometheus.yml` (new config)

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
request_count = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

request_latency = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
)

active_connections = Gauge(
    "api_active_connections",
    "Number of active WebSocket connections",
)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

# Usage in endpoints:
@app.get("/api/v1/data/quotes")
async def quotes(...):
    start = time.time()
    try:
        data = fetch_quotes(...)
        request_count.labels(method="GET", endpoint="/quotes", status=200).inc()
        request_latency.labels(endpoint="/quotes").observe(time.time() - start)
        return data
    except Exception as e:
        request_count.labels(method="GET", endpoint="/quotes", status=500).inc()
        raise
```

**Prometheus config:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "api"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

#### 5.5 Security Scanning & Dependency Audit

**Files:**
- `.github/workflows/security.yml` (new)

```yaml
name: Security Checks

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Python dependency audit
        run: |
          pip install safety
          safety check --json > safety-report.json || true
      
      - name: SAST (Bandit)
        run: |
          pip install bandit
          bandit -r api/ core/ models/ -f json -o bandit-report.json || true
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json
```

### Acceptance Criteria for Phase 5

- ✅ OAuth2/JWT: Login → token issued; protected endpoints require token
- ✅ Rate limiting: Backtest limited to 10/hour; enforced across requests
- ✅ Structured logging: All requests/responses logged as JSON with request ID
- ✅ Prometheus metrics: API up, latency histogram, active connections tracked
- ✅ Security scans: No high/critical CVEs; pass SAST checks
- ✅ CORS: Configured for known origins; preflight handled
- ✅ Secrets: No API keys in code; all in `.env`; docs warn against committing `.env`

---

## Phase 6: Testing & Documentation (Week 16)

**Goal:** Comprehensive test coverage, documentation, and deployment validation.

### Deliverables

#### 6.1 Test Coverage

**Files:**
- `tests/` — Organize by domain:
  - `tests/data/` — Data providers, fetcher, cold storage
  - `tests/quant/` — Backtesting, risk, factors, options
  - `tests/ml/` — Model training, MLflow, feature store
  - `tests/api/` — Endpoints, auth, rate limiting
  - `tests/e2e/` — Full workflows (terminal → backtest, etc.)

**Target coverage:**
- Critical paths (config, backtesting, risk): ≥ 85%
- API endpoints: ≥ 80%
- UI/utils: ≥ 70%
- Overall: ≥ 75%

#### 6.2 Documentation

**Files to create/update:**
- `README.md` — Quick start (already exists; update with new features)
- `ARCHITECTURE.md` — Keep current
- `API_DOCUMENTATION.md` — Add new endpoints (factor models, options, LLM agent, RL, auth)
- `WORKFLOWS.md` — Keep current; add RL, LLM workflows
- `FINAL_DEVELOPMENT_PLAN.md` — Update status (mark phases complete)
- `DEPLOYMENT_GUIDE.md` — How to deploy to production (Docker, K8s, env setup)
- `SECURITY.md` (new) — Auth, secrets, rate limiting, security best practices
- `DATASET_DOCUMENTATION.md` (new) — Data sources, versioning, reproducibility
- `MODEL_REGISTRY.md` (new) — MLflow model management, promotion workflow
- `TROUBLESHOOTING.md` (new) — Common issues, logs, debugging tips

#### 6.3 Deployment Validation

**Checklist:**

```
Pre-Deployment:
- [ ] All tests passing (pytest with coverage report)
- [ ] No high/critical security findings
- [ ] Type checking: mypy + ESLint with no errors
- [ ] API docs complete: OpenAPI schema valid
- [ ] Secrets rotation: new API keys for prod
- [ ] Performance benchmarks: < 2s load time, < 200ms API latency p95
- [ ] Database migrations: schema up-to-date
- [ ] Cache warm-up: backfill symbols for fast first load
- [ ] Monitoring: Prometheus/Grafana dashboards created
- [ ] Alerts: Critical endpoints monitored; alert thresholds set

Deployment (Staging):
- [ ] Docker build + image pushed to registry
- [ ] docker-compose up: all services start
- [ ] Health checks pass: /health, /info
- [ ] Data endpoints respond: /quotes, /macro, /health-check
- [ ] Auth works: login → obtain token; protected endpoints reject without token
- [ ] Rate limiting works: excess requests rejected
- [ ] Logging visible: structured logs in stdout
- [ ] Metrics collected: Prometheus scrapes successfully
- [ ] WebSocket connects: /ws/prices/{symbol} streams data
- [ ] Terminal UI loads: frontend served, connects to backend

Production Deployment:
- [ ] Blue-green swap: old → new traffic shifted
- [ ] Smoke tests: critical workflows (login, backtest, risk metrics) pass
- [ ] Rollback plan ready: switch back to blue if issue
- [ ] Monitoring alerts active: CPU, memory, error rates, latency
- [ ] Logs aggregated: all services' logs centralized (ELK, Datadog, etc.)
- [ ] SLA tracking: 99.9% uptime target established
```

#### 6.4 Final Integration Test

**File:** `tests/e2e/test_end_to_end.py`

```python
"""
End-to-end workflow tests.
Require running backend + frontend; can use docker-compose.
"""

def test_daily_macro_snapshot():
    """
    Workflow: User goes to Economic panel → sees macro indicators.
    1. Fetch FRED data
    2. Display time-series
    3. Assert data freshness
    """
    ...

def test_backtest_workflow():
    """
    Workflow: User runs backtest → sees results.
    1. Login
    2. POST /backtest/run with strategy
    3. GET /backtest/{id} → results with metrics
    4. Assert Sharpe, max drawdown, equity curve
    """
    ...

def test_factor_analysis():
    """
    Workflow: User analyzes portfolio factor exposures.
    1. POST /portfolio with symbols/weights
    2. GET /factors → exposures
    3. Assert factor keys match Fama-French
    """
    ...

def test_llm_agent_query():
    """
    Workflow: User asks question → LLM agent returns analysis.
    1. POST /ai/agent-query with question
    2. Agent calls tools (fetch, backtest, risk)
    3. Assert response contains analysis
    """
    ...
```

### Acceptance Criteria for Phase 6

- ✅ Test coverage ≥ 75% overall; ≥ 85% critical paths
- ✅ All API endpoints documented in OpenAPI schema
- ✅ User guide + operator runbook created
- ✅ E2E tests: 5+ critical workflows pass
- ✅ Deployment checklist: 100% items checked off
- ✅ Monitoring: Dashboards + alerts configured
- ✅ No open security issues (SAST + dependency audit)
- ✅ Ready for production: can deploy today

---

## Deployment Gates & Acceptance

### Gate 1: Code Quality (Phase 0 → 1)
- ✅ Backlog defined + prioritized
- ✅ Acceptance criteria for each phase written
- ✅ Repo structure organized by domain
- **Proceed if:** Backlog + criteria reviewed and approved

### Gate 2: Data (Phase 1 → 2)
- ✅ Multi-provider connectors working (5+ sources)
- ✅ 10-year backfill complete for top-500 symbols + macro
- ✅ Point-in-time reproducibility verified
- ✅ Cold storage operational
- **Proceed if:** Data integrity tests pass 100%

### Gate 3: Quant (Phase 2 → 3)
- ✅ Backtest validation: known price series → expected PnL
- ✅ Risk models: VaR/CVaR match benchmarks
- ✅ Factors: Fama-French exposures computed
- ✅ Options: Greeks match Black-Scholes references
- **Proceed if:** Quant tests pass 100% + no numerical discrepancies

### Gate 4: AI/ML (Phase 3 → 4)
- ✅ MLflow: Models logged, registered, promoted
- ✅ Feature store: Features versioned, reproducible
- ✅ LLM agent: 5+ tools callable; queries answered
- ✅ RL training: Agent improves over episodes
- **Proceed if:** Model governance + agent works end-to-end

### Gate 5: UI (Phase 4 → 5)
- ✅ All 6 D3 charts rendered (no basic plots)
- ✅ Terminal: 8 modules fully functional
- ✅ Command bar: Parses all major commands
- ✅ Workspaces: Save/load working
- **Proceed if:** UI load time < 2s; no React errors in console

### Gate 6: Production (Phase 5 → 6)
- ✅ Auth working: JWT issued, protected endpoints secured
- ✅ Rate limiting: Enforced per endpoint
- ✅ Logging: Structured JSON logs visible
- ✅ Security: No CVEs; SAST pass
- **Proceed if:** All security checks green; deployment readiness checklist 100%

### Final Acceptance (Phase 6 Complete)
- ✅ Test coverage ≥ 75%; critical paths ≥ 85%
- ✅ All 6 features from context.md addressed:
  - Data: ✅ Multi-source, PIT-safe, 10-year, cold storage
  - Quant: ✅ Factors, regime, options, risk, cross-asset backtest
  - AI/ML: ✅ MLflow, features, LLM agents, RL training
  - UI: ✅ D3 charts, command bar, Bloomberg-style terminal
  - Security: ✅ OAuth, rate limiting, logging, metrics
  - DevOps: ✅ CI/CD, tests, docs, deployment guides
- ✅ **System is production-ready and Bloomberg-terminal-level**

---

## Summary: 16-Week Roadmap

| Week | Phase | Deliverable | Effort | Owner |
|------|-------|-------------|--------|-------|
| 1 | Phase 0 | Backlog, acceptance criteria, gap matrix | M | You |
| 2–4 | Phase 1 | Data layer: 5+ providers, PIT, cold storage, backfill | L | You |
| 5–7 | Phase 2 | Quant: factors, regime, options, stress, cross-asset backtest | L | You |
| 8–10 | Phase 3 | AI/ML: MLflow, features, LLM agents, RL training | L | You |
| 11–13 | Phase 4 | Terminal: 6 D3 charts, command bar, workspaces, polish | L | You |
| 14–15 | Phase 5 | Security: OAuth, rate limiting, logging, metrics, scanning | M | You |
| 16 | Phase 6 | Tests, docs, validation, deployment readiness | M | You |

**Total: 16 weeks. Fully institutional, production-grade system. Bloomberg level.**

---

Now I'm ready to start Phase 0. Want me to begin with the Feature Backlog?