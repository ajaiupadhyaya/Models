# Models (FIN-TERMINAL) — Project Rundown

**Audit date:** 2026-07-08
**Commit audited:** `8796d4d4` (`main`, clean tree, nothing unpushed)
**Live:** https://models-terminal.fly.dev — up and healthy
**Method:** every claim was checked against code, a live test run, or an HTTP probe against production. Where a doc contradicted the code, the code wins. The two headline bugs were reproduced with runnable scripts (Appendix A). Nothing was modified, committed, or pushed.

---

## 1. Verdict

The engineering is real. The frontend genuinely calls the backend, the backend genuinely computes from live market data, the endpoint contract between them is exact, 374 backend tests and 24 frontend tests pass, and the app is deployed and serving traffic. This is not a demo skeleton.

But two of the numbers it shows users are **wrong, not approximate**:

- **The backtest engine subtracts profits from equity.** Run a long-only strategy on a market that rises 59%, with zero commission, and the terminal's `BACKTEST` command reports **−11.8%**. Reproduced below.
- **The portfolio optimizer recommends the worst available portfolio,** because it subtracts an annual risk-free rate from a daily mean return, which inverts the objective into "maximize volatility." Reproduced below.

Neither is covered by a test. The backtest engine *is* tested — four tests, all of which assert only that keys exist and that equity is non-negative. Not one asserts that a winning strategy makes money.

Separately: production has 5 of the ~12 secrets it needs, so roughly half the panels are dark for anyone who opens the link; ~21,000 lines (32%) of the codebase has no production importer; and the published `v1.0.0` tag shares **no git history at all** with `main`.

Rough completion against the project's own v1.0 definition: **~75%.** The remaining work is concentrated in correctness and operations, not in building new features.

---

## 2. What the project is

A Bloomberg-terminal-style single-user web app for personal quant research.

| Layer | Stack | Size |
|---|---|---|
| Backend | FastAPI, 19 domain routers + auth, APScheduler | 10,048 lines (`api/`) |
| Domain logic | data fetchers, backtest engines, AI service, DB layer | 21,992 lines (`core/`) |
| Quant library | risk, options, factors, ML, RL, NLP, valuation, macro | 15,774 lines (`models/`) |
| Frontend | React 18 + Vite + TypeScript, hand-written D3 charts | 8,619 lines (`frontend/`) |
| Tests | pytest + vitest | 7,337 lines (`tests/`) |

338 tracked files, ~66k lines. Deployed as a single Fly.io container (512 MB, `iad`, scale-to-zero) that serves the built SPA same-origin from FastAPI.

---

## 3. Feature inventory — what works

### 3.1 Backend — 122 live routes, all 19 routers mounted

Verified against `https://models-terminal.fly.dev/openapi.json` (122 paths) and by importing the app locally. Routers register at `api/main.py:404-450`.

| Prefix | Routes | State |
|---|---|---|
| `/api/v1/risk` | 14 | VaR/CVaR/vol/max-DD, stress tests, options Greeks, factor IC — real |
| `/api/v1/monitoring` | 10 | Real |
| `/api/v1/data` | 10 | Quotes, macro, yield curve, correlation, news, calendar — real |
| `/api/v1/paper-trading` | 9 | Real code; unusable in prod (no Alpaca keys) |
| `/api/v1/predictions` | 9 | Real code; JWT-gated; ARIMA/tsfresh/sentiment paths broken (§5.6) |
| `/api/v1/ai` | 8 | Real code; dark in prod (no OpenAI key) |
| `/api/v1/orchestrator` | 8 | Real |
| `/api/v1/backtest` | 6 | Standard engine **inverted** (§5.1); institutional engine correct |
| `/api/v1/company` | 6 | Real — live yfinance fundamentals |
| `/api/v1/equity` | 6 | Real — search, statements, comps, genuine DCF + LBO math |
| `/api/v1/reports` | 6 | Real |
| `/api/v1/comprehensive` | 5 | Real, but surfaces a fake `ml_predicted_volatility` (§5.3) |
| `/api/v1/quant` | 5 | `backtest` inverted (§5.1); `pairs` silently fake (§5.4); `options-chain` 500s (§5.5) |
| `/api/v1/models` | 4 | Real, but unreachable from the browser (§5.7) |
| `/api/auth` | 4 | Real — JWT, single-user |
| `/api/v1/automation` | 4 | Real; contains the only live-order code path (§6.1) |
| `/api/v1/institutional` | 4 | Real |
| `/api/v1/ws` | 3 | Real WebSocket price stream |
| `/api/v1/news` | 1 | Real code; dark in prod (no keys) |
| `/api/v1/screener` | 1 | Needs Postgres; errors in prod (§5.8) |

The synthetic-data path, `api/fallback_market_data.py`, is **honest**: deterministic sha256-seeded candles, every response tagged `"source": "fallback"` with a warning the frontend renders as a visible banner. That design is good and should be kept. (The one place data *is* fabricated undisclosed is §5.3.)

### 3.2 Frontend — 18 panels, 7 D3 charts, exact endpoint contract

Every panel fetches a live endpoint. No `Math.random()`, no mock arrays, no "Coming soon". **Every frontend endpoint resolves to a mounted backend route with a matching HTTP method** — zero code-level 404s or 405s.

Panels: Primary Instrument, Fundamental, Technical, Quant, Economic, News, News Sentiment, Portfolio, Optimizer, Stress Test, Screening, Backtest, Paper Trading, Automation, Data Status, AI Assistant, AI Insights, Market Overview.

Charts (all real D3, data via props): `AreaChart`, `BarChart`, `Heatmap`, `TimeSeriesLine`, `XyLineChart`, `YieldCurve`, `CandlestickVolume`.

Indicators in `PrimaryInstrument` (SMA, EMA, RSI, MACD, Bollinger, ATR) are genuine client-side math over real fetched OHLC bars.

Also real: JWT login, `ProtectedRoute`, an `ErrorBoundary` class component, per-panel `PanelErrorState` with retry, a `/health` gate that shows "API unreachable" rather than a blank page, and localStorage workspace persistence (layout, watchlist, command history, active module).

### 3.3 Command bar — 18 commands

`DATA`/`STATUS`, `GP`, bare ticker, `FA`, `FLD`/`FLDS`, `ECO`, `N`, `NS`/`SENT`, `PORT`, `OPT`, `STRESS`, `PAPER`, `AUTO`/`ORCH`, `SCREEN`, `AI`, `BACKTEST`/`BT`, `TRAIN`/`QUANT`, `WORKSPACE`, `?`/`HELP`. Unrecognized input routes to the AI panel. History persists.

### 3.4 Quant library — the math is mostly real

Genuinely implemented and reachable from the API: Black-Scholes with full Greeks and IV via `brentq`; CRR binomial and SABR with real calibration; scipy SLSQP portfolio optimizers; historical/parametric/Monte-Carlo VaR and CVaR; 11 curated crisis stress scenarios; Fama-French OLS; Almgren-Chriss transaction costs; bootstrap and permutation statistical validation; Black-Litterman; real Altman/Piotroski/Beneish scores from live fundamentals; a real DCF with a sensitivity grid; sklearn ensemble prediction.

**No module uses `np.random` to fabricate results that pretend to be model output.** Every `np.random` use is legitimate (bootstrap CIs, Monte-Carlo VaR, epsilon-greedy exploration, stochastic slippage).

Three things are mislabeled rather than fake — see §5.3 and §6.2.

### 3.5 Gates that currently pass

- Backend: **374 passed, 17 skipped** in ~30 s.
- Frontend: **24 passed** (3 files), `tsc --noEmit` clean, `vite build` clean.
- `ruff check config/ api/ core/` — clean.
- CI (GitHub Actions): backend tests, ruff, frontend typecheck + test. Green on `main`.

Read §7 before drawing comfort from those numbers.

---

## 4. Production reality check

The Fly app has **only 5 secrets**: `AUTH_SECRET`, `CORS_ORIGINS`, `TERMINAL_USER`, `TERMINAL_PASSWORD`, `DISABLE_LIVE_MARKET_DATA`. Everything else is unset. Verified by direct probe:

| Feature | Live status | Evidence |
|---|---|---|
| Quotes / charts (yfinance) | ✅ Works | `/api/v1/data/quotes?symbols=AAPL` → real price |
| Company fundamentals | ✅ Works | `/api/v1/company/analyze/AAPL` → 200, real data |
| Equity search / DCF / LBO | ✅ Works | `/api/v1/equity/search?q=AAPL` → real profile |
| Risk metrics | ✅ Works | `/api/v1/risk/metrics/AAPL` → real VaR/CVaR |
| **Backtest** | 🔴 **Wrong numbers** | §5.1 |
| **Portfolio optimizer** | 🔴 **Inverted** | §5.2 |
| Pairs / cointegration | 🔴 Silently fake | `pvalue: 1.0` — §5.4 |
| Options chain | 🔴 HTTP 500 | §5.5 |
| Model list (Quant panel) | 🔴 Browser-blocked | §5.7 |
| Economic / macro | ⚫ Dark | `FRED_API_KEY not configured` |
| News | ⚫ Dark | `FINNHUB_API_KEY not configured` |
| News sentiment | ⚫ Dark | `No news sources configured` |
| AI assistant | ⚫ Dark | `/api/v1/reports/health` → `openai_configured: false` |
| Screener | ⚫ Errors | no `DATABASE_URL`; leaks internal error |
| Data status | ⚫ Empty | no `DATABASE_URL` |
| Paper trading | ⚫ Unusable | no Alpaca keys |
| Enhanced risk metrics | ⚫ Errors | leaks `No module named 'riskfolio'` |

Auth **is** correctly enforced in prod (`POST /api/v1/backtest/run` → 401 unauthenticated; `/api/auth/status` → `configured: true`). `/docs` is publicly reachable.

Cosmetic but telling: the FRED error says *"Set FRED_API_KEY in Render Environment."* The app hasn't run on Render in months.

---

## 5. Bugs, ranked

### 5.1 🔴 P0 — The backtest engine subtracts profits from equity

`core/backtesting.py:306` and `:343`:

```python
trade.close(date, price)
self.trades.append(trade)
equity -= trade.pnl        # <-- realized PnL is SUBTRACTED
```

`Trade.close()` (line 33) defines `pnl` as positive for a winner (`exit_price - entry_price` for a long). So a profitable trade *reduces* equity and a losing trade *increases* it. Line 334 in the same loop marks unrealized PnL with `equity + unrealized_pnl` — the correct sign — which makes the inconsistency plain.

Reproduced (Appendix A): a market rising 100 → 159 (+59%), held long the entire time, zero commission, realized PnL **+5,900**, reported `final_equity` **94,100**, `total_return` **−5.90%**.

**Blast radius.** `core/backtest_service.py:73` hardcodes this standard engine:

```python
engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
```

and that is the path behind `POST /api/v1/quant/backtest` (`api/quant_api.py:344` → `core/backtest_api_adapter.run_backtest_contract` → `core/backtest_service`). That endpoint is what **`BacktestPanel` calls — the `BACKTEST` command, the headline demo in the README.** Through this path the same +59% market reports **−11.8%**. The endpoint is not auth-gated.

`InstitutionalBacktestEngine` overrides `run_backtest` and uses `equity += trade.pnl` — **correct**. `/api/v1/backtest/run` defaults to institutional (`BACKTEST_USE_INSTITUTIONAL_DEFAULT=True`, `config/settings.py:73`), so *that* endpoint is fine unless a caller passes `use_institutional=false`. So the bug is confined to the standard engine — but the standard engine is exactly what the terminal's Backtest panel uses, unconditionally.

*Fix:* `equity += trade.pnl` at `core/backtesting.py:306` and `:343`. Then add a regression test asserting a long-only strategy on a monotonically rising series returns > 0.

### 5.2 🔴 P0 — The max-Sharpe optimizer recommends the worst portfolio

`core/optimizer_service.py:46-48` computes **daily** statistics and passes an **annual** risk-free rate:

```python
exp_ret = returns.mean()   # daily,  ~0.001
cov     = returns.cov()    # daily
opt = MeanVarianceOptimizer(exp_ret, cov, risk_free_rate)   # 0.02 — ANNUAL
```

`models/portfolio/optimization.py:53` then evaluates `(portfolio_return - risk_free_rate) / portfolio_std`. The numerator is always strongly negative, so maximizing it over σ means **maximizing σ**. The optimizer converges on the highest-volatility asset.

Reproduced (Appendix A): given a low-vol asset with a *higher* return and a high-vol asset with a *lower* return, the production call path allocates **100% to the high-vol, low-return asset**, Sharpe `−0.65`. Annualizing both inputs flips it to 100% low-vol, Sharpe `+1.56`.

Live: `/api/v1/risk/optimize?symbols=AAPL,MSFT,GOOGL` → `weights {AAPL: 0.0, MSFT: 0.0, GOOGL: 1.0}`, `expected_return 0.0031`, `volatility 0.0188`, `sharpe_ratio −0.8998`. Note 0.0031/0.0188 = **+0.165**, not −0.90 — that discrepancy is the tell.

Feeds the **Portfolio (`PORT`)** and **Optimizer (`OPT`)** panels. The same units error repeats in the efficient-frontier loop at `core/optimizer_service.py:70`.

`core/utils.calculate_sharpe_ratio` handles this correctly (`returns - risk_free_rate / 252`, then `× √252`), so this is a local inconsistency, not a systemic misunderstanding. **Neither `core/optimizer_service.py` nor `models/portfolio/optimization.py` is imported by any test** — coverage reports both as "never imported."

*Fix:* annualize (`exp_ret * 252`, `cov * 252`) before constructing the optimizer, fix the frontier Sharpe at line 70, and add a regression test asserting the optimizer prefers a dominant asset.

### 5.3 🔴 P0 — A hardcoded multiplier is served as an ML prediction

`core/comprehensive_integration.py:269-281`:

```python
def _predict_volatility_with_ml(self, returns: pd.Series) -> float:
    """Predict volatility using ML."""
    if len(returns) >= 20:
        rolling_vol = returns.rolling(20).std()
        predicted_vol = rolling_vol.iloc[-1] * 1.1   # "Slight upward bias"
        return predicted_vol
```

There is no model. It is a 20-day rolling standard deviation times `1.1`. It is surfaced to clients as `"ml_predicted_volatility"` (lines 257 and 403) via `ComprehensiveIntegration`, which `api/comprehensive_api.py:11` imports and exposes at `/api/v1/comprehensive/analyze/{symbol}` — mounted, open, live.

This is the one place the system fabricates a number and labels it as something it isn't. Everything else degrades honestly.

*Fix:* rename to `rolling_volatility_estimate`, drop the 1.1, or wire it to a real model.

### 5.4 🟠 P1 — The cointegration test is silently disabled in production

`api/quant_api.py:219-228`:

```python
try:
    from arch.unitroot import engle_granger
    ...
except ImportError:
    coint = False
    pvalue = 1.0
    test_stat = 0.0
```

`arch` is in `requirements.txt` and `requirements-ci.txt` but **not `requirements-api.txt`**, which is what the Docker image installs (`INSTALL_OPTIONAL_DEPS` defaults to `false`, `Dockerfile:31`). So in production the import fails, the exception is swallowed, and the endpoint returns a confident-looking `"cointegrated": false, "pvalue": 1.0` — a statistically meaningless result presented as a result. Confirmed against live: `/api/v1/quant/pairs?symbol1=AAPL&symbol2=MSFT` → `pvalue: 1.0, test_statistic: 0.0`.

Tests pass locally because `.venv-ci` *does* have `arch`. Green CI, wrong production answer, no signal anywhere.

*Fix:* add `arch` to `requirements-api.txt`, and make the fallback return `"cointegration_unavailable": true` instead of a fake p-value.

### 5.5 🟠 P1 — `/api/v1/quant/options-chain/{ticker}` returns HTTP 500 in production

```
GET /api/v1/quant/options-chain/AAPL
→ 500  {"error":"Cannot subtract tz-naive and tz-aware datetime-like objects."}
```

The Quant panel calls this. A timezone mismatch in expiry-date arithmetic.

### 5.6 🟠 P1 — Four `models/` modules cannot be imported; six dependencies are declared nowhere

| Module | Hard import of | Declared in |
|---|---|---|
| `models/timeseries/advanced_ts.py` | `pmdarima`, `tsfresh` | **nowhere** |
| `models/portfolio/advanced_optimization.py` | `riskfolio` | **nowhere** |
| `models/nlp/sentiment.py` | `torch`, `transformers` | `torch` only in `requirements.txt` (not prod); `transformers` **nowhere** |
| `models/rl/deep_rl_trading.py` | `gymnasium` | `requirements.txt` only (not prod) |

Also imported but declared in no requirements file: `tensorflow`/`keras`, `alphalens`, `praw`, `tweepy`, `alpaca_trade_api`. (`autogluon`, mentioned in old docs, is used nowhere.)

Because `models/nlp/sentiment.py` imports `torch` at column 0, the hard import takes down its own `SimpleSentiment` fallback — the graceful-degradation path is unreachable.

These modules back real endpoints — `/api/v1/risk/portfolio/enhanced-metrics`, `/api/v1/predictions/forecast-arima/{ticker}`, `/api/v1/predictions/extract-features/{ticker}`, `/api/v1/predictions/sentiment/{ticker}` — inside `try/except` blocks that leak the raw exception to the client:

```
GET /api/v1/risk/portfolio/enhanced-metrics?symbols=AAPL,MSFT
→ 200  {"error":"No module named 'riskfolio'"}
```

HTTP 200 with a Python `ModuleNotFoundError` string is both a UX bug and mild information disclosure.

### 5.7 🟠 P1 — The Quant panel's model list is blocked by the browser

`Dockerfile:54` runs uvicorn **without `--proxy-headers`**, so it doesn't trust Fly's `X-Forwarded-Proto: https`. FastAPI's trailing-slash redirect then emits an absolute URL with the wrong scheme:

```
GET https://models-terminal.fly.dev/api/v1/models
→ 307  location: http://models-terminal.fly.dev/api/v1/models/
```

An `https` page fetching an `http` URL is blocked as mixed content. `QuantPanel.tsx:40` fetches `TERMINAL_API_ENDPOINTS.models` = `/api/v1/models` (no trailing slash), so the model list silently fails in the deployed app. It works locally over plain http, which is why it was never caught.

*Fix (one line):* `uvicorn ... --proxy-headers --forwarded-allow-ips='*'`.

### 5.8 🟠 P1 — Internal exceptions leaked to clients

```
GET /api/v1/screener/run
→ 200  {"results":[],"count":0,"error":"'NoneType' object has no attribute 'connect'"}
```

The no-`DATABASE_URL` path surfacing a raw `AttributeError`. Should be a clean "screener requires a database" message.

### 5.9 🟠 P1 — Two latent `NameError`s, invisible because CI doesn't lint `models/`

`ruff check models/` reports **189 errors** (117 unused imports, 54 bare `except:`, 14 unused vars, 2 undefined names). CI lints only `config/ api/ core/`. The two undefined names are real crashes:

- `models/quant/advanced_econometrics.py:334` — `K = self.covariance @ H.T @ inv(S)`; `inv` is never imported. The Kalman gain update raises `NameError` on first call.
- `models/timeseries/advanced_ts.py:250` — calls `extract_relevant_features(...)`, but only `extract_features` is imported from tsfresh.

### 5.10 🟡 P2 — Other real defects found while reading

- `models/portfolio/advanced_optimization.py` — `efficient_frontier_cvar` never passes `target_return`, so every frontier point is identical.
- `models/valuation/institutional_dcf.py` — `scenario_analysis` mutates free cash flows and never restores them, so scenarios compound into each other.
- `models/risk/stress_testing.py` — `max_gain` initialized to `+inf`, breaking best-position tracking (lines 458, 486); `EEM` appears twice in the scenario table.
- `models/macro/macro_indicators.py` — two malformed ISM FRED series IDs.
- `core/realtime_streamer.py:468` — un-awaited `asyncio.sleep` inside a thread. `core/realtime_streaming.py:133` — `asyncio.create_task` with no running loop. (Both modules are orphaned.)
- `models/quant/institutional_grade.py` — `HestonStochasticVolatility.call_price` is a self-labeled Black-Scholes approximation, not Heston.

### 5.11 🟡 P2 — Stale-symbol race in the primary chart

`frontend/src/terminal/panels/PrimaryInstrument.tsx:212-246` fetches on `[primarySymbol, timeframe, retryKey]` with no `AbortController` and no stale-response guard. Switch symbol quickly and a slower earlier response can resolve last, calling `setData()` with the wrong symbol's candles under the new heading. This matches the open FOLLOW-UP note in `.superpowers/sdd/progress.md` (`CLOSE showed TSLA 419.77 under AAPL heading`).

### 5.12 🟡 P2 — Auth fails *open*

`api/auth_api.py:96-102` — `get_current_user_if_configured` returns `"anonymous"` unless `TERMINAL_USER`, `TERMINAL_PASSWORD` **and** a non-default `AUTH_SECRET` are all set. A deployment that forgets one serves every "protected" route to the public, with no error and no log. Production is currently configured correctly, so this is a footgun rather than an active vulnerability — but it should fail closed.

Only 5 of 19 routers are gated at all (`ai`, `automation`, `paper_trading`, `predictions`, plus 4 backtesting POSTs). `models`, `data`, `equity`, `quant`, `news`, `risk`, `company`, `screener`, `monitoring`, `reports`, `comprehensive`, `institutional`, `orchestrator` and all WebSockets are open by design, and `/docs` is public in production.

---

## 6. Live trading, dead code, duplication

### 6.1 Live trading is *disabled by default and broken*, not structurally blocked

The README says "not for live trading." That is currently true, but not for the reason it implies — and it is worth being precise, because this is the one area where being wrong is expensive.

`core/live_trading.py` (491 lines) is a red herring. Its docstring claims *"Live Trading Engine … Production-ready trading execution,"* but it has **zero broker wiring** — `execute_order()` (line 431) runs risk checks, sets `order.status = FILLED`, and appends to a list. Nothing imports it. It is a simulator with a misleading docstring.

The **only** real broker-order code is `core/paper_trading.py::AlpacaAdapter`, called from two reachable places:

- `api/automation_api.py:206,227` — `POST /api/v1/automation/predict-and-trade?execute_trades=true`
- `core/automated_trading_orchestrator.py:367,389`, reachable via `POST /api/v1/orchestrator/start-automated?execute=true`

What actually protects you today:

1. `execute_trades` / `execute` default to `False`.
2. `ALPACA_API_KEY` / `ALPACA_API_SECRET` must be set (they are not, in prod).
3. `ALPACA_API_BASE` defaults to `https://paper-api.alpaca.markets`.
4. `alpaca_trade_api` is in **no requirements file**, so the SDK isn't installed.
5. Both callers invoke `alpaca.is_authenticated()` and `alpaca.submit_order(...)` — **neither method exists** on `AlpacaAdapter`, which only defines async `place_order`. Verified: `hasattr(AlpacaAdapter, 'submit_order') == False`. The call would raise `AttributeError` before any order.

So: safe defaults plus a missing dependency plus broken glue. There is **no kill switch and no structural block**, and `ALPACA_API_BASE` is read from the environment with no validation — pointing it at `https://api.alpaca.markets` is the intended toggle. Someone who installed the SDK and fixed the method names could route real orders by changing one env var. "Structurally disabled" overstates it; **"disabled by default and non-functional as written"** is accurate.

*Recommendation:* add an explicit `ALLOW_LIVE_TRADING` guard that refuses any non-paper base URL, and fix or delete the broken `submit_order` call sites so the failure mode is intentional rather than accidental.

### 6.2 Dead code: 54 modules, ~21,000 lines (32% of the codebase)

Modules with **zero importers anywhere in `api/`, `core/`, `automation/`, or `workers/`**. Some are exercised by tests or notebooks; none are reachable from the running app.

`models/` (~7,900 lines): all three `fixed_income/` modules; `fundamental/{comparable_analysis, financial_statements, ratios}`; `valuation/institutional_dcf`; `macro/{advanced_models, economic_models, macro_indicators, central_bank_analysis, geopolitical_risk}`; `ml/{feature_engineering, transformer_models}`; `rl/deep_rl_trading`; `risk/scenario_analysis`; all three `sentiment/` modules; `trading/backtesting` (a *third* backtester).

`core/` (~8,600 lines): `quant_engine`, `price_prediction`, `reinforcement_learning`, `ensemble_models`, `anomaly_detection`, `signal_generator`, `sentiment_analysis`, `unified_fetcher`, `advanced_visualizations`, `bloomberg_terminal_ui`, `dashboard`, `realtime_streamer`, `realtime_streaming`, `ai/llm_provider`, `predictive_cache`, `parallel_fetcher`, `config_manager`, `startup_validation`, `backfill`, `live_trading`.

`workers/` (262 lines): the Celery app and ingestion tasks. No `.delay()` / `.apply_async()` call exists anywhere, and no deploy config starts a worker. `api/scheduler.py` (APScheduler) replaced it.

Most of this is **real, working code that was simply never wired in** — not junk. It should be archived with provenance, not deleted.

### 6.3 Genuine stubs and mislabels

- `models/macro/geopolitical_risk.py` — **fake.** "Risk scores" are invented hardcoded coefficient tables. *(orphaned)*
- `models/sentiment/social_sentiment.py` — near-stub; without `praw`/`tweepy` every method returns a canned error, and even live it uses upvote counts as a sentiment proxy. *(orphaned)*
- `models/macro/central_bank_analysis.py` — real FRED + Taylor rule, but `analyze_fed_communications` / `rate_expectations` return hardcoded placeholders. *(orphaned)*
- `models/sentiment/news_sentiment.py` — real analyzer, but `get_market_news` returns `[]`. *(orphaned)*
- `core/comprehensive_integration.py::_predict_volatility_with_ml` — **reachable and live**, see §5.3.
- `core/backtesting.py::SimpleMLPredictor` — named "ML"; is rules-based, `self.model = None`.
- `core/ensemble_models.py` — named like an ML ensemble; is deterministic rule fusion, no sklearn or torch.
- `core/data_fetcher_enhanced.py` — misnamed. It's validators, a rate limiter, and HTTP health-checkers, not a fetcher.

Legitimate, not stubs: the `NotImplementedError`s in `core/data_providers/{newsapi,sec_edgar}_provider.py` are deliberate interface segregation.

### 6.4 Duplication

- **Three backtest paths**: `/api/v1/backtest/*` (institutional by default, correct), `/api/v1/quant/backtest` (standard engine, **inverted**), `/api/v1/institutional/backtest`. Plus a fourth, orphaned engine in `models/trading/backtesting.py`.
- **Five data-access modules**: `core/data_fetcher.py` is the real workhorse (~27 importers). `core/market_data_facade.py` is the self-described "canonical" DB-first facade but only 3 core services use it — **no API router does**. `core/data_fetcher_enhanced.py` is misnamed health-check code. `core/unified_fetcher.py` duplicates `data_fetcher`'s provider registry and is used only by the dead `backfill.py`. `core/parallel_fetcher.py` is orphaned.
- **Two realtime modules**, both orphaned, both with async bugs: `realtime_streamer.py`, `realtime_streaming.py`.
- **Two viz stacks**: the `core/advanced_viz/` package (primary, 1,782 lines) and the legacy `core/advanced_visualizations.py`, which defines a *second, unrelated* `PublicationCharts` class kept alive by one test.
- **Three `RegimeDetector` classes** (`core/quant_engine.py:316`, `models/quant/advanced_models.py:92`, `core/signal_generator.py:76`) — **none exposed via any HTTP route**, despite the README, the login page, and the v1 backlog all advertising regime detection.
- `frontend/src/charts/CandlestickVolume.tsx` is fully implemented but unused. `endpoints.ts` `backtestRun` is an unused constant.
- `tests/test_cpp_quant.py` tests a `quant_accelerated` C++ extension and a `cpp_core/` directory **that do not exist** — 10 permanently-skipped tests.

---

## 7. Why 374 passing tests didn't catch any of this

**Coverage is 30% overall** (`api/` 32%, `core/` 36%, `models/` 21%), and what coverage exists is mostly structural.

The backtest tests are the clearest example. `tests/test_core_backtesting.py` has four tests against the engine in §5.1. Their assertions, in full: keys exist in the result dict; `final_equity >= 0`; the same inputs give the same outputs; `num_trades` matches `len(engine.trades)`; and institutional equity doesn't exceed standard equity *by more than 5% of initial capital*. **Not one asserts that a profitable strategy produces a gain.** A sign inversion sails straight through. The optimizer is worse: neither `core/optimizer_service.py` nor `models/portfolio/optimization.py` is imported by any test at all.

Subsystems at **0% coverage**, several of which run in production: `api/scheduler.py` (152 stmts), `api/ai_tools.py` (120), `core/config_manager.py` (286), `core/data_fetcher_enhanced.py` (295), `core/signal_generator.py` (256), `core/realtime_streamer.py` (256), `core/live_trading.py` (243), `core/predictive_cache.py` (223), `core/parallel_fetcher.py` (196), `core/cold_storage.py` (174).

Thinly covered but user-facing: `api/quant_api.py` 7% (the `QUANT` command), `api/screener_api.py` 11%, `api/equity_api.py` 15%, `api/risk_api.py` 17% (the `PORT` command), `api/data_api.py` 23%.

**17 tests skip by default.** Ten are the nonexistent C++ extension. Four gate the *live data-provider integration* behind `RUN_LIVE_PROVIDER_TESTS`, so the real external data path is never exercised in CI. And `tests/test_improvements.py` and `tests/test_institutional_metrics.py` call `pytest.skip(result["error"])` — **a test that skips itself when the code under test errors**, converting real breakage into a green run.

**CI gaps** (`.github/workflows/ci.yml`):
- `ruff` covers only `config/ api/ core/`. `models/` — 15.7k lines, 189 errors, 2 real `NameError`s — is unlinted.
- **No Python typechecking at all.** No mypy, no config, not in any requirements file. Only the frontend is typechecked.
- The frontend job runs `typecheck` + `test` but **not `npm run build`** — which is how commit `8cc58969 "Fix syntax error in CSS for build"` became necessary.
- `frontend/package.json`'s `lint` script is a no-op `echo`. There is no ESLint.
- No coverage gate, no security scanning (bandit / pip-audit), no deploy job — Fly deploys are manual.

**Requirements drift across three files.** `requirements-api.txt` (what prod installs) lacks `statsmodels`, `arch`, `cvxpy`, `PyPortfolioOpt`, `pyarrow`, and every heavy ML dep — while *including* `celery[redis]` and `redis`, which nothing uses. `requirements-ci.txt` adds `statsmodels`/`arch`/`cvxpy`/`PyPortfolioOpt` (so tests exercise code prod can't run) but omits `sqlalchemy`, `alembic`, `apscheduler`, `psycopg2`, `anthropic`, `vaderSentiment`. Six imported packages are declared nowhere (§5.6). `requirements-ci.txt` pins `pytest>=9.0.3` while `.venv-ci` runs 7.4.2.

**`docker-compose.yml` describes a different system than what ships**: it targets `backend.main:app` with TimescaleDB + Redis + Celery worker + Celery beat + Prometheus. Fly runs slim single-service `api.main:app` with none of that.

---

## 8. Release and documentation state

**The `v1.0.0` tag shares no git history with `main`.**

```
tag v1.0.0 → f710aca1 (2026-05-18)
HEAD       → 8796d4d4 (2026-07-06)
git merge-base v1.0.0 HEAD  →  (none — disjoint histories)
commits in HEAD not in tag: 104   (= every commit on main)
commits in tag not in HEAD:  86
```

The `9fe4f93d "remove committed venv"` rewrite created a new root. The published GitHub release `v1.0.0 — Quant Terminal` points at a tree that no longer exists on `main`, and the live site runs code with no ancestral relationship to what the release advertises. (`.git` is 286 MB, a further consequence.)

`docs/RELEASE_CHECKLIST.md` has **0 of ~30 items checked**. The release was published without the checklist ever being run — including the demo-workflow step "`BACKTEST AAPL` — backtest completes with equity curve and metrics," which would not have caught §5.1 anyway, since it only asks for *a* number.

**Doc rot:**
- `docs/architecture/current-state.md` (May 16) still lists "unmounted routers used by frontend" as **critical risk #1**. Fixed — all three routers mount cleanly and their routes are live. The whole "Main Risks by Severity" section is stale.
- `docs/SYSTEM_INVENTORY.md` is a Feb baseline claiming "99 routes across 16 routers" (actual: 122 / 19 + auth), `auth_api` "🔴 Stub" (it's real), factor models and regime detection as stubs. It self-labels as archived but is still linked.
- `API_DOCUMENTATION.md` was last touched **Feb 4** — five months of drift.
- `.env.example` is missing **12 variables** that `.env` uses (`AI_ANALYSIS_ENABLED`, `AI_SENTIMENT_ANALYSIS`, `AI_SUMMARY_MAX_TOKENS`, `BACKTEST_DEFAULT_COMMISSION`, `BACKTEST_DEFAULT_SLIPPAGE`, `BACKTEST_USE_INSTITUTIONAL_DEFAULT`, `COIN_GECKO_API_KEY`, `ENABLE_METRICS`, `PYTHONUNBUFFERED`, `SAMPLE_DATA_SOURCE`, `TIINGO_API_KEY`, `WEBSOCKET_ENABLED`). Backlog item A4 claims "covers all required vars ✅".
- `BACKTEST_METHODOLOGY.md` links to `METRICS.md`, which doesn't exist. `LAUNCH_GUIDE.md` is linked and doesn't exist.
- `GETTING_STARTED.md` and `README.md` prescribe `python -m venv` + `pip`, against your standing "always use uv" rule — and the repo's `.venv` is broken, which is why every plan doc says to use `.venv-ci`.
- README and `DEPLOYMENT_GUIDE.md` recommend deploying the frontend to Vercel, but production bundles the SPA into the Fly image. A stale `frontend/.vercel/project.json` remains.

**Scope claims not met.** The README's v1.0 table says the Quant module ships "regime classification"; the login page advertises "regime detection"; `docs/FEATURE_BACKLOG.md` B2 requires a regime timeline and B5 requires a correlation matrix and factor heatmap. None of the three exists in the UI, and no regime endpoint exists — despite three `RegimeDetector` implementations sitting in the codebase.

**Untracked:** `docs/superpowers/` (the repair plan) and `stitch_screens/` (design references).

---

## 9. What's left to be "fully finished"

### P0 — Correctness. These produce wrong numbers today.
1. **Fix the backtest sign inversion** (`core/backtesting.py:306`, `:343` → `equity += trade.pnl`). Add a regression test: long-only on a rising series must return > 0. *This is the single most important line in this document — the `BACKTEST` command currently reports losses on winning strategies.*
2. **Fix the optimizer units bug** (annualize in `core/optimizer_service.py:46-48`; fix the frontier Sharpe at `:70`). Add a test asserting the optimizer prefers a dominant asset.
3. **Stop serving `rolling_std × 1.1` as `ml_predicted_volatility`** (`core/comprehensive_integration.py:277`).
4. **Add `arch` to `requirements-api.txt`** and make the cointegration fallback report unavailability instead of `pvalue: 1.0`.
5. Fix `/api/v1/quant/options-chain/{ticker}` — the tz-naive/tz-aware 500.
6. Add `--proxy-headers --forwarded-allow-ips='*'` to the uvicorn command in `Dockerfile:54`.

### P1 — Make the live deployment real
7. Set the missing Fly secrets: `FRED_API_KEY`, `OPENAI_API_KEY`, `FINNHUB_API_KEY` (or `NEWSAPI_KEY`), and — for the screener and Data Status — `DATABASE_URL` (Supabase free tier) plus `db/init.sql`. Right now roughly half the terminal is dark for anyone who opens the link.
8. Stop leaking raw exceptions to clients (§5.6, §5.8). Map `ModuleNotFoundError` and DB-absent paths to clean messages.
9. Declare `pmdarima`, `tsfresh`, `riskfolio`, `transformers`, `tensorflow`, `alphalens` — or delete the endpoints that depend on them. They are broken in *every* environment today.
10. Fix the two `NameError`s in `models/` and extend `ruff` in CI to cover `models/`.
11. Add an `AbortController` to `PrimaryInstrument`'s fetch (§5.11) — already a known open follow-up.
12. Put an explicit guard on the live-order path (§6.1): reject any non-paper `ALPACA_API_BASE` unless `ALLOW_LIVE_TRADING=true`, and fix or remove the broken `submit_order` / `is_authenticated` calls.

### P2 — Engineering hygiene
13. Make auth fail *closed*: refuse to start when `AUTH_SECRET` is unset outside dev. Consider gating `/docs`.
14. Add `npm run build` to the frontend CI job; add a real ESLint config.
15. Add a startup assertion that all 19 routers mounted — the lazy `try/except` loader (`api/main.py:105-111`) currently degrades to a warning log.
16. Remove the `pytest.skip(result["error"])` pattern in the two tests that hide real failures.
17. Reconcile the three requirements files. Drop `celery[redis]` + `redis` from `requirements-api.txt`.
18. Introduce mypy (even `--ignore-missing-imports`) and a coverage floor. 30% with the scheduler at 0% is the real risk surface.
19. Fix the §5.10 defects.

### P3 — Consolidation (~21,000 lines, 32% of the codebase)
20. Archive — don't delete — the 54 orphaned modules (§6.2). Start with `core/live_trading.py`: at minimum rewrite the "production-ready trading execution" docstring, which describes something that does not exist.
21. Pick one data path. Either finish migrating the API layer onto `core/market_data_facade.py` or admit `DataFetcher` is canonical and retire the facade. Retire `unified_fetcher`, `parallel_fetcher`, one realtime module, one viz stack.
22. Collapse four backtest engines to one — ideally the institutional one, which is the correct one.
23. Remove `workers/` (Celery) and `tests/test_cpp_quant.py`, or restore the `cpp_core/` build they reference.

### P4 — Close the scope claims
24. Either ship regime detection (an endpoint over the existing `core/quant_engine.RegimeDetector`, plus a timeline chart) or remove the claim from the README, the login page, and the backlog. Same for the correlation matrix and factor heatmap in B5.

### P5 — Release process
25. Re-tag. Delete and recreate `v1.0.0` on an actual `main` ancestor — or, cleaner, tag `v1.1.0` at current `main` after the P0 fixes — and republish the GitHub release.
26. Actually run `docs/RELEASE_CHECKLIST.md`, and strengthen its demo steps to assert *direction*, not just presence, of numbers.
27. Refresh the stale docs (§8): retire `current-state.md` and `SYSTEM_INVENTORY.md`, regenerate `API_DOCUMENTATION.md`, sync `.env.example`, fix the broken links, switch setup instructions to `uv`.
28. Commit or discard `docs/superpowers/` and `stitch_screens/`.

### Deliberately deferred (v2, already documented)
`docs/FEATURE_BACKLOG.md` is a good doc and its v2 roadmap holds: cross-asset backtesting, MLflow + model registry, feature store, RL training pipeline, vol-surface modeling, LLM research agent, OAuth2, Prometheus metrics, Playwright E2E, Redis-backed rate limiting (the current limiter is in-memory, 100 req/60 s per IP, single-process only).

---

## 10. The short version

Six fixes (§9 P0) and five Fly secrets take this from "impressive but quietly wrong" to a genuinely finished v1. Two of those six are one-line sign/units changes.

Everything else — 21,000 lines of dead code, 30% coverage, the disjoint tag, five months of doc drift — is cleanup that makes the project maintainable. It is not what's standing between you and a working terminal.

**The one thing to fix today:** `core/backtesting.py:306`. `equity -= trade.pnl` should be `equity += trade.pnl`. A backtesting tool that reports a loss on a strategy that made 59% is worse than no backtesting tool, because it will talk you out of good ideas.

---

## Appendix A — Reproductions

Both run from the repo root with `.venv-ci` (the repo's `.venv` is broken).

### A.1 The backtest engine inverts realized PnL

```python
import numpy as np, pandas as pd
from core.backtesting import BacktestEngine

n = 60
prices = np.linspace(100, 159, n)          # market rises 59%
df = pd.DataFrame({"Open":prices,"High":prices,"Low":prices,"Close":prices,
                   "Volume":np.full(n,1e6)},
                  index=pd.date_range("2024-01-01", periods=n, freq="D"))

eng = BacktestEngine(initial_capital=100_000, commission=0.0)   # no costs
res = eng.run_backtest(df, np.ones(n), signal_threshold=0.3, position_size=0.1)

# trades: 1 | realized pnl: +5900.0
# final_equity: 94100.00      <-- lost money on a +59% market
# total_return: -0.0590
```

Through the endpoint the terminal actually calls (`POST /api/v1/quant/backtest` → `core/backtest_service._run_backtest_engine`), the same inputs give `final_equity: 88200.00`, `total_return: -0.118`.

### A.2 The optimizer picks the dominated asset

```python
import numpy as np, pandas as pd
from models.portfolio.optimization import MeanVarianceOptimizer

rng = np.random.default_rng(0); n = 1000
low  = rng.normal(0.0008, 0.005, n)   # ~20%/yr return,  ~8%/yr vol
high = rng.normal(0.0002, 0.030, n)   # ~5%/yr return,  ~48%/yr vol
df = pd.DataFrame({"LOW_VOL": low, "HIGH_VOL": high})
exp_ret, cov = df.mean(), df.cov()

# Exactly how core/optimizer_service.py calls it: DAILY stats, ANNUAL rf
MeanVarianceOptimizer(exp_ret, cov, risk_free_rate=0.02).optimize_sharpe()
# weights {'LOW_VOL': 0.0, 'HIGH_VOL': 1.0}   sharpe -0.6529   <-- picks the worse asset

# Correct units
MeanVarianceOptimizer(exp_ret*252, cov*252, risk_free_rate=0.02).optimize_sharpe()
# weights {'LOW_VOL': 1.0, 'HIGH_VOL': 0.0}   sharpe +1.5610
```

`LOW_VOL` strictly dominates: higher return, one-sixth the volatility.

---

## Appendix B — How this was verified

- Backend suite: `PYTHONPATH=. .venv-ci/bin/python -m pytest -q` → `374 passed, 17 skipped` in 29.75 s.
- Coverage: `pytest --cov=api --cov=core --cov=models` → 30% overall.
- Lint: `ruff check config/ api/ core/` clean; `ruff check models/` → 189 errors, 2× F821.
- Frontend: `npm ci && npm run build && npx tsc --noEmit && npm test` → all exit 0, 24 tests pass.
- Live route inventory: `GET https://models-terminal.fly.dev/openapi.json` → 122 paths.
- Live behaviour: direct probes of ~25 endpoints (§4).
- Fly config: `flyctl secrets list -a models-terminal` → 5 secrets (names only; no values read).
- Git: `git merge-base v1.0.0 HEAD` → no common ancestor.
- Broker path: `hasattr(AlpacaAdapter, 'submit_order')` → `False`, confirming the call sites in `api/automation_api.py` cannot execute.
- Dead-code map: per-module importer scan across `api/ core/ automation/ workers/`.
- The two P0 bugs: reproduced with the runnable scripts in Appendix A.

No files were modified, nothing was committed or pushed, and no credential values were read or printed.
