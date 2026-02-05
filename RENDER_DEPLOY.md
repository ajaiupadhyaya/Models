# Deploy on Render — Terminal, Charts, AI, ML

This guide ensures the Bloomberg-style terminal works on Render: charts, AI, ML, and all APIs.

## One Web Service (recommended)

1. **Connect repo** to Render and create a **Web Service** from the repo.
2. **Build**: set **Dockerfile Path** to `./Dockerfile`, **Docker Context** to `.`.
3. **Environment**: In the Render dashboard → your service → **Environment**, add the variables below. Formatting matters — see the rules in the next section.

---

## Render environment page — formatting rules

These rules prevent the terminal from failing to load features (tabs, charts, API calls) due to misconfigured env vars.

- **Key names**: Copy exactly. Render is case-sensitive (e.g. `FRED_API_KEY` not `fred_api_key`). No spaces before or after the key.
- **Values**: No trailing slashes on URLs. For example:
  - `ALPHA_VANTAGE_API_KEY`: paste only the key string (no quotes).
  - `ALPACA_API_BASE`: use `https://paper-api.alpaca.markets` (no trailing `/`).
- **API / WebSocket**: This app uses one Web Service (API + frontend on the same URL). You do **not** set a separate API URL or webhook URL. All `/api/*` and WebSocket (`wss://`) requests go to the same host Render gives you (e.g. `https://your-service.onrender.com`). The frontend is built without `VITE_API_ORIGIN`, so it correctly uses the same origin.

### Do not set on Render (single Web Service)

| Key | Why |
|-----|-----|
| `PORT` | Render sets this automatically. |
| `VITE_API_ORIGIN` | Only for split deploys (frontend on a different domain). For one Web Service, leave unset so the app and API use the same origin. Setting it incorrectly can break all API and WebSocket requests and make the terminal show "API unreachable" or empty tabs. |
| Any "webhook URL" or "API base URL" | Not used. API and WebSocket are same-origin; no extra URL needed. |

---

## Environment variables to add

Add these in **Render dashboard → your service → Environment**. Click **Add Environment Variable** and use **Key** and **Value** as below. Values: no trailing slashes; no quotes around simple values.

### Required for login and core app

| Key | Value (example / notes) |
|-----|--------------------------|
| `TERMINAL_USER` | Username to sign in (e.g. `admin`) |
| `TERMINAL_PASSWORD` | Your chosen password |
| `AUTH_SECRET` | Long random string for JWT (e.g. output of `openssl rand -hex 32`) |

### Required for charts and data

| Key | Value (example / notes) |
|-----|--------------------------|
| `FRED_API_KEY` | Your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) — Economic tab, yield curve, macro |
| `ALPHA_VANTAGE_API_KEY` | Your [Alpha Vantage key](https://www.alphavantage.co/support/#api-key) — charts; yfinance is fallback if unset |

### Optional but recommended

| Key | Value (example / notes) |
|-----|--------------------------|
| `OPENAI_API_KEY` | Your [OpenAI key](https://platform.openai.com/api-keys) — AI tab, NL query, summaries |
| `FINNHUB_API_KEY` | Your [Finnhub key](https://finnhub.io/) — News tab headlines |
| `ENABLE_METRICS` | `true` (literal, no quotes) for monitoring dashboard |
| `WEBSOCKET_ENABLED` | `true` (literal) for live price ticker; WebSocket uses same host, no separate URL |
| `SAMPLE_DATA_SOURCE` | `yfinance` (default) or `data_fetcher` — no trailing slash |

### Optional — paper trading (Alpaca)

| Key | Value (example / notes) |
|-----|--------------------------|
| `ENABLE_PAPER_TRADING` | `true` to enable Paper Trading tab |
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `ALPACA_API_BASE` | `https://paper-api.alpaca.markets` — **no trailing slash** |

Render sets `PORT` automatically. For a single Web Service, do **not** add `VITE_API_ORIGIN` — the app and API are on the same URL, and the frontend is built to use same-origin for all API and WebSocket requests.

---

## After deploy

1. Open your Render URL (e.g. `https://terminal-api.onrender.com`).
2. You should see the login page. Sign in with `TERMINAL_USER` and `TERMINAL_PASSWORD`.
3. **Check API health**: open `https://your-service.onrender.com/health` — should return `{"status":"healthy",...}`.
4. **Check loaded routers**: open `https://your-service.onrender.com/info` — `routers_loaded` lists which APIs are active (e.g. `models`, `predictions`, `ai`, `data`, `backtesting`).

## Full functionality (no "No data" / empty tabs)

- **Charts (Primary, Fundamental, etc.)** — Stock OHLCV: uses **Yahoo Finance** with a browser User-Agent so Render gets real data. No API key needed. If you still see "No data", check **Logs** (below).
- **Economic tab (macro, yield curve)** — **Requires `FRED_API_KEY`**. Get a free key at [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html). Without it, the Economic panel will show "FRED API key not configured".
- **Quotes / ticker strip** — Same yfinance + User-Agent; no key. If empty, check Logs.
- **AI tab** — **Requires `OPENAI_API_KEY`** for analysis and NL query.

## Checking Render logs when features show "No data"

1. In Render dashboard → your service → **Logs**.
2. Reproduce the issue (e.g. open the tab that shows "No data"), then look in the logs for:
   - `Sample data fetch failed for AAPL: ...` — charts; the message after the colon is the real error (e.g. network, JSON decode).
   - `Quotes fetch failed: ...` — ticker strip / quotes.
   - `Macro endpoint failed: ...` — Economic tab (often missing `FRED_API_KEY`).
3. Fix from the message: missing env var → add it in **Environment** and **Redeploy**; network/JSON errors → Yahoo may be rate-limiting (retry later or check Render outbound access).

## If tabs show errors or "not found"

1. **Environment formatting**: In Render → your service → **Environment**, ensure:
   - Keys are spelled exactly (e.g. `FRED_API_KEY`, `TERMINAL_USER`). No trailing slashes on any URL (e.g. `ALPACA_API_BASE` = `https://paper-api.alpaca.markets`).
   - **Do not** set `VITE_API_ORIGIN` for a single Web Service — if it is set, remove it and redeploy so the frontend uses the same origin for API and WebSocket.
2. **Same-origin**: You are using one Web Service. The app expects API and frontend on the same URL; no separate API or webhook URL is needed.
3. **Logs**: In Render → your service → **Logs**, look for:
   - `Routers loaded: ['auth', 'models', 'predictions', ...]` — confirms which routers started.
   - `Router X not available: ...` — that router failed to load (missing dep or config).
4. **Missing routers**: If e.g. `ai` is not in `routers_loaded`, add `OPENAI_API_KEY`. If `data` is missing, check that `requirements-api.txt` deps installed (build logs).
5. **Charts / sample data**: Uses yfinance default (no custom session). If you see "Impersonating chrome136 is not supported" in Logs, the app no longer passes a custom session so that error should be gone after redeploy. If charts are still empty, check Logs for "Sample data fetch failed" and fix the reported error.
6. **Cold start**: Free tier may sleep; first request can take 30–60 s. Use **Retry** in the app or hit `/health` again.

## Build notes

- The Dockerfile installs `requirements-api.txt` first (required). Then it runs `pip install -r requirements.txt || true` so optional project deps do not fail the build.
- Each API router is loaded in a try/except; if one fails (e.g. missing optional dep), the others still run. Check `/info` to see which routers loaded.

See [DEPLOYMENT_STEP_BY_STEP.md](DEPLOYMENT_STEP_BY_STEP.md) for full deployment steps and [FEATURE_FIX_PLAN.md](FEATURE_FIX_PLAN.md) for endpoint details.
