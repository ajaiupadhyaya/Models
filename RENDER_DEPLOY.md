# Deploy on Render — Terminal, Charts, AI, ML

This guide ensures the Bloomberg-style terminal works on Render: charts, AI, ML, and all APIs.

## One Web Service (recommended)

1. **Connect repo** to Render and create a **Web Service** from the repo.
2. **Build**: set **Dockerfile Path** to `./Dockerfile`, **Docker Context** to `.`.
3. **Environment**: In the Render dashboard → your service → **Environment**, add these variables.

### Required for login and core app

| Variable | Required | Notes |
|----------|----------|--------|
| `TERMINAL_USER` | Yes | Username to sign in (e.g. `admin`) |
| `TERMINAL_PASSWORD` | Yes | Password to sign in |
| `AUTH_SECRET` | Yes | Long random string for JWT (e.g. `openssl rand -hex 32`) |

### Required for charts and data

| Variable | Required | Notes |
|----------|----------|--------|
| `FRED_API_KEY` | Yes | [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) — Economic tab, yield curve, macro |
| `ALPHA_VANTAGE_API_KEY` | Yes | [Alpha Vantage key](https://www.alphavantage.co/support/#api-key) — optional for charts; yfinance is default |

### Optional but recommended

| Variable | Required | Notes |
|----------|----------|--------|
| `OPENAI_API_KEY` | No | [OpenAI key](https://platform.openai.com/api-keys) — AI tab, NL query, summaries |
| `FINNHUB_API_KEY` | No | [Finnhub key](https://finnhub.io/) — News tab headlines |
| `ENABLE_METRICS` | No | Set `true` for monitoring dashboard |
| `WEBSOCKET_ENABLED` | No | Set `true` for live price ticker |

Render sets `PORT` automatically; the Dockerfile uses it. Do **not** set `VITE_API_ORIGIN` — the app and API are on the same URL.

## After deploy

1. Open your Render URL (e.g. `https://terminal-api.onrender.com`).
2. You should see the login page. Sign in with `TERMINAL_USER` and `TERMINAL_PASSWORD`.
3. **Check API health**: open `https://your-service.onrender.com/health` — should return `{"status":"healthy",...}`.
4. **Check loaded routers**: open `https://your-service.onrender.com/info` — `routers_loaded` lists which APIs are active (e.g. `models`, `predictions`, `ai`, `data`, `backtesting`).

## If tabs show errors or "not found"

1. **Same-origin**: You are using one Web Service. Do **not** set `VITE_API_ORIGIN`.
2. **Logs**: In Render → your service → **Logs**, look for:
   - `Routers loaded: ['auth', 'models', 'predictions', ...]` — confirms which routers started.
   - `Router X not available: ...` — that router failed to load (missing dep or config).
3. **Missing routers**: If e.g. `ai` is not in `routers_loaded`, add `OPENAI_API_KEY`. If `data` is missing, check that `requirements-api.txt` deps installed (build logs).
4. **Charts / sample data**: Uses yfinance by default; no key required. For more data, set `ALPHA_VANTAGE_API_KEY`.
5. **Cold start**: Free tier may sleep; first request can take 30–60 s. Use **Retry** in the app or hit `/health` again.

## Build notes

- The Dockerfile installs `requirements-api.txt` first (required). Then it runs `pip install -r requirements.txt || true` so optional project deps do not fail the build.
- Each API router is loaded in a try/except; if one fails (e.g. missing optional dep), the others still run. Check `/info` to see which routers loaded.

See [DEPLOYMENT_STEP_BY_STEP.md](DEPLOYMENT_STEP_BY_STEP.md) for full deployment steps and [FEATURE_FIX_PLAN.md](FEATURE_FIX_PLAN.md) for endpoint details.
