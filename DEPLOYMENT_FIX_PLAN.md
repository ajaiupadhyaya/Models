# Deployment Fix Plan – Terminal Fully Operational

This document summarizes fixes applied so that **every feature in the codebase is fully operational** in the next deployment.

**Does this accomplish everything in `context.md`?** No. The fixes make the **existing** terminal and APIs deploy and run reliably (API base URL, errors, charts, Investor Reports entry). For a full gap between `context.md` and the codebase—what’s implemented vs partial vs missing—see **`CONTEXT_GAP_ANALYSIS.md`**.

## Render: Where to set API keys (recommended)

**Yes — you should set API keys in Render.** That is the correct and secure approach for production.

- **Do not** commit API keys to the repo or hardcode them.
- **Do** set them in the Render dashboard: open your service → **Environment** → add each variable.
- The app reads all configuration from the environment via `config/settings.py` and `os.environ`; Render injects these at runtime, so keys set in Render are used automatically.

**Keys to set in Render (Environment):**

| Variable | Required for | Notes |
|----------|--------------|--------|
| `FRED_API_KEY` | Economic tab (macro, calendar, yield curve) | Get at fred.stlouisfed.org |
| `OPENAI_API_KEY` | AI Insights, NL query, News summarization, Investor Reports | Enables full AI features |
| `FINNHUB_API_KEY` | Real news headlines in News tab | Optional; without it, news may be placeholder |
| `TERMINAL_USER` | Login (if you use auth) | Set with a strong username |
| `TERMINAL_PASSWORD` | Login (if you use auth) | Set with a strong password |
| `AUTH_SECRET` | JWT signing (if you use auth) | Long random string |

**Optional:** `ALPHA_VANTAGE_API_KEY`, `BACKTEST_USE_INSTITUTIONAL_DEFAULT`, `ENABLE_PAPER_TRADING`, `ALPACA_*` for paper trading, etc. See `.env.example` for the full list.

**Single service on Render:** The Dockerfile builds the frontend and serves it from the same app; no `VITE_API_ORIGIN` is needed. The health check uses `GET /health`; Render will mark the service healthy when that returns 200.

---

## Fixes for "Check API on port 8000" and site unusable

1. **Removed all "port 8000" and "localhost" from user-facing copy**  
   Every panel and hook now shows deployment-agnostic messages, e.g. "Check that the API is running and reachable." (PrimaryInstrument, FundamentalPanel, PortfolioPanel, AiAssistantPanel, useFetchWithRetry).

2. **Auth when not configured**  
   If you don't set `TERMINAL_USER` / `TERMINAL_PASSWORD` / `AUTH_SECRET` on Render, the app used to block at login (503). Now:
   - **Backend:** `GET /api/auth/status` returns `{ "configured": false }` when auth env vars are missing.
   - **Frontend:** When there is no token, the app calls `/api/auth/status`; if `configured === false`, it **allows access without login** so the terminal is usable. If auth is configured, login is required as before.
   - **Login page:** 503 "Auth not configured" shows a clear message to set env vars; network errors show "Cannot reach API. Check that the service is running and reachable."

3. **Vite proxy**  
   `vite.config.ts` proxy target is documented as local-dev only; production build uses same origin (no proxy). Proxy target set to `127.0.0.1:8000` for clarity.

4. **Same-origin on Render**  
   All frontend `fetch` calls use `resolveApiUrl(path)`; when `VITE_API_ORIGIN` is unset (Docker/single service), requests go to the same origin, so the SPA and API on Render work without extra config.

## Problems Addressed

1. **"Not found" errors** – API calls could 404 when frontend and API were on different origins, or when resources were missing; error messages were unclear.
2. **No graphs** – Economic and Fundamental tabs had no D3 charts; only Primary/Technical had candlestick charts.
3. **Missing components** – Investor Reports API existed but had no UI entry point; deployment needed a way to point the frontend at the API when split-hosted.

## Fixes Implemented

### 1. API base URL for deployment (`VITE_API_ORIGIN`)

- **`frontend/src/apiBase.ts`** – New module that reads `import.meta.env.VITE_API_ORIGIN` and exposes:
  - `getApiBase()` – base URL for REST (e.g. `https://your-api.onrender.com`)
  - `getWsBase()` – WebSocket origin (e.g. `wss://your-api.onrender.com`)
  - `resolveApiUrl(path)` – resolves `/api/...` to full URL when base is set
- **All fetch and WebSocket usage** – `useFetchWithRetry` and `useWebSocketPrice` use the base when set; every direct `fetch()` in panels and auth uses `resolveApiUrl()`.
- **`.env.example`** – Documented `VITE_API_ORIGIN` for frontend builds.

**For split deployment (e.g. static site + separate API):**  
Set `VITE_API_ORIGIN` when building the frontend, e.g.  
`VITE_API_ORIGIN=https://your-api.onrender.com npm run build`  
Then deploy the built `frontend/dist` to your static host. All `/api/*` and WebSocket requests will go to that origin.

**For single-service deployment (Docker):**  
Do not set `VITE_API_ORIGIN`. The app is served from the same origin as the API; relative `/api` and `ws://same-host` work as before.

### 2. Friendlier error messages

- **`frontend/src/hooks/useFetchWithRetry.ts`** – `normalizeError()`:
  - 404: uses API `detail` when it looks like “not found”/“no data”, otherwise: “Resource not found. Check symbol or configuration and try again.”
  - Network/status 0: “Network error. Check API URL and CORS.”
  - Other 4xx: “Request failed (status)” instead of raw “HTTP 404”.

### 3. Economic panel – D3 macro chart

- **`frontend/src/terminal/panels/EconomicPanel.tsx`** – New `MacroChart` component:
  - Uses the first macro series from `/api/v1/data/macro` that has at least two points.
  - Renders a D3 time-series line chart (date vs value) with axes and series label.
  - Placed above the existing “latest value” list so the Economic tab always shows a graph when macro data exists (e.g. when `FRED_API_KEY` is set).

### 4. Fundamental panel – D3 price trend chart

- **`frontend/src/terminal/panels/FundamentalPanel.tsx`** – New `PriceTrendChart` component:
  - Fetches `/api/v1/backtest/sample-data?symbol={primarySymbol}&period=1y`.
  - Renders a D3 line chart of close price over time (1Y).
  - Placed at the top of the Fundamental tab so “Historical trend charts” from `context.md` are present.

### 5. Investor Reports in the UI

- **`frontend/src/terminal/panels/PortfolioPanel.tsx`** – New “Investor reports” block:
  - Fetches `/api/v1/reports/health`.
  - Shows status (e.g. “Available (OpenAI configured)” or “Check API /docs for report generation”).
  - Link to API docs so users can call `POST /api/v1/reports/generate` and related endpoints.
  - Ensures the Investor Reports feature in the codebase is visible and usable from the terminal.

## What to Verify Before Next Deployment

1. **Single-service (Docker)**  
   - Build frontend (`npm run build` in `frontend/`).  
   - Run API with `frontend/dist` present so SPA is served; confirm `/health` and `/api/*` both work.  
   - No `VITE_API_ORIGIN` needed.

2. **Split deployment (static + API)**  
   - Build with `VITE_API_ORIGIN=https://your-api-url` (no trailing slash).  
   - Deploy `frontend/dist` to static host; deploy API separately.  
   - Ensure API has CORS allowing the frontend origin (main.py already has `allow_origins=["*"]`; tighten for production if required).  
   - Confirm login, tabs, charts, and WebSocket (if used) all hit the API origin.

3. **Environment**  
   - **API:** `FRED_API_KEY` for Economic macro + calendar; `OPENAI_API_KEY` for AI and Investor Reports; `FINNHUB_API_KEY` for News; auth (`TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET`) if you use login.  
   - **Frontend (split deploy):** Only `VITE_API_ORIGIN` at build time.

4. **Charts**  
   - **Primary / Technical:** Candlestick + indicators use `/api/v1/backtest/sample-data`; works with default yfinance.  
   - **Economic:** Macro chart appears when `/api/v1/data/macro` returns at least one series with data (FRED configured).  
   - **Fundamental:** Price trend chart appears when sample-data returns 1Y candles for the selected symbol.

5. **“Not found”**  
   - If a symbol or resource is missing, the API may still return 404; the UI now shows a clearer message and suggests checking symbol or configuration.  
   - If the whole API is unreachable (wrong origin or down), the user sees “Network error. Check API URL and CORS.”

## Feature Checklist (context.md alignment)

| Feature / Tab        | Status |
|----------------------|--------|
| Primary + candlestick| ✅ D3 chart + sample-data |
| Fundamental          | ✅ Analysis + D3 price trend (1Y) |
| Technical            | ✅ PrimaryInstrument with indicators |
| Quant                | ✅ Models, backtest, walk-forward |
| Economic             | ✅ Macro list + D3 macro chart + yield curve + correlation heatmap + calendar |
| News                 | ✅ `/api/v1/data/news` (FINNHUB); AI Summarize per article |
| Portfolio            | ✅ Dashboard, risk, stress, optimization block, Investor reports block |
| Paper                | ✅ Health, positions, orders, execute-signal |
| Automation           | ✅ Orchestrator status, run-cycle, retrain |
| Screening            | ✅ Sectors, screener run; sortable table + CSV export |
| AI                   | ✅ Stock analysis, sentiment, price target + confidence, NL query (right panel) |
| Investor Reports API | ✅ Exposed; UI entry in Portfolio + link to docs |

All listed features are wired and operational; missing data (e.g. no FRED key) shows clear messaging instead of raw “not found” where applicable.
