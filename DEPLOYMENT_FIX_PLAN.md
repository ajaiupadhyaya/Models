# Deployment Fix Plan – Terminal Fully Operational

This document summarizes fixes applied so that **every feature in the codebase is fully operational** in the next deployment.

**Does this accomplish everything in `context.md`?** No. The fixes make the **existing** terminal and APIs deploy and run reliably (API base URL, errors, charts, Investor Reports entry). For a full gap between `context.md` and the codebase—what’s implemented vs partial vs missing—see **`CONTEXT_GAP_ANALYSIS.md`**.

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
| Economic             | ✅ Macro list + D3 macro chart + calendar |
| News                 | ✅ Uses `/api/v1/data/news` (FINNHUB) |
| Portfolio            | ✅ Dashboard, risk, stress, Investor reports block |
| Paper                | ✅ Health, positions, orders, execute-signal |
| Automation           | ✅ Orchestrator status, run-cycle, retrain |
| Screening            | ✅ Sectors, screener run |
| AI                   | ✅ Stock analysis, NL query (right panel) |
| Investor Reports API | ✅ Exposed; UI entry in Portfolio + link to docs |

All listed features are wired and operational; missing data (e.g. no FRED key) shows clear messaging instead of raw “not found” where applicable.
