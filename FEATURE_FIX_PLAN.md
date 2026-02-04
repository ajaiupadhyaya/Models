# Terminal Tabs & Features Fix Plan

This document explains why tabs/features show "not found" or other errors and how to fix them so the next build/deploy has fully operational features.

---

## 1. Root causes

### 1.1 API unreachable (wrong origin)

- When the **frontend is served from a different origin** than the API (e.g. static site on `https://app.example.com` and API on `https://api.example.com`), the frontend uses **relative** URLs like `/api/v1/data/quotes`. Those requests go to the **frontend’s** origin and get **404** (static host has no `/api/*` routes).
- **Fix**: For a **split deploy** (frontend + API on different URLs), set `VITE_API_ORIGIN` at **build time** so all API calls target the API base URL. Example: `VITE_API_ORIGIN=https://your-api.onrender.com` then `npm run build`. See [Build/deploy config](#3-builddeploy-config) below.

### 1.2 Optional routers not loaded

- These routers are loaded in `api/main.py` inside **try/except**. If an import or dependency fails, the router is **not** registered and all its routes return **404**:
  - **Automation** (`/api/v1/automation/*`)
  - **Orchestrator** (`/api/v1/orchestrator/*`) — e.g. if `schedule` or `stable-baselines3` missing
  - **Screener** (`/api/v1/screener/*`)
  - **News** (mounted under `/api/v1/data/news`) — if `news_api` import fails
- **Fix**: Install the dependencies required by those routers (see `requirements-api.txt` and optional deps). Ensure no import errors at startup; check API logs for "X router not available". Frontend already shows hints (e.g. "Orchestrator API may not be loaded"); we can add an **API capabilities** check so the UI can show which tabs are supported.

### 1.3 404 with non-JSON body

- If the server returns **404 with HTML** (e.g. SPA fallback or proxy error page), the frontend does `res.json().catch(() => ({}))`, gets `{}`, and shows a generic "Resource not found" message. Users can’t tell if the **API is unreachable** vs a bad symbol.
- **Fix**: In `useFetchWithRetry`, detect 404 + non-JSON (e.g. `Content-Type` not `application/json`) and show a clearer message: e.g. "API unreachable or endpoint not found. Check that the API is running and, if frontend and API are on different domains, that VITE_API_ORIGIN was set at build time."

### 1.4 Auth and protected route

- **Auth** is only required for `/api/auth/me`. Other API routes do **not** require a token. If auth **is** configured and the user is not logged in, `ProtectedRoute` redirects to login. If `/api/auth/status` or `/api/auth/me` is unreachable (wrong origin or API down), the user may be stuck on "Checking authentication…" or redirected to login. So **API reachability** (and correct origin) also affects auth flow.
- **Fix**: Ensure `resolveApiUrl("/api/auth/status")` and `resolveApiUrl("/api/auth/me")` point to the real API when frontend and API are on different origins (again, `VITE_API_ORIGIN` at build time). Optionally, add a **health check** on app load and show a banner if the API is unreachable.

---

## 2. Endpoint audit (frontend → backend)

All paths are relative to the API base (same origin or `VITE_API_ORIGIN`). Method and path must match.

| Tab / Feature        | Frontend path / usage | Backend route (method + path) | Notes |
|----------------------|------------------------|-------------------------------|--------|
| **Auth**              | `/api/auth/login` (POST), `/api/auth/me` (GET), `/api/auth/status` (GET) | Same in `auth_api` | Prefix `/api/auth` in main. |
| **MarketOverview**    | `/api/v1/data/quotes`, `/api/v1/ai/market-summary` | `data_api`: GET `/quotes`; `ai_analysis_api`: GET `/market-summary` | data prefix `/api/v1/data`; ai has own prefix `/api/v1/ai`. |
| **PrimaryInstrument** | `/api/v1/backtest/sample-data` | `backtesting_api`: GET `/sample-data` | Prefix `/api/v1/backtest`. |
| **FundamentalPanel**  | `/api/v1/company/analyze/{ticker}`, `/api/v1/company/sector/{sector}`, `/api/v1/backtest/sample-data` | `company_analysis_api`: GET `/analyze/{ticker}`, GET `/sector/{sector}`; backtest as above | Prefix `/api/v1/company`. |
| **TechnicalPanel**    | Uses PrimaryInstrument only (same backtest/sample-data) | As above | No extra endpoints. |
| **QuantPanel**       | `/api/v1/models` (GET), `/api/v1/models/train` (POST), `/api/v1/backtest/run` (POST), `/compare`, `/walk-forward`, `/technical` (POST) | `models_api`, `backtesting_api` | Prefixes `/api/v1/models`, `/api/v1/backtest`. |
| **EconomicPanel**    | `/api/v1/data/correlation`, `/api/v1/data/yield-curve`, `/api/v1/data/macro`, `/api/v1/data/economic-calendar` | `data_api`: GET `/correlation`, `/yield-curve`, `/macro`, `/economic-calendar` | All under `/api/v1/data`. |
| **NewsPanel**        | `/api/v1/data/news`, `/api/v1/ai/summarize` (POST) | `news_api`: GET `/news` (mounted under `/api/v1/data`); `ai_analysis_api`: POST `/summarize` | News: requires news router loaded. |
| **PortfolioPanel**   | `/api/v1/risk/optimize`, `/api/v1/reports/health`, `/api/v1/monitoring/dashboard`, `/api/v1/risk/metrics/{ticker}`, `/api/v1/risk/stress`, `/api/v1/predictions/quick-predict` | `risk_api`, `investor_reports_api`, `monitoring`, `predictions_api` | Reports prefix `/api/v1/reports`; risk `/api/v1/risk`; monitoring `/api/v1/monitoring`; predictions `/api/v1/predictions`. |
| **PaperTradingPanel** | `/api/v1/paper-trading/health`, `/positions`, `/portfolio`, `/orders/place` (POST), `/execute-signal` (POST), plus predictions/backtest | `paper_trading_api` | Prefix `/api/v1/paper-trading`. |
| **AutomationPanel**   | `/api/v1/orchestrator/status`, `/api/v1/orchestrator/signals`, `/api/v1/orchestrator/trades`, `/api/v1/orchestrator/run-cycle` (POST), `/api/v1/orchestrator/retrain` (POST) | `orchestrator_api` (optional) | All under `/api/v1/orchestrator`. If router not loaded → 404. |
| **ScreeningPanel**    | `/api/v1/company/sectors`, `/api/v1/screener/run` | `company_analysis_api`: GET `/sectors`; `screener_api`: GET `/run` | Screener prefix `/api/v1/screener`; optional. |
| **AiInsightsPanel**   | `/api/v1/ai/stock-analysis/{symbol}` | `ai_analysis_api`: GET `/stock-analysis/{symbol}` | Prefix `/api/v1/ai` (in router). |
| **AiAssistantPanel** | `/api/v1/ai/stock-analysis/{sym}`, `/api/v1/ai/nl-query` (POST) | Same ai router | — |
| **TickerStrip**       | `/api/v1/data/quotes`, `/api/v1/ai/market-summary` | Same as MarketOverview | — |
| **WebSocket**         | `/api/v1/ws/prices/{symbol}` | `websocket_api` | Prefix `/api/v1/ws`. |

No path mismatches were found; "not found" is explained by (1) wrong origin, (2) optional routers not loaded, or (3) 404 with HTML/non-JSON.

---

## 3. Build/deploy config

### Same-origin deploy (recommended: one Docker service)

- **Dockerfile** builds the frontend and serves it from the same app that serves the API (`frontend/dist` + SPA fallback in `api/main.py`). No `VITE_API_ORIGIN` is needed; all `/api/*` requests are same-origin and hit FastAPI.
- **Render/Railway/Fly.io** using this single Docker service: set only **backend** env vars (FRED, OPENAI, TERMINAL_USER, etc.). Do **not** set `VITE_*` in the dashboard for this service (they are build-time and the Dockerfile doesn’t pass them).

### Split deploy (frontend and API on different URLs)

- Build the frontend with the API base URL:  
  `VITE_API_ORIGIN=https://your-api.onrender.com npm run build`  
  (no trailing slash).
- Deploy the built `frontend/dist` as a **static site** (e.g. Render static site, Vercel, Netlify). Ensure the API service allows CORS (already configured in `api/main.py` with `allow_origins=["*"]` for development; tighten for production if needed).
- In `.env.example` (and deploy docs), document:  
  "For split deploy: set VITE_API_ORIGIN to your API URL (e.g. https://your-api.onrender.com) when building the frontend."

---

## 4. Implementation checklist

- [x] **Plan document** (this file): endpoint audit and root causes.
- [x] **useFetchWithRetry**: On 404, if response body is not JSON (e.g. Content-Type is text/html), set a clear "API unreachable or endpoint not found…" message (see `API_UNREACHABLE_404_MESSAGE` and `normalizeError(..., contentType)`).
- [x] **.env.example**: Clarified that for Docker same-origin deploy `VITE_API_ORIGIN` can be left unset; for split deploy set it at build time.
- [x] **DEPLOYMENT_STEP_BY_STEP.md**: Added "Tabs show Not found or API unreachable" troubleshooting: same-origin vs split deploy, optional routers, link to this plan.
- [x] **API health on load**: Terminal shell calls `GET /health` on mount; if it fails, shows a banner with "API unreachable" and Retry button.

---

## 5. Optional routers and dependencies

- **Orchestrator**: Used by Automation tab. Depends on orchestrator module (and optionally `schedule`, `stable-baselines3`). If import fails, Automation tab will 404.
- **Screener**: Used by Screening tab. If `screener_api` fails to load, Screening tab’s "Run" will 404; sector list still comes from company API.
- **News**: Mounted under `/api/v1/data/news`. If `news_api` import fails, News tab will 404. FINNHUB_API_KEY is for content, not for route registration.
- **Automation** (separate from Orchestrator): `/api/v1/automation/*`; if loaded, Automation tab could use it; currently the panel uses Orchestrator endpoints.

After implementing the checklist above, the next build/deploy should have:
- Correct API base URL in split deploy (VITE_API_ORIGIN).
- Clearer "not found" messaging when the API is unreachable or returns HTML.
- Documentation so operators can fix "tabs not found" by checking origin and optional routers.
