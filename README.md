# Models — Quant Terminal

A Bloomberg-terminal-style web app for personal quant research. FastAPI backend, React + D3 frontend, ships with backtesting, risk analytics, factor exposure, AI commentary, and paper trading — all behind a keyboard-driven command bar.

[![CI](https://github.com/ajaiupadhyaya/Models/actions/workflows/ci.yml/badge.svg)](https://github.com/ajaiupadhyaya/Models/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Demo in 30 seconds

Open the terminal and try these commands in the command bar (`/` or `Cmd+K` to focus):

| Command | What you see |
| --- | --- |
| `GP AAPL` | Candlestick chart with volume and indicators |
| `QUANT AAPL` | Factor exposure and quant models |
| `PORT` | Portfolio risk metrics and allocation |
| `BACKTEST AAPL` | Strategy backtest with equity curve |
| `AI why is AAPL moving?` | AI research assistant (needs API key) |

The command bar includes a **Try:** quick-start strip for first-time users. Panels degrade gracefully when optional provider keys are missing.

---

## What's in v1.0

| Module | Status |
| --- | --- |
| Primary Instrument (candlestick + volume, indicators, multi-timeframe) | ✅ |
| Quant (factor exposure, regime classification, walk-forward) | ✅ |
| Portfolio (risk metrics, stress scenarios, allocation) | ✅ |
| Backtest / AI (equity-only strategies + AI narrative) | ✅ |
| 11 bonus panels (News, Economic, Screening, Paper Trading, ...) | ✅ |
| D3 charts (candlestick, correlation matrix, factor heatmap, ...) | ✅ |
| Command bar (18 commands, slash/Cmd+K, history) | ✅ |
| Workspace persistence (localStorage) | ✅ |
| JWT auth on expensive routes | ✅ |
| Structured JSON logs + request IDs | ✅ |
| In-process scheduler (no Redis required) | ✅ |

Full scope and what's deferred to v2: [`docs/FEATURE_BACKLOG.md`](docs/FEATURE_BACKLOG.md).

---

## Try it free

Three free-tier deploy paths in [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md). The recommended one (Fly.io + Vercel + Supabase) is always-on at $0/mo typical:

```bash
# 1. Postgres — paste db/init.sql into Supabase SQL editor.
# 2. Backend — Fly.io:
fly launch --no-deploy --copy-config --name <your-app>
fly secrets set TERMINAL_USER=demo TERMINAL_PASSWORD=... AUTH_SECRET=... \
                DATABASE_URL=postgres://... CORS_ORIGINS=https://<your-frontend>.vercel.app
fly deploy
# 3. Frontend — Vercel:
#    Root: frontend/, set VITE_API_ORIGIN=https://<your-app>.fly.dev, deploy.
```

For a no-credit-card option, use the Render single-service path (also documented). Trade-off: free-tier cold starts.

---

## Local development

```bash
cp .env.example .env                              # set TERMINAL_USER/PASSWORD/AUTH_SECRET
pip install -r requirements-api.txt
(cd frontend && npm ci && npm run dev)            # http://localhost:5173
python -m uvicorn api.main:app --reload --port 8000
```

The Vite dev server proxies `/api/*` to the backend automatically. Optional API keys (FRED, Polygon, OpenAI, ...) are listed in `.env.example`; any left blank disable that feature without breaking the rest of the app.

Docker:
```bash
docker build -t models-terminal .
docker run --rm -p 8000:8000 --env-file .env models-terminal
```

---

## Architecture

```
┌────────────┐    HTTPS    ┌──────────────┐
│  React SPA │ ──────────▶ │  FastAPI app │ ─┬─▶ models/ + core/  (pure Python)
│  (Vite,    │   /api/*    │  19 routers  │  ├─▶ data providers (yfinance, FRED,
│   D3)      │             │  JWT auth    │  │   Polygon, FMP, EDGAR, NewsAPI,
└────────────┘             │  JSON logs   │  │   CoinGecko, Finnhub)
                           │  APScheduler │  └─▶ Postgres (Supabase / Neon)
                           └──────────────┘       — optional; app works keyless
```

- **`api/`** — 19 routers, request ID + JSON logging middleware, lifespan-managed scheduler.
- **`core/`** — data fetcher, provider adapters, backtesting engine, AI service, DB layer (graceful no-DB mode).
- **`models/`** — quant, risk, options, factors, ML, RL, NLP, sentiment, derivatives, time-series, valuation.
- **`frontend/`** — React 18, Vite, TypeScript, custom D3 charts, command bar, localStorage workspace.
- **`workers/`** — legacy Celery tasks kept for v2 reference; v1 uses `api/scheduler.py` (APScheduler in-process).

---

## Security

JWT (single-user MVP) protects state-mutating and CPU-heavy routes. See [`SECURITY.md`](SECURITY.md) for the full model, secret rotation, and disclosure email.

---

## Docs

- [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) — three free-tier deploy paths, verification checklist.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — common failure modes by stage.
- [`SECURITY.md`](SECURITY.md) — auth model, secret handling, disclosure.
- [`docs/FEATURE_BACKLOG.md`](docs/FEATURE_BACKLOG.md) — v1.0 scope + v2 roadmap.
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) — pre-tag verification checklist.

---

## License

MIT. Personal-use project — not investment advice, not for live trading.
