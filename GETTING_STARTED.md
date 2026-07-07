# Getting Started

This project is a Bloomberg-style quant terminal for personal research: FastAPI backend, React/Vite frontend, D3 charts, backtesting, risk analytics, AI commentary, paper trading, and optional data-provider integrations.

The old interactive launcher scripts were removed during the v1 cleanup. Use the direct commands below; they match the current CI and deployment docs.

## Prerequisites

- Python 3.12
- Node.js 20+
- Optional: Postgres connection string for persisted data
- Optional: provider keys for FRED, Polygon, IEX, Finnhub, NewsAPI, OpenAI, and Alpaca

## First-Time Setup

From the repo root:

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-api.txt
cd frontend
npm ci
```

For local experimentation, `.env.example` uses a placeholder `AUTH_SECRET`, which makes `/api/auth/status` return `configured: false` and lets the frontend skip login. For any deployed or shared environment, set `TERMINAL_USER`, `TERMINAL_PASSWORD`, and a real secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Run Locally

Use two terminals.

Backend:

```bash
source .venv/bin/activate
python -m uvicorn api.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to `http://127.0.0.1:8000`.

**Auth in local dev:** With the default placeholder `AUTH_SECRET` in `.env`, `/api/auth/status` returns `configured: false` and the frontend skips login. Set real `TERMINAL_USER`, `TERMINAL_PASSWORD`, and `AUTH_SECRET` for deployed or shared environments. The login page shows a **Fill demo credentials** button only in dev mode (`npm run dev`).

Useful backend URLs:

- `http://localhost:8000/health`
- `http://localhost:8000/info`
- `http://localhost:8000/docs`
- `http://localhost:8000/api/auth/status`

## First Terminal Commands

Try these in the command bar:

- `GP AAPL` - primary instrument chart
- `FA AAPL` - fundamentals
- `FLDS AAPL` - technical indicators
- `QUANT AAPL` - quant models and factor views
- `PORT` - portfolio/risk panel
- `BACKTEST AAPL` - strategy backtest
- `AI why is AAPL moving?` - AI assistant flow
- `HELP` or `?` - command list

Most panels work without paid keys. Blank provider keys disable those integrations gracefully.

## Test And Build

Backend CI gate:

```bash
python -m pytest tests/test_api_contract_routes.py tests/test_db_schema_contract.py -v --tb=short
python -m pytest tests/ -v --tb=short -x
ruff check config/ api/ core/ --output-format=concise
```

Frontend gate:

```bash
cd frontend
npm run test
npm run build
```

Live provider checks are skipped by default so CI stays deterministic. To run them intentionally:

```bash
RUN_LIVE_PROVIDER_TESTS=1 POLYGON_API_KEY=... IEX_API_KEY=... NEWSAPI_KEY=... \
  python -m pytest tests/test_data_providers.py -v
```

## Docker

```bash
docker build -t models-terminal .
docker run --rm -p 8000:8000 --env-file .env models-terminal
```

The production image builds the frontend and serves the SPA from the FastAPI app.

## Deploy

Use `DEPLOYMENT_GUIDE.md`. The recommended daily-driver path is Fly.io for the backend, Vercel for the frontend, and Supabase for Postgres. Render single-service remains the no-credit-card option, with cold starts.

## Troubleshooting

- Auth problems: check `TERMINAL_USER`, `TERMINAL_PASSWORD`, and `AUTH_SECRET`.
- CORS problems: set `CORS_ORIGINS` to the exact deployed frontend origin.
- Empty DB-backed panels: set `DATABASE_URL`, run `db/init.sql`, and optionally enable `SCHEDULER_ENABLED=true`.
- Provider failures: confirm the relevant API key is set, or leave it blank to disable that provider.

More detail lives in `TROUBLESHOOTING.md`, `API_DOCUMENTATION.md`, `ARCHITECTURE.md`, and `docs/FEATURE_BACKLOG.md`.
