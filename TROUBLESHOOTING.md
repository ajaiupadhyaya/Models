# Troubleshooting

Symptom → diagnosis → fix. Grouped by deploy stage.

---

## Boot / startup

### Container exits immediately after deploy

**Diagnose:** `fly logs` (Fly) or Render's runtime log. Look for the last line before exit.

- `ModuleNotFoundError: backend` → you're on an old Dockerfile. Pull latest; the CMD should be `api.main:app`, not `backend.main:app`.
- `alembic: error` → also old Dockerfile; the current build doesn't run migrations. Use `db/init.sql` instead.
- `ImportError: PyJWT is required` → `requirements-api.txt` didn't install. Rebuild without the Docker cache (`fly deploy --no-cache` / Render → Clear build cache & deploy).

### Health check fails but the process is up

- Render: free-tier services occasionally take >30 s to bind on cold start. The `HEALTHCHECK` `start-period` was bumped to 15 s; if you still see failures, raise it in the Dockerfile.
- Fly: confirm `internal_port = 8000` matches `PORT` env. If you changed `PORT`, you must also change the check.

### `yfinance` returns no data / "401 Client Error"

This is the Yahoo Finance bot block. The repo ships a `curl_cffi`-free shim at `core/yfinance_session.py`; if you see this, confirm the shim is imported before any other `yfinance` call (it already is in `api/main.py`).

---

## Auth

### Login returns 503 "Auth not configured"

`TERMINAL_USER` or `TERMINAL_PASSWORD` is unset. Set both:
```bash
fly secrets set TERMINAL_USER=demo TERMINAL_PASSWORD="$(openssl rand -base64 24)"
```

### Login returns 200 but `/api/v1/backtest/run` returns 401

Working as designed — the 5 expensive routers require a `Bearer` token. Pull the token from the login response and pass it:
```bash
curl -H "Authorization: Bearer $TOKEN" https://your-app/api/v1/backtest/...
```

### Frontend skips login screen entirely

`/api/auth/status` returned `configured: false`. The `AUTH_SECRET` is still the default placeholder. Set a real one:
```bash
fly secrets set AUTH_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
```

---

## CORS / split-host

### Browser shows `CORS error: Origin <vercel> not allowed`

The Fly backend doesn't have your Vercel domain in `CORS_ORIGINS`. Update:
```bash
fly secrets set CORS_ORIGINS="https://your-frontend.vercel.app"
fly deploy
```
For multiple origins, comma-separate. Setting `*` works for development but **disables** credentialed CORS automatically.

### Cookies / Authorization header dropped between frontend and backend

`allow_credentials` is only true when `CORS_ORIGINS` is an explicit list, not `*`. If you need credentialed CORS, list your exact origin(s).

---

## Database

### `psycopg2.OperationalError: FATAL: too many connections`

You're on Supabase free and used the **direct** (port 5432) URL instead of the **pooler** (port 6543). Switch to the pooler URL:
```
postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:6543/postgres
```

### App loads but every panel that needs DB shows empty

Expected when `DATABASE_URL` is unset — read helpers return `[]` / `None` by design. Either:
- Set `DATABASE_URL` + run `db/init.sql`, then enable `SCHEDULER_ENABLED=true` and wait for the first refresh tick, or
- Use endpoints that fetch live (yfinance) data — most charts work without the DB.

### `relation "ohlcv" does not exist`

You set `DATABASE_URL` but never ran `db/init.sql`. Paste it into your Supabase SQL Editor and re-run.

---

## Frontend

### Vite build fails on Vercel with `Cannot find module '@types/...`

Vercel's default install command is `npm install`, not `npm ci`. The `frontend/vercel.json` in this repo pins `installCommand: npm ci`. If you overrode it in the Vercel UI, revert.

### `apiBase.ts` resolves to `http://localhost:8000` in production

`VITE_API_ORIGIN` wasn't set at build time. Vite inlines env vars at build, not runtime — adding it to Vercel env vars *after* a deploy doesn't fix the already-built bundle. Trigger a new deploy after setting it.

---

## Scheduler

### `apscheduler not installed; scheduler disabled` in logs

`pip install -r requirements-api.txt` either failed or you're on an old image. Rebuild without cache. If only `requirements.txt` was rebuilt, APScheduler won't be present.

### Scheduled jobs never run on Fly

Fly auto-stops the machine when idle (the default in `fly.toml`). A stopped machine can't fire a scheduled job. Either:
- Set `min_machines_running = 1` (small monthly cost), or
- Trigger refresh manually via `POST /api/v1/data/refresh/{data_type}` (auth required).

---

## Logging / observability

### Log lines aren't JSON

Either you set `LOG_LEVEL=DEBUG` low enough to surface uvicorn's plain-text access log (we mute that to WARNING by default — re-check `api/logging_config.py`), or a hook ran `logging.basicConfig` after import. Search for `basicConfig` in your changes.

### `request_id` is always `-`

Logs emitted *before* `RequestIDMiddleware` runs (e.g. lifespan startup logs) get the default `-`. That's correct. Anything mid-request should have a real UUID.
