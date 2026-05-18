# Deployment Guide

Three free-tier paths, from cheapest to most resilient. Pick one.

| Path | Backend | Frontend | DB | Always-on? | Monthly cost |
| --- | --- | --- | --- | --- | --- |
| **A. Render single-service** | Render free | served by backend | Supabase free | No (cold starts) | $0 |
| **B. Fly + Vercel + Supabase** | Fly hobby | Vercel free | Supabase free | Yes (with CC on file) | $0 typical, ~$3 if you exceed allowances |
| **C. Render + Supabase** | Render free | served by backend | Supabase free | No | $0 |

**Recommended for daily-driver use: Path B.** Fly's free hobby tier keeps the backend warm and avoids the ~30-second wake-up that Render's free tier inflicts.

---

## Path B — Fly.io + Vercel + Supabase (recommended)

### 1. Postgres on Supabase

1. Create an account at <https://supabase.com> → New Project (free tier; choose a region near you).
2. Once provisioned, go to **Project Settings → Database** and copy the **Connection string (URI)**. It looks like:
   ```
   postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres
   ```
   Use the **pooler** URL (port 6543) — Supabase's free direct connection limit is tiny.
3. Open **SQL Editor** → paste the contents of `db/init.sql` from this repo → **Run**.

### 2. Backend on Fly.io

1. Install `flyctl` ([docs](https://fly.io/docs/hands-on/install-flyctl/)).
2. `fly auth signup` (CC required for hobby tier, but typical usage stays in the free allowance).
3. From the repo root:
   ```bash
   fly launch --no-deploy --copy-config --name <your-app-name>
   ```
   This consumes the existing `fly.toml`. Pick a region near you (e.g. `iad`, `lhr`).
4. Set secrets:
   ```bash
   fly secrets set \
     TERMINAL_USER=demo \
     TERMINAL_PASSWORD="$(openssl rand -base64 24)" \
     AUTH_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')" \
     DATABASE_URL="postgresql://..." \
     CORS_ORIGINS="https://<your-frontend>.vercel.app" \
     FRED_API_KEY="..." \
     OPENAI_API_KEY="..."   # add whichever provider keys you want
   ```
5. Deploy:
   ```bash
   fly deploy
   ```
6. Verify:
   ```bash
   curl https://<your-app-name>.fly.dev/health
   curl https://<your-app-name>.fly.dev/api/auth/status
   ```

### 3. Frontend on Vercel

1. Install Vercel CLI (`npm i -g vercel`) or import the repo from <https://vercel.com/new>.
2. **Root directory**: `frontend`.
3. **Build & Output**: auto-detected (Vite). The committed `frontend/vercel.json` handles SPA routing.
4. **Environment variables** → add `VITE_API_ORIGIN` = `https://<your-fly-app>.fly.dev`.
5. Deploy. The first deploy returns a `*.vercel.app` URL — copy it back into the Fly `CORS_ORIGINS` secret:
   ```bash
   fly secrets set CORS_ORIGINS="https://<your-vercel-app>.vercel.app"
   ```

### 4. Smoke test

- Open the Vercel URL → you should land on the login screen.
- Sign in with `TERMINAL_USER` / `TERMINAL_PASSWORD`.
- Run `GP AAPL` in the command bar → candlestick renders.
- Open the **DataStatus** panel — DB-backed rows show as empty until the first scheduled refresh runs.

### 5. Enable the scheduler (optional)

If you want background ingestion of OHLCV / macro / news:
```bash
fly secrets set SCHEDULER_ENABLED=true
fly deploy
```
This runs APScheduler in-process. Note: if Fly auto-stops the machine when idle, schedules won't fire. For real always-on behavior, set `min_machines_running = 1` in `fly.toml` and redeploy.

---

## Path A — Render single-service (zero-CC)

1. SQL setup: same as Path B step 1, but on Supabase or Neon (`https://neon.tech`). Both free, no CC needed.
2. In Render → **New + → Blueprint** → connect this repo. Render detects `render.yaml`.
3. After the service spins up, open **Environment** and fill the `sync: false` secrets (TERMINAL_USER, TERMINAL_PASSWORD, AUTH_SECRET, CORS_ORIGINS=`https://<your-service>.onrender.com`, DATABASE_URL, any provider keys).
4. Trigger a manual redeploy.
5. Verify `/health`, `/docs`, and `/api/auth/status`.

Caveats:
- Free tier sleeps after 15 min of inactivity (first request after sleep takes ~30 s).
- 512 MB RAM ceiling — the heavy ML imports in `requirements.txt` may push you over; the Dockerfile already swallows install errors for optional deps to keep cold-start lean.

---

## Local development

```bash
cp .env.example .env             # fill TERMINAL_USER/PASSWORD/AUTH_SECRET at minimum
pip install -r requirements-api.txt
(cd frontend && npm ci && npm run dev)
python -m uvicorn api.main:app --reload --port 8000
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`, so the dev server at `http://localhost:5173` Just Works.

---

## Verification checklist (any path)

- [ ] `GET /health` returns `{"status":"healthy", ...}` with `X-Request-ID` header
- [ ] `GET /api/auth/status` returns `{"configured": true}` (false means `AUTH_SECRET` is still the placeholder)
- [ ] `POST /api/auth/login` with valid creds returns a token
- [ ] `POST /api/v1/backtest/...` without a token returns **401** (proves JWT wiring works)
- [ ] Same call with `Authorization: Bearer <token>` returns 200
- [ ] Frontend loads, login works, `GP AAPL` shows a chart
- [ ] Logs are JSON lines with `request_id`, `path`, `status`, `duration_ms`

If any step fails, see `TROUBLESHOOTING.md`.
