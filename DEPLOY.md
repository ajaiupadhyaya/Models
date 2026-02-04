# Deploy the Terminal to the Web

This guide gets your Bloomberg-style terminal **live on the internet** so you (and others) can open a URL and sign in — no need to run anything on your computer. You pay for server space; the app runs 24/7 on the host.

**→ For exact step-by-step instructions (push to GitHub, then Railway / Render / Fly.io with every click and field described):** see [DEPLOYMENT_STEP_BY_STEP.md](DEPLOYMENT_STEP_BY_STEP.md).

---

## Goal

- **One URL** (e.g. `https://your-app.railway.app` or `https://terminal.yourdomain.com`)
- **Sign in** on the web and use the terminal from any browser
- **No local launch** — the app runs on the provider’s servers

---

## Prerequisites

- Code in a Git repo (GitHub, GitLab, etc.)
- API keys and secrets (FRED, Alpha Vantage, OpenAI, etc.) to set in the host’s environment
- A paid or free tier account on one of: **Railway**, **Render**, or **Fly.io** (or a VPS)

---

## Option 1: Railway

1. Go to [railway.app](https://railway.app) and sign in (e.g. with GitHub).
2. **New Project** → **Deploy from GitHub repo** → select your `Models` repo.
3. Railway detects the **Dockerfile** and builds the image. It sets **PORT** automatically.
4. In the service **Variables** tab, add (no `.env` file in repo):

   | Variable | Purpose |
   |----------|--------|
   | `FRED_API_KEY` | Macro data |
   | `ALPHA_VANTAGE_API_KEY` | Market data |
   | `OPENAI_API_KEY` | AI analysis (optional) |
   | `TERMINAL_USER` | Login username (if auth is enabled) |
   | `TERMINAL_PASSWORD` | Login password |
   | `AUTH_SECRET` | Secret for JWT/session (if auth is enabled) |

5. **Settings** → **Networking** → **Generate domain**. You get a URL like `https://your-app.railway.app`.
6. Open that URL in a browser. (If you added auth, you’ll see the sign-in page.)

**Cost:** Railway has a free tier; beyond that you pay for usage. See [railway.app/pricing](https://railway.app/pricing).

---

## Option 2: Render

1. Go to [render.com](https://render.com) and sign in (e.g. with GitHub).
2. **New** → **Web Service** → connect your `Models` repo.
3. Configure:
   - **Environment:** Docker  
   - **Build Command:** (leave default; Render uses the Dockerfile)  
   - **Start Command:** (leave default; Dockerfile CMD is used)
4. **Environment** tab: add the same variables as in the Railway table above (`FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `OPENAI_API_KEY`, `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET`).
5. **Create Web Service**. Render builds the image and assigns a URL like `https://your-service.onrender.com`.
6. Open that URL in a browser.

**Optional:** Use the [render.yaml](render.yaml) Blueprint in this repo so Render creates the web service from the repo (see [Render Blueprint Spec](https://render.com/docs/blueprint-spec)).

**Scheduled runs:** For automated daily analysis, run a worker or the API with the data scheduler enabled (see [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) § 3.2). Set `RUN_SCHEDULER=1` (or equivalent) if your start command supports it; register `automated_daily_analysis` as a daily job in `core/pipeline/data_scheduler.py`.

**Cost:** Free tier available; paid plans for always-on and more resources. See [render.com/pricing](https://render.com/pricing).

---

## Option 3: Fly.io

1. Install [flyctl](https://fly.io/docs/hub/installing-flyctl/) and sign in: `fly auth login`.
2. In your repo root (where the Dockerfile is):

   ```bash
   fly launch
   ```

   Accept defaults or name the app; Fly will create a `fly.toml` (you can commit it).
3. Set secrets (env vars visible only to the app):

   ```bash
   fly secrets set FRED_API_KEY=your_key
   fly secrets set ALPHA_VANTAGE_API_KEY=your_key
   fly secrets set OPENAI_API_KEY=your_key
   fly secrets set TERMINAL_USER=admin
   fly secrets set TERMINAL_PASSWORD=your_password
   fly secrets set AUTH_SECRET=random_secret_string
   ```

4. Deploy:

   ```bash
   fly deploy
   ```

5. Open the URL Fly prints (e.g. `https://your-app.fly.dev`).

**Cost:** Free tier; paid for more resources. See [fly.io/docs/pricing](https://fly.io/docs/pricing).

---

## Option 4: VPS (DigitalOcean, Linode, AWS EC2, etc.)

1. Create a Linux server (Ubuntu 22.04 or similar) and SSH in.
2. Install Docker and Docker Compose.
3. Clone your repo and create a `.env` file (or export env vars) with `FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `OPENAI_API_KEY`, `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET`.
4. From the repo root:

   ```bash
   docker build -t terminal-app .
   docker run -d --name terminal -p 80:8000 --env-file .env terminal-app
   ```

   (If your platform sets `PORT`, map it: e.g. `-p 80:${PORT:-8000}`.)
5. Point a domain to the server IP and optionally put nginx/Caddy in front for HTTPS (e.g. Let’s Encrypt).

---

## Environment variables (all options)

Set these on the host (never commit real values to Git):

| Variable | Required | Purpose |
|----------|----------|--------|
| `FRED_API_KEY` | Yes (for macro) | [Get key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `ALPHA_VANTAGE_API_KEY` | Yes (for market data) | [Get key](https://www.alphavantage.co/support/#api-key) |
| `OPENAI_API_KEY` | No | AI analysis |
| `TERMINAL_USER` | If auth enabled | Login username |
| `TERMINAL_PASSWORD` | If auth enabled | Login password |
| `AUTH_SECRET` | If auth enabled | Secret for signing tokens (use a long random string) |
| `PORT` | Set by host | Railway, Render, Fly set this; don’t set manually |

---

## After deploy

- Open the URL in a browser. If auth is implemented, you’ll see a sign-in page; otherwise the API/docs at `/docs` and `/health`.
- The app runs on the provider’s servers; you don’t need to keep your computer on.
- To update: push to your repo and trigger a redeploy (Railway/Render do this automatically; Fly: `fly deploy`).

---

## What you get when you deploy

The Docker image is a **single container** that includes the built React frontend and the FastAPI backend. Deploying it gives you **one URL** where:

- Users see a **sign-in page** (credentials from `TERMINAL_USER` / `TERMINAL_PASSWORD` or defaults).
- After signing in, they use the **Bloomberg-style terminal** (watchlist, charts, portfolio, AI assistant, etc.).
- The same origin serves both the UI and the API (`/api/...`, `/docs`, `/health`).

No separate frontend or backend to run; the app runs 24/7 on the host you choose.
