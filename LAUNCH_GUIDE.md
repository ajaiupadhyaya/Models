# Launch Guide — Get the Project Up and Running

This is the **single authoritative guide** for running and deploying the financial terminal: local development, Docker, and **deploying to the web**.

---

## Deploy to the web (recommended for “final product”)

To have the terminal **accessible on the internet** (no need to run anything on your computer):

1. Push your code to a Git repo (e.g. GitHub).
2. Sign up for a host that runs Docker (e.g. **Railway**, **Render**, or **Fly.io** — you pay for server space).
3. Connect the repo and set environment variables (API keys, login credentials) in the host’s dashboard.
4. Deploy. The host builds the image (frontend + API in one container) and gives you a **public URL** (e.g. `https://your-app.railway.app`).
5. Open that URL in a browser and sign in. The same image serves the terminal UI and the API from one URL; the app runs 24/7 on their servers.

**→ Full steps:** [DEPLOY.md](DEPLOY.md) — Railway, Render, Fly.io, and VPS.  
**→ Step-by-step (exact clicks and values):** [DEPLOYMENT_STEP_BY_STEP.md](DEPLOYMENT_STEP_BY_STEP.md).  
**→ Next improvements (UI, security, reliability):** [TERMINAL_IMPROVEMENT_PLAN.md](TERMINAL_IMPROVEMENT_PLAN.md).

Local development (two terminals or Docker) is below for contributors; the **final product** is a live web app you (and others) can use from any browser.

---

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Node.js 18+** (for the frontend)
- **Docker** (optional, for API + Redis + PostgreSQL + Prometheus)

If you see **“externally-managed-environment”** when running `pip` on macOS, your default `python3` is likely Homebrew’s. To fix it for all Python projects:

**→ See [SYSTEM_PYTHON_FIX.md](SYSTEM_PYTHON_FIX.md).** Do that once, then return here.

---

## 1. One-time setup

### 1.1 Get the code

```bash
git clone https://github.com/ajaiupadhyaya/Models.git
cd Models
```

Or open your existing project root in a terminal.

### 1.2 Virtual environment and dependencies

**First time only** (or after deleting `venv`):

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Every time** you open a new terminal to work on the project:

```bash
cd /path/to/Models
source venv/bin/activate
```

**Optional — full API stack (LSTM/TensorFlow):**

```bash
pip install -r requirements-api.txt
```

**Optional — C++ quant library:** See [CPP_QUANT_GUIDE.md](CPP_QUANT_GUIDE.md):

```bash
./build_cpp.sh
```

### 1.3 Configuration

Copy the environment template and set your API keys. **This is required before first run** (and for Docker if you use `env_file`):

```bash
cp .env.example .env
```

Edit `.env` and set at least:

| Variable | Purpose |
|----------|--------|
| `FRED_API_KEY` | Macro/economic data — [get key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `ALPHA_VANTAGE_API_KEY` | Market data — [get key](https://www.alphavantage.co/support/#api-key) |
| `OPENAI_API_KEY` | AI analysis (optional but recommended) |
| `TERMINAL_USER` | Sign-in username (default: demo) |
| `TERMINAL_PASSWORD` | Sign-in password (default: demo) |
| `AUTH_SECRET` | JWT signing secret (set a long random string in production) |

All runtime settings are loaded from `.env`. Optional overrides (e.g. default WACC, chart theme) can go in `config/config.py`; see `config/config_example.py`. Primary config is `.env`.

### 1.4 Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 2. Run locally (development)

Use **two terminals**.

**Terminal 1 — API**

From the project root, with the venv activated:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or:

```bash
bash start-api.sh
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

- **API:** http://localhost:8000  
- **Docs:** http://localhost:8000/docs  
- **Health:** http://localhost:8000/health  

Leave this terminal running.

**Terminal 2 — Web UI**

```bash
cd /path/to/Models/frontend
npm run dev
```

When Vite is ready, open the URL it shows (usually **http://localhost:5173**) in your browser.

You should see the terminal with Watchlist, Primary Instrument chart, Portfolio & Strategies, and AI Assistant. The frontend **proxies `/api` to the backend** at `http://localhost:8000`, so the API must be running on port 8000.

---

## 3. Quick checks

**Backend:**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
```

**Frontend:** Load http://localhost:5173 and confirm panels load.

**Optional validation:**

```bash
python validate_environment.py
python -m pytest tests/ -v
```

---

## 4. Docker — one-command launch (recommended)

The Docker image builds the frontend and serves it with the API from **one URL**. No separate frontend process.

1. **Create `.env`** from `.env.example` and set at least:
   - `FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY` (and optionally `OPENAI_API_KEY`)
   - For sign-in: `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET` (or use defaults: demo / demo)

2. Build and start:

```bash
docker-compose up --build
```

3. Open **http://localhost:8000** in a browser. Sign in (e.g. demo / demo if you didn’t set credentials). The terminal UI and API are served from the same origin.

Optional: run `./launch.sh` to start Docker and open the browser automatically.

The same `docker-compose.yml` can start Redis, PostgreSQL, and Prometheus; see [DOCKER.md](DOCKER.md).

---

## 5. Production / deploy

- **Backend:** Run the API with all required env vars (from `.env` or your platform’s environment). Example:

  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
  ```

  Or run the API container (Docker) with `env_file` or equivalent so keys are present.

- **Frontend:** Build the static bundle, then serve it from the **same host** as the API so that `/api` is the same origin:

  ```bash
  cd frontend && npm run build
  ```

  Serve the `frontend/dist` directory with a reverse proxy (e.g. nginx or your platform’s static hosting) and **proxy `/api` to the FastAPI backend**. The app uses relative URLs (`/api/...`), so the frontend and API must be on the same origin (or you configure your proxy so `/api` goes to the backend).

- **Secrets:** Do not commit API keys. Use `.env` (ignored by git) or your platform’s secret/env configuration only.

---

## 6. Summary: two terminals to run the app

**Terminal 1 — API**

```bash
cd /path/to/Models
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Web UI**

```bash
cd /path/to/Models/frontend
npm run dev
```

Then open **http://localhost:5173**.

---

## 7. Where to go next

| Need | Document |
|------|----------|
| **Deploy to the web (live URL, no local run)** | [DEPLOY.md](DEPLOY.md) |
| Fix system Python (macOS, all projects) | [SYSTEM_PYTHON_FIX.md](SYSTEM_PYTHON_FIX.md) |
| Project overview, features, company analysis CLI | [README.md](README.md) |
| Full API reference | [API_DOCUMENTATION.md](API_DOCUMENTATION.md) |
| Architecture and backtest methodology | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Step-by-step user workflows | [WORKFLOWS.md](WORKFLOWS.md) |
| Docker / production | [DOCKER.md](DOCKER.md) |
| Company analysis (analyze_company.py) | [COMPANY_ANALYSIS_GUIDE.md](COMPANY_ANALYSIS_GUIDE.md) |
| C++ quant library | [CPP_QUANT_GUIDE.md](CPP_QUANT_GUIDE.md) |
| Project spec / requirements | [context.md](context.md) |

---

## 8. Troubleshooting

**“externally-managed-environment” when running pip (macOS)**  
Fix your system Python once: [SYSTEM_PYTHON_FIX.md](SYSTEM_PYTHON_FIX.md).

**“No module named 'pandas'” / import errors**  
Activate the venv and run `pip install -r requirements.txt`.

**Frontend shows “Backend not reachable”**  
Start the API first on port 8000. The frontend proxies `/api` to the backend.

**AI Assistant says “AI analysis unavailable”**  
Set `OPENAI_API_KEY` in `.env`.

**Docker: API returns 401, empty data, or “missing key”**  
Create `.env` from `.env.example`, set `FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY`, and optionally `OPENAI_API_KEY`. The API service uses `env_file: .env`; without it, keys are not loaded.

**Automation / orchestrator endpoints missing**  
Install the scheduler: `pip install schedule` (included in `requirements.txt`).

**Port 8000 or 5173 in use**  
Run the API on another port, e.g. `uvicorn api.main:app --port 8001`, and set the proxy target in `frontend/vite.config.ts` to `http://localhost:8001`.
