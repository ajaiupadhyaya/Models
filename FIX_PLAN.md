# Fix Plan: Site Loading & GitHub Actions Errors

This document outlines the plan to fix (1) features appearing unavailable on the deployed site and (2) GitHub Actions CI errors.

---

## 1. Site loading / “features unavailable”

### Root causes (and fixes)

| Cause | Fix |
|-------|-----|
| **API unreachable** (wrong origin, CORS, or service down) | Frontend checks `/health` on load; show “API unreachable” with retry. |
| **Auth required but not logged in** | When auth is configured on Render, user must log in. Ensure `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET` are set in Render env. |
| **API returns 5xx** (missing keys, bugs) | Backend already uses graceful fallbacks (see RUNTIME_FIXES). Ensure required env vars are set per RENDER_DEPLOY.md. |
| **SPA not served** (only API JSON at `/`) | Dockerfile builds frontend and copies `frontend/dist`; `api/main.py` serves SPA when `frontend/dist/index.html` exists. If build fails, Docker build fails. No change needed if build succeeds. |
| **Wrong `VITE_API_ORIGIN`** (split deploy mistake) | For single Web Service on Render, do **not** set `VITE_API_ORIGIN`. Doc in RENDER_DEPLOY.md. |

### Implemented

- **API reachability check**: Terminal (or shell) calls `/health` when the app loads. On failure, show a clear “API unreachable” message and a retry action instead of blank or generic errors.
- **Docs**: RENDER_DEPLOY.md and .env.example already describe required env vars. No change unless you add new vars.

### After deploy (Render)

1. Open `https://your-service.onrender.com/health` — should return `{"status":"healthy",...}`.
2. Open `https://your-service.onrender.com/info` — confirms loaded routers.
3. Set in Render → Environment: `TERMINAL_USER`, `TERMINAL_PASSWORD`, `AUTH_SECRET`, `FRED_API_KEY`, `ALPHA_VANTAGE_API_KEY` (and optional keys as in RENDER_DEPLOY.md).

---

## 2. GitHub Actions CI errors

### Jobs

- **backend**: `pytest tests/ -v --tb=short -x` on Python 3.12 with `requirements-ci.txt`. Fails if any test fails.
- **lint**: `ruff check config/ api/ core/`. Previously run with `|| true`, so lint never failed the job.
- **frontend**: `npm ci` + `npm run test` in `frontend/`. Already passing (24 tests).

### Fixes applied

1. **Lint**
   - **Remove `|| true`** so real lint failures fail the job.
   - **Ruff config**: Add config (e.g. in `pyproject.toml` or `.ruff.toml`) so the current codebase passes. Many existing issues are style (unused imports F401, import not at top E402, unused variables F841). Config ignores these for now so CI is green; you can fix them later with `ruff check --fix` and manual review.
   - Result: Lint job fails only on the rules we enforce (e.g. real errors), not on F401/E402/F841.

2. **Backend tests**
   - **Dependency fixes (verified):**
     - **websockets**: yfinance imports `websockets.asyncio.client`; `websockets==12.0` does not have that module. In `requirements-ci.txt`, use `websockets>=13.0` so CI and runtime work.
     - **httpx**: Starlette/FastAPI `TestClient` passes `app=` to httpx; httpx 0.28+ removed that parameter. In `requirements-ci.txt`, use `httpx>=0.25.0,<0.28.0` so tests run.
   - CI uses Python 3.12 and `requirements-ci.txt`. `SKIP_AI_VALIDATION=1` is set in workflow `env`. Optional deps (e.g. OpenAI, TensorFlow) are skipped or mocked when missing.

3. **Frontend tests**
   - No change; keep `npm ci` and `npm run test`.

---

## 3. Implementation checklist

- [x] Add API reachability check in frontend (e.g. in shell or wrapper) with “API unreachable” + retry.
- [x] Add Ruff config so `ruff check config/ api/ core/` passes (ignore F401, E402, F841, etc. for now).
- [x] Remove `|| true` from the lint step in `.github/workflows/ci.yml`.
- [ ] (Optional) Run `ruff check --fix` and fix remaining issues over time.
- [ ] (Optional) If a specific backend test fails in Actions, fix or skip that test and document in this file.

---

## 4. Quick verification (all verified locally)

- **Backend** (Python 3.12): `python3.12 -m venv .venv-ci && .venv-ci/bin/pip install -r requirements-ci.txt && SKIP_AI_VALIDATION=1 .venv-ci/bin/python -m pytest tests/ -v --tb=short` → **109 passed, 11 skipped**
- **Lint**: `ruff check config/ api/ core/` → **All checks passed**
- **Frontend**: `cd frontend && npm ci && npm run test` → **24 passed**
- **Render**
  - Open `/health` and `/info`; set env vars per RENDER_DEPLOY.md; log in and use one feature (e.g. quotes, charts).

---

## 5. Summary

- **Site**: API health check + clear “API unreachable” message; correct Render env (no `VITE_API_ORIGIN` for single service); SPA already served when `frontend/dist` exists.
- **CI**: Lint step now fails on real errors; Ruff config keeps current codebase passing; backend and frontend tests unchanged; fix or skip any failing test when it appears in Actions.
