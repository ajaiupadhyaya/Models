# Security

## Threat model — v1.0

Single-user MVP intended for personal / portfolio use, deployed publicly but addressed by one operator. Multi-user, RBAC, OAuth, secret scanning, and audit logging are deliberately deferred to v2 (see `docs/FEATURE_BACKLOG.md`).

## Authentication

- **Single-user JWT.** `POST /api/auth/login` with `TERMINAL_USER` / `TERMINAL_PASSWORD` returns a bearer token signed with `AUTH_SECRET` (HS256).
- **Token TTL** defaults to 60 minutes (`AUTH_TOKEN_EXPIRE_MINUTES`).
- **Frontend behavior**: if `GET /api/auth/status` returns `configured: false` (the `AUTH_SECRET` is the default placeholder), the login screen is skipped — *only* safe for local dev. Set a real `AUTH_SECRET` for any deployed environment.

## What is protected

Router-level `Depends(get_current_user)` is wired on the routes that mutate state or run expensive work:

| Router | Reason |
| --- | --- |
| `/api/v1/backtest` | CPU-heavy strategy runs |
| `/api/v1/ai` | Paid OpenAI calls |
| `/api/v1/paper-trading` | State-mutating order entry |
| `/api/v1/automation` | State-mutating rules / triggers |
| `/api/v1/predictions` | CPU-heavy model inference |

Read-only routes (data, quant, risk, news, equity, company, monitoring) are open behind the global rate limiter (`api/rate_limit.py`). If you want everything behind auth, add `dependencies=[Depends(get_current_user)]` to the corresponding `include_router(...)` calls in `api/main.py`.

## Secret management

- **Never commit `.env`.** `.env.example` is the only source-of-truth template.
- Generate `AUTH_SECRET` with:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(48))"
  ```
- **Rotate `AUTH_SECRET`** by updating the env var and redeploying. All outstanding tokens are invalidated immediately (signature mismatch).
- Provider keys (`POLYGON_API_KEY`, `FMP_API_KEY`, `OPENAI_API_KEY`, ...) are loaded lazily; missing keys disable only that provider — the rest of the app keeps working.

## Transport

- Production deploys MUST be HTTPS-terminated by the platform (Fly.io / Render / Vercel handle this automatically).
- `CORS_ORIGINS` must be set to the exact frontend origin(s) in production. Leaving it as `*` disables credentialed CORS automatically (see `api/main.py`).

## Rate limiting

- Global per-IP limiter on `/api/*` (in-memory, single-process). See `api/rate_limit.py`.
- **Limitation**: resets on restart; not shared across replicas. A Redis-backed limiter is on the v2 roadmap.

## Logging

- Structured JSON logs (`api/logging_config.py`).
- Every request carries an `X-Request-ID` header + `request_id` field in logs.
- Auth credentials are never logged; tokens are not logged.

## What is NOT in v1.0 (planned for v2)

- OAuth2 / multi-user / RBAC
- Per-route rate limits + quotas
- Redis-backed distributed rate limiter
- Security scanning in CI (Bandit, pip-audit, truffleHog)
- Audit log for state-mutating actions

## Reporting a vulnerability

Please email **ajaiupad@gmail.com** with subject prefix `[security] models` and a minimal reproduction. Do not open a public GitHub issue for security reports.
